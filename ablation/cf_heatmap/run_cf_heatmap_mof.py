#!/usr/bin/env python
"""
CF back-propagation heatmaps — MoF-Models-7B (LLaVA-family, CLIP+DINO dual tower)
==================================================================================
对每张采样的 STA 攻击图像：
  * 一次带梯度前向（prefill），捕获 CLIP/DINO 视觉塔输出（hook）+ 层 25 的
    MLP down-proj 输入（直接 forward 返回 3 元组）
  * 用可微闭式 kappa 算 LAYER_IDX 层每个图像 token 的冲突（1152 = 576 CLIP + 576 DINO）
  * GradCAM：S = mean(图像 token kappa)，S.backward() → CLIP 视觉塔输出梯度 → saliency
  * 无梯度生成答案（对/错标注）
  * 每张存一个 NPZ

Usage: python ablation/cf_heatmap/run_cf_heatmap_mof.py [N_SAMPLES]

与 Qwen 模板（run_cf_heatmap_qwen.py）的差异：
  [M1] 直接调用 model(input_ids=..., images=...) 的裸 forward（grad 开启），返回
       3 元组 (CausalLMOutputWithPast, hidden_states, input_down_proj)；
       不用 generate 做前向（transformers generate 被 @no_grad 装饰）。
  [M2] 必须绕过 CLIP/DINO 视觉塔 wrapper 的 @torch.no_grad()：
       model.vision_tower.forward = model.vision_tower.forward.__wrapped__（functools.wraps），
       dino 同理；否则梯度在视觉塔处被截断，pixel 到视觉塔不可导。
  [M3] 图像 token = 单个 -200 placeholder 展开成的 1152 个连续 token
       （[p, p+1152)，偶数位 CLIP、奇数位 DINO，均 raster）。kappa_map 取两者平均。
  [M4] 视觉 hook 目标 model.vision_tower（CLIP，输出 [1,576,1024] = 24x24 raster）；
       gradcam_map 用 CLIP 的 saliency。
  [M5] 输入图先补边成 336 正方形（保持长宽比，不拉伸、不裁剪），24x24 网格覆盖；
       npz 的 image_path 指向保存的补边后图片。
"""
import sys
import os

_USER_SITES = [
    os.path.expanduser(p) for p in (
        "~/.local/lib/python3.10/site-packages",
        "~/.local/lib/python3.9/site-packages",
        "~/.local/lib/python3.8/site-packages",
    )
]
for p in _USER_SITES:
    while p in sys.path:
        sys.path.remove(p)

_IJAR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_IJAR, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image as PILImage

from myllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from myllava.conversation import conv_templates, SeparatorStyle
from myllava.model.builder import load_pretrained_model
from myllava.utils import disable_torch_init
from myllava.mm_utils import tokenizer_image_token

from diff_cf import differentiable_kappa
from token_mapping import values_to_grid
from gradcam import gradcam_saliency

MODEL_PATH = "/data/public/models/MoF_Models/MoF_Models/MoF_Models"
CONV_MODE = "llava_v1"
LAYER_IDX = 25
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/liuxiaofei/MMVP/scripts/weight"
WEIGHT_PATTERN = "LLama_MLP_layer_{layer}_down_proj_weight.pth"
RESIZE_SQUARE = 336
DEVICE = os.environ.get("CF_GPU", "cuda:1")

STA_CSV = os.path.join(_IJAR, "datasets/STA/sta_mini.csv")
STA_IMAGE_DIR = os.path.join(_IJAR, "datasets/STA/images")
NPZ_DIR = os.path.join(_IJAR, "results/cf_heatmap/npz/mof")
IMAGES_DIR = os.path.join(_IJAR, "results/cf_heatmap/images/mof")
N_SAMPLES = 4
RANDOM_SEED = 42

NUM_PATCHES = (RESIZE_SQUARE // 14) ** 2   # 336 -> 24x24 = 576


def pad_to_square(img, size, fill=(128, 128, 128)):
    """Scale-to-fit + center pad 到 size x size 正方形，保持长宽比不拉伸。"""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), PILImage.BICUBIC)
    canvas = PILImage.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def build_prompt(model, question):
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_SAMPLES
    print("=" * 66)
    print(f"CF heatmap extraction — MoF-Models-7B  (N={n}, layer L{LAYER_IDX})")
    print("=" * 66)
    os.makedirs(NPZ_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, "llava", device_map=DEVICE, device=DEVICE)
    # [M2] 绕过视觉塔 wrapper 的 @torch.no_grad()
    model.vision_tower.forward = model.vision_tower.forward.__wrapped__
    model.dino_tower.forward = model.dino_tower.forward.__wrapped__
    model.requires_grad_(False)   # 只需 images 梯度
    print(f"  model loaded (fp16) on {DEVICE}; no_grad bypass applied to CLIP+DINO towers")

    wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=LAYER_IDX))
    B = torch.load(wp, map_location=DEVICE).detach().float().requires_grad_(False)  # [J, K]
    print(f"  down-proj weight L{LAYER_IDX}: {tuple(B.shape)}")

    df = pd.read_csv(STA_CSV).sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="samples"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        gt = str(row["answer"]).strip().lower()[:1]
        image_path = os.path.join(STA_IMAGE_DIR, image_id)

        # [M5] 补边成 336 正方形（保持长宽比），存下模型实际看到的图
        pil_img = pad_to_square(PILImage.open(image_path).convert("RGB"), RESIZE_SQUARE)
        stem = os.path.splitext(image_id)[0]
        seen_path = os.path.join(IMAGES_DIR, f"{stem}.png")
        pil_img.save(seen_path)
        image_tensor = image_processor.preprocess(
            pil_img, return_tensors="pt")["pixel_values"][0]   # [3,336,336] fp32 CPU
        images = image_tensor.unsqueeze(0).half().to(DEVICE).requires_grad_(True)

        prompt = build_prompt(model, question)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)
        attention_mask = torch.ones_like(input_ids)

        vision_out = {}

        def clip_hook(_m, _inp, out):
            out.retain_grad()
            vision_out["clip"] = out
        hclip = model.vision_tower.register_forward_hook(clip_hook)

        def dino_hook(_m, _inp, out):
            out.retain_grad()
            vision_out["dino"] = out
        hdino = model.dino_tower.register_forward_hook(dino_hook)

        try:
            _outputs, _hidden, all_down_proj = model(
                input_ids=input_ids, attention_mask=attention_mask, images=images)
        finally:
            hclip.remove()
            hdino.remove()

        # [M3] 图像 token 块 = 单个 -200 展开成的 1152 个连续位置
        p = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().item()
        X_img = all_down_proj[LAYER_IDX][0][p:p + 2 * NUM_PATCHES]   # [1152, K]
        kappa = differentiable_kappa(X_img.t().contiguous(), B)      # [1152]
        grid = (1, NUM_PATCHES // 24, NUM_PATCHES // 24)             # 24x24
        clip_k = values_to_grid(kappa[0::2], grid)
        dino_k = values_to_grid(kappa[1::2], grid)
        kappa_map = (clip_k + dino_k) / 2.0
        kappa_map = kappa_map.detach()

        # GradCAM：S = mean(图像 token kappa)，反传到 CLIP 视觉塔输出
        s = kappa.mean()
        s.backward()
        feat = vision_out["clip"]                                     # [1,576,1024]
        cam = gradcam_saliency(feat, feat.grad)                       # [1,576]
        cam_map = values_to_grid(cam, grid).detach()

        # 无梯度答案
        with torch.no_grad():
            output_ids, *_ = model.generate(
                input_ids, images=images, max_new_tokens=64, do_sample=False,
                output_hidden_states=True)
        gen = output_ids[0][input_ids.shape[1]:]
        output_text = tokenizer.batch_decode(
            gen.unsqueeze(0), skip_special_tokens=True)[0].strip()
        pred = output_text.lower()[:1] if output_text else ""

        out_path = os.path.join(NPZ_DIR, f"{stem}.npz")
        np.savez_compressed(
            out_path,
            image_path=seen_path,
            image_id=image_id,
            question=question,
            ground_truth=gt,
            model_output=output_text,
            kappa_map=kappa_map.cpu().numpy(),
            gradcam_map=cam_map.cpu().numpy(),
            mean_kappa=float(kappa.mean().item()),
            model="mof",
        )
        print(f"  saved {os.path.basename(out_path)}  "
              f"pred={pred} gt={gt} {'OK' if pred == gt else 'XX'}  "
              f"mean_kappa={float(kappa.mean().item()):.5f}")

    print(f"\nDone. NPZ files in {NPZ_DIR}")


if __name__ == "__main__":
    main()
