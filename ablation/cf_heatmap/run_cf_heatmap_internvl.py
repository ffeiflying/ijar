#!/usr/bin/env python
"""
CF back-propagation heatmaps — InternVL2.5-8B
==============================================
对每张采样的 STA 攻击图像：
  * 一次带梯度前向（prefill），捕获 vision 输出（hook model.mlp1）+ 层 20 的
    MLP down-proj 输入（hook feed_forward）
  * 用可微闭式 kappa 算 LAYER_IDX 层每个图像 token 的冲突
  * GradCAM：S = mean(图像 token kappa)，S.backward() → mlp1 输出梯度 → saliency
  * 无梯度生成答案（对/错标注）
  * 每张存一个 NPZ

Usage: python ablation/cf_heatmap/run_cf_heatmap_internvl.py [N_SAMPLES]

与 Qwen 模板（run_cf_heatmap_qwen.py）的差异：
  [I1] 不直接用 model.chat 的返回值（其 all_input_down_proj 有 bug：flag 恒 0，
       只保留最后一层 layer31，见 modeling_internlm2.py InternLM2Model.forward）。
       改用 forward hook 抓 layer20 feed_forward 的 (down_proj, hidden_states)[1]。
  [I2] 视觉 hook 目标为 model.mlp1（输出 [1,256,4096]，16x16 行主序 merged token），
       与 256 个图像 token 一一对应；不是 model.vision_model（那是 merge 前的
       [1,1025,1024]）。
  [I3] 图像 token = input_ids 中 == IMG_CONTEXT(92546) 的位置，连续 256 个；
       需自行设置 model.img_context_token_id。
  [I4] 输入图先补边成 448 正方形（保持长宽比，不拉伸），16x16 网格覆盖该方图；
       npz 的 image_path 指向保存的补边后图片，保证热力图叠加到模型实际看到的图。
  [I5] 模型 bf16 加载（8B）；model.requires_grad_(False) 仅 pixel_values 需要梯度。
  [I6] 答案用 model.chat()（其内部 generate 已 no_grad），与 selectlayer 一致。
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

# InternVL 远程代码的 snapshot 目录（含 conversation.py），供 get_conv_template 使用
_SNAP = "/data/public/models/InternVL/InternVL2_5-8B/models--OpenGVLab--InternVL2_5-8B/snapshots/e9e4c0dc1db56bfab10458671519b7fa3dd29463"
if _SNAP not in sys.path:
    sys.path.insert(0, _SNAP)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image as PILImage
from transformers import AutoModel, AutoTokenizer
from conversation import get_conv_template

from diff_cf import differentiable_kappa
from token_mapping import image_token_indices, values_to_grid
from gradcam import gradcam_saliency

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
MODEL_CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
LAYER_IDX = 20
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/InternVL/weight"
WEIGHT_PATTERN = "internvl_MLP_Layer{layer}_down_proj.pth"
RESIZE_SQUARE = 448
DEVICE = os.environ.get("CF_GPU", "cuda:1")

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

STA_CSV = os.path.join(_IJAR, "datasets/STA/sta_mini.csv")
STA_IMAGE_DIR = os.path.join(_IJAR, "datasets/STA/images")
NPZ_DIR = os.path.join(_IJAR, "results/cf_heatmap/npz/internvl")
IMAGES_DIR = os.path.join(_IJAR, "results/cf_heatmap/images/internvl")
N_SAMPLES = 4
RANDOM_SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pad_to_square(img, size, fill=(114, 114, 114)):
    """Scale-to-fit + center pad 到 size x size 正方形，保持长宽比不拉伸。"""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), PILImage.BICUBIC)
    canvas = PILImage.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def build_inputs(model, tokenizer, question):
    """复刻 model.chat 的输入构造（internvl2_5 模板 + <img>256*IMG_CONTEXT</img>）。"""
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    if "<image>" not in question:
        question = "<image>\n" + question
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token + IMG_END_TOKEN
    query = query.replace("<image>", image_tokens, 1)
    model_inputs = tokenizer(query, return_tensors="pt")
    return (model_inputs["input_ids"].to(model.device),
            model_inputs["attention_mask"].to(model.device))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_SAMPLES
    print("=" * 66)
    print(f"CF heatmap extraction — InternVL2.5-8B  (N={n}, layer L{LAYER_IDX})")
    print("=" * 66)
    os.makedirs(NPZ_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True,
        cache_dir=MODEL_CACHE).eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False, cache_dir=MODEL_CACHE)
    model.requires_grad_(False)   # [I5] 只需 pixel_values 梯度
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    print(f"  model loaded (bf16) on {DEVICE}; img_context_token_id={img_context_token_id}")

    wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=LAYER_IDX))
    B = torch.load(wp, map_location=DEVICE).detach().float().requires_grad_(False)  # [J, K]
    print(f"  down-proj weight L{LAYER_IDX}: {tuple(B.shape)}")

    df = pd.read_csv(STA_CSV).sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="samples"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        gt = str(row["answer"]).strip().lower()[:1]
        image_path = os.path.join(STA_IMAGE_DIR, image_id)

        # [I4] 补边成 448 正方形（保持长宽比），存下模型实际看到的图
        pil_img = pad_to_square(PILImage.open(image_path).convert("RGB"), RESIZE_SQUARE)
        stem = os.path.splitext(image_id)[0]
        seen_path = os.path.join(IMAGES_DIR, f"{stem}.png")
        pil_img.save(seen_path)
        pixel_values = build_transform(pil_img).unsqueeze(0).to(torch.bfloat16).to(DEVICE).requires_grad_(True)  # [I5] 视觉输出需在梯度图上

        input_ids, attention_mask = build_inputs(model, tokenizer, question)

        vision_out = {}

        def vision_hook(_m, _inp, out):
            out.retain_grad()
            vision_out["feat"] = out
        vhook = model.mlp1.register_forward_hook(vision_hook)

        captured = {}

        def layer_hook(_m, _inp, out):
            captured["downproj_in"] = out[1]      # [1, seq, intermediate]
        lhook = model.language_model.model.layers[LAYER_IDX].feed_forward.register_forward_hook(layer_hook)

        try:
            with torch.enable_grad():       # [I7] 本 fork 的 language_model.forward 返回 2 元组，
                #      chat wrapper 的 `outputs.logits` 会崩；绕开它手动拼 prefill。
                #      extract_feature 内部跑 vision_model + mlp1（mlp1 hook 抓 GradCAM 目标）。
                vit_embeds = model.extract_feature(pixel_values)      # [1, 256, C]
                input_embeds = model.language_model.get_input_embeddings()(input_ids)
                B_, N_, C_ = input_embeds.shape
                input_embeds = input_embeds.reshape(B_ * N_, C_)
                sel = (input_ids.reshape(-1) == img_context_token_id)
                assert sel.sum().item() == vit_embeds.shape[1]
                input_embeds[sel] = vit_embeds.reshape(-1, C_)
                input_embeds = input_embeds.reshape(B_, N_, C_)
                model.language_model(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            vhook.remove()
            lhook.remove()

        # 逐图像 token 的 kappa（LAYER_IDX 层，可微）
        img_idx = image_token_indices(input_ids[0], img_context_token_id)
        X_img = captured["downproj_in"][0][img_idx]              # [256, K]
        kappa = differentiable_kappa(X_img.t().contiguous(), B)  # [256]
        grid = (1, RESIZE_SQUARE // 28, RESIZE_SQUARE // 28)     # 448 -> 16x16
        kappa_map = values_to_grid(kappa, grid).detach()

        # GradCAM：S = mean(图像 token kappa)，反传到 mlp1 输出
        s = kappa.mean()
        s.backward()
        feat = vision_out["feat"]                                 # [1,256,C]
        cam = gradcam_saliency(feat, feat.grad)                   # [1,256]
        cam_map = values_to_grid(cam, grid).detach()

        # [I6] 无梯度答案（chat 内部 generate 已 no_grad）
        with torch.no_grad():
            response, *_ = model.chat(
                tokenizer, pixel_values, question,
                dict(max_new_tokens=64, do_sample=False))
        output_text = response.strip()
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
            model="internvl",
        )
        print(f"  saved {os.path.basename(out_path)}  "
              f"pred={pred} gt={gt} {'OK' if pred == gt else 'XX'}  "
              f"mean_kappa={float(kappa.mean().item()):.5f}")

    print(f"\nDone. NPZ files in {NPZ_DIR}")


def build_transform(img):
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode
    t = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return t(img)


if __name__ == "__main__":
    main()
