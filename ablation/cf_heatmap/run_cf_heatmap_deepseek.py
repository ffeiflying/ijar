#!/usr/bin/env python
"""
CF back-propagation heatmaps — DeepSeek-VL2-Tiny
================================================
对每张采样的 STA 攻击图像：
  * 一次带梯度前向（prefill）：enable_grad 下裸调 vl_gpt.prepare_inputs_embeds +
    vl_gpt.language.forward(return_dict=True, use_cache=False)，返回
    (CausalLMOutputWithPast, all_input_down_proj[12])
  * 用可微闭式 kappa 算 LAYER_IDX=1 层共享专家 down-proj 输入上每个图像 token 的冲突
    （X = all_input_down_proj[1][0][图像token] [.,1792]；B = shared_experts.down_proj
    权重 [1280,1792]）
  * GradCAM：S = mean(图像 token kappa)，S.backward() → vl_gpt.projector 输出
    （[N_tiles,196,1280]，14x14/图块）梯度 → saliency
  * 无梯度生成答案（对/错标注）
  * 每张存一个 NPZ

Usage: python ablation/cf_heatmap/run_cf_heatmap_deepseek.py [N_SAMPLES]

与 Qwen 模板（run_cf_heatmap_qwen.py）的差异：
  [D1] generate 被 @no_grad 装饰，所以带梯度前向必须裸调 language.forward：
       with torch.enable_grad(): inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs);
       outputs = vl_gpt.language.forward(input_ids=None, inputs_embeds=..., attention_mask=...,
       use_cache=False, return_dict=True)。
  [D2] 视觉 hook 目标为 vl_gpt.projector 输出 images_embeds [batch_all_tiles,196,1280]
       （每个图块 14x14 = 196 个 token，全局视图在位置 0，局部图块从 1 开始、行主序），
       而不是 vl_gpt.vision 输出 [.,729,1152]（27x27，与图像 token 不对齐）。
  [D3] 图像 token = prepare_inputs.images_seq_mask[0] 标记的位置；顺序为
       [全局 14*15=210, 分隔 1, 局部 (th*14)*(tw*14+1)]（h=w=14）。kappa_map /
       gradcam_map 只取局部图块，网格 (th*14, tw*14)，与模型看到的 best-resolution
       补边画布对齐。
  [D4] 画布 = ImageOps.pad(pil_img, (tw*384, th*384), fill=image_mean*255)（保持长宽比
       不拉伸），与 processor 生成的 local_view 完全一致；存下该补边图作为 image_path。
  [D5] 答案用 vl_gpt.language.generate（无梯度），返回元组 outputs[0]=output_ids。
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
from PIL import ImageOps
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor
from diff_cf import differentiable_kappa
from gradcam import gradcam_saliency

MODEL_PATH = ("/data/public/models/deepseek/"
              "models--deepseek-ai--deepseek-vl2-tiny/"
              "snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa")
LAYER_IDX = 1
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/DeepSeek-VL2/weight"
WEIGHT_PATTERN = "DeepSeek_MLP_down_proj_{layer}_weight.pth"
DEVICE = os.environ.get("CF_GPU", "cuda:1")

STA_CSV = os.path.join(_IJAR, "datasets/STA/sta_mini.csv")
STA_IMAGE_DIR = os.path.join(_IJAR, "datasets/STA/images")
NPZ_DIR = os.path.join(_IJAR, "results/cf_heatmap/npz/deepseek")
IMAGES_DIR = os.path.join(_IJAR, "results/cf_heatmap/images/deepseek")
N_SAMPLES = 4
RANDOM_SEED = 42

# SigLIP 下采样后 h = w = 14（196 token/图块），全局视图 token = h*(w+1) = 210
_H = _W = 14


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_SAMPLES
    print("=" * 66)
    print(f"CF heatmap extraction — DeepSeek-VL2-Tiny  (N={n}, layer L{LAYER_IDX})")
    print("=" * 66)
    os.makedirs(NPZ_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16)
    vl_gpt = vl_gpt.eval().to(DEVICE)
    vl_gpt.requires_grad_(False)   # [D1] 只需 images 梯度
    print(f"  model loaded (bf16) on {DEVICE}; image_size={vl_chat_processor.image_size}")

    wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=LAYER_IDX))
    B = torch.load(wp, map_location=DEVICE).detach().float().requires_grad_(False)  # [J, K]
    print(f"  down-proj weight L{LAYER_IDX} (shared_experts): {tuple(B.shape)}")

    df = pd.read_csv(STA_CSV).sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="samples"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        gt = str(row["answer"]).strip().lower()[:1]
        image_path = os.path.join(STA_IMAGE_DIR, image_id)
        pil_img = PILImage.open(image_path).convert("RGB")

        conversation = [
            {"role": "<|User|>", "content": f"<image>\n{question}",
             "images": [pil_img]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=[pil_img],
            force_batchify=True, system_prompt="").to(DEVICE)
        prepare_inputs.images.requires_grad_(True)   # [D1]

        # [D4] 重建模型看到的 best-resolution 补边画布（与 local_view 一致）
        tw, th = prepare_inputs.images_spatial_crop[0, 0]
        canvas = ImageOps.pad(
            pil_img,
            (int(tw) * vl_chat_processor.image_size,
             int(th) * vl_chat_processor.image_size),
            color=tuple(int(x * 255) for x in vl_chat_processor.image_transform.mean))
        stem = os.path.splitext(os.path.basename(image_path))[0]
        seen_path = os.path.join(IMAGES_DIR, f"{stem}.png")
        canvas.save(seen_path)
        print(f"  [{image_id}] tiles={int(tw)}x{int(th)} canvas={canvas.size}")

        projector_out = {}

        def proj_hook(_m, _inp, out):
            out.retain_grad()
            projector_out["feat"] = out
        hproj = vl_gpt.projector.register_forward_hook(proj_hook)

        try:
            with torch.enable_grad():                      # [D1]
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                outputs = vl_gpt.language.forward(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            hproj.remove()

        _lm_out, all_input_down_proj = outputs             # [12][1, seq, K]

        # [D3] 图像 token（含全局+分隔+局部），顺序固定
        img_idx = torch.argwhere(prepare_inputs.images_seq_mask[0]).squeeze(1)
        X_img = all_input_down_proj[LAYER_IDX][0][img_idx]   # [n_img, 1792]
        kappa = differentiable_kappa(X_img.t().contiguous(), B)   # [n_img]

        n_global = _H * (_W + 1)                     # 210
        local_row_tok = int(tw) * _W + 1             # tw*14 + 1（每行含换行 token）
        local_rows = int(th) * _H                    # th*14
        local_k = kappa[n_global + 1: n_global + 1 + local_rows * local_row_tok]
        kappa_map = local_k.reshape(local_rows, local_row_tok)[:, :int(tw) * _W]
        kappa_map = kappa_map.detach()

        # GradCAM：S = mean(图像 token kappa)，反传到 projector 输出
        s = kappa.mean()
        s.backward()
        feat = projector_out["feat"]                         # [1+tw*th, 196, 1280]
        cam = gradcam_saliency(feat, feat.grad)              # [1+tw*th, 196]
        cam_local = cam[1: 1 + int(tw) * int(th)]            # 局部图块，行主序
        cam_map = cam_local.reshape(int(th), int(tw), _H, _W).permute(
            0, 2, 1, 3).reshape(local_rows, int(tw) * _W).detach()

        # [D5] 无梯度答案（复用 grad pass 的 inputs_embeds）
        with torch.no_grad():
            gen_out = vl_gpt.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=64,
                do_sample=False,
                use_cache=True,
                output_hidden_states=True,
            )
        output_ids = gen_out[0]
        output_text = tokenizer.decode(
            output_ids[0].cpu().tolist(), skip_special_tokens=False)
        for mark in ("<|_eos_id|>", "<｜end▁of▁sentence｜>"):
            output_text = output_text.replace(mark, "")
        output_text = output_text.strip()
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
            model="deepseek",
        )
        print(f"  saved {os.path.basename(out_path)}  "
              f"pred={pred} gt={gt} {'OK' if pred == gt else 'XX'}  "
              f"mean_kappa={float(kappa.mean().item()):.5f}")

    print(f"\nDone. NPZ files in {NPZ_DIR}")


if __name__ == "__main__":
    main()
