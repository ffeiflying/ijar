#!/usr/bin/env python
"""
CF back-propagation heatmaps — Qwen2.5-VL-7B
=============================================
对每张采样的 STA 攻击图像：
  * 一次带梯度前向（prefill），捕获视觉塔输出（hook）+ 逐层 down-proj 输入
  * 用可微闭式 kappa 算 LAYER_IDX 层每个图像 token 的冲突
  * GradCAM：S = mean(图像 token kappa)，S.backward() → 视觉塔输出梯度 → saliency
  * 无梯度 greedy 生成答案（用于对/错标注）
  * 每张存一个 NPZ

Usage: python ablation/cf_heatmap/run_cf_heatmap_qwen.py [N_SAMPLES]

DEVIATIONS FROM THE BRIEF (documented in .superpowers/sdd/task-4-report.md):
  [G1] 输入图像不做任何预拉伸：把原始文件路径直接交给 processor，走其 native
       动态分辨率（aspect-preserving smart_resize）。原因：brief 的"448px → 16x16"
       建立在两个错误假设上——(a) grid_thw 已是 merge 后单位（实际是 patch 单位，
       需再除以 spatial_merge_size=2）；(b) STA 图像是正方形（实测 500 张中仅 22 张
       为正方形，其余如 500x375、500x442、407x500）。把非正方形图压成 448x448 会
       拉伸模型实际看到的攻击图，kappa_map/gradcam_map 也在拉伸图上计算、与原始
       图像错位。因此 kappa_map/gradcam_map 是随图而变的"合并后"网格
       (gh//merge, gw//merge)（如 13x18、16x18），而非固定 16x16——这是正确且预期
       的行为；下游绘图用 imshow(extent=[0,W,H,0], interpolation="bilinear") 把每个
       网格拉伸到原始图像范围，任意 2-D 形状均可用。
  [G2] values_to_grid 传入的是"合并后"的网格 (h//merge, w//merge)，而非原始
       grid_thw。原因：grid_thw 是 patch 单位（448/14=32），Qwen2.5-VL 视觉塔会把
       2x2 patch merge 成 1 个 token，所以逐 token 的 kappa/cam 数量 = (h*w)/4，
       直接 reshape 到 (1,h,w) 会触发 token_mapping.values_to_grid 的 assert。
  [G3] model.requires_grad_(False)（仅 pixel_values 需要梯度）。
       原因：S.backward() 默认会给全部 ~7B 参数分配 .grad（约 +28GB 显存），
       而本脚本只用到视觉塔输出梯度 feat.grad；关闭参数梯度不影响任何数值结果。
       保持 fp32 权重以维持梯度数值稳定。
  [G5] 设备用 cuda:1（可用 CF_GPU 环境变量覆盖）。
       原因：cuda:0 被长期运行的其它任务占用（PyTorch 实测仅约 26GB 空闲，
       fp32 7B 权重本身约 28GB，无法加载）；cuda:1 约 48GB 空闲，足够
       fp32 权重 + 关闭参数梯度 + 一次小序列反传。nvidia-smi 显示 0MiB 是
       误导（沙箱视图），以 torch.cuda.mem_get_info 为准。
  [G4] generate 返回 4 元组 (input_ids, all_probs, hidden_states_list,
       all_input_down_proj)，[0] 即 generated_ids——与 brief 一致，无需改动。
  [G6] generate 需传 output_hidden_states=True。
       原因：本环境补丁后的 _sample 无条件执行
       outputs.hidden_states[-1][:,-1,:]，而 output_hidden_states 只有在
       generation_config.output_hidden_states 为真时才会传给 forward 并填充
       hidden_states；否则 outputs.hidden_states 为 None 直接 TypeError。
       brief 原文漏掉该参数。
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from diff_cf import differentiable_kappa
from token_mapping import image_token_indices, values_to_grid
from gradcam import gradcam_saliency

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_CACHE = "/data/public/models/Qwen"
LAYER_IDX = 11
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/weight"
WEIGHT_PATTERN = "QWen_MLP_down_proj_{layer}_weight.pth"
DEVICE = os.environ.get("CF_GPU", "cuda:1")  # [G5] cuda:0 被长期任务占满

STA_CSV = os.path.join(_IJAR, "datasets/STA/sta_mini.csv")
STA_IMAGE_DIR = os.path.join(_IJAR, "datasets/STA/images")
OUTPUT_DIR = os.path.join(_IJAR, "results/cf_heatmap/npz/qwen")
N_SAMPLES = 4
RANDOM_SEED = 42

FORWARD_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_SAMPLES
    print("=" * 66)
    print(f"CF heatmap extraction — Qwen2.5-VL-7B  (N={n}, layer L{LAYER_IDX})")
    print("=" * 66)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map=DEVICE,
        cache_dir=MODEL_CACHE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
    model.requires_grad_(False)   # [G3] 只需 pixel_values 梯度，避免为全参分配 grad
    print(f"  model loaded (fp32) on {DEVICE}")

    wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=LAYER_IDX))
    B = torch.load(wp, map_location=DEVICE).float().requires_grad_(False)  # [J, K]
    print(f"  down-proj weight L{LAYER_IDX}: {tuple(B.shape)}")

    merge = model.visual.spatial_merge_size   # 2: 2x2 patch -> 1 token

    df = pd.read_csv(STA_CSV).sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="samples"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        gt = str(row["answer"]).strip().lower()[:1]
        image_path = os.path.join(STA_IMAGE_DIR, image_id)

        # [G1] 原始图像路径直传 processor：native 动态分辨率，保持宽高比、不预拉伸
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _video = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=_video,
                           padding=True, return_tensors="pt")
        inputs = {k: inputs[k] for k in FORWARD_KEYS if k in inputs}
        inputs["pixel_values"] = inputs["pixel_values"].to(DEVICE).float().requires_grad_(True)
        inputs["input_ids"] = inputs["input_ids"].to(DEVICE)
        inputs["attention_mask"] = inputs["attention_mask"].to(DEVICE)
        inputs["image_grid_thw"] = inputs["image_grid_thw"].to(DEVICE)

        vision_out = {}

        def hook_fn(_m, _inp, out):
            out.retain_grad()
            vision_out["feat"] = out
        hook = model.visual.register_forward_hook(hook_fn)

        try:
            _outputs, all_down_proj = model(**inputs)
        finally:
            hook.remove()

        # 逐图像 token 的 kappa（LAYER_IDX 层，可微）
        img_idx = image_token_indices(inputs["input_ids"][0], model.config.image_token_id)
        X_img = all_down_proj[LAYER_IDX][0][img_idx]          # [n_img, K]
        kappa = differentiable_kappa(X_img.t().contiguous(), B)  # [n_img]
        # [G2] grid_thw 是 patch 单位；视觉塔 2x2 merge 后逐 token 网格 = h/merge x w/merge
        _, gh, gw = (int(x) for x in inputs["image_grid_thw"][0])
        grid_merged = (1, gh // merge, gw // merge)
        kappa_map = values_to_grid(kappa, grid_merged).detach()

        # GradCAM：S = mean(图像 token kappa)，反传到视觉塔输出
        s = kappa.mean()
        s.backward()
        feat = vision_out["feat"]
        cam = gradcam_saliency(feat, feat.grad)              # [n_merged]
        cam_map = values_to_grid(cam, grid_merged).detach()

        # 无梯度生成答案（对/错标注）；generate 返回 4 元组，[0] = generated_ids
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                output_hidden_states=True)[0]  # [G6] 补丁 _sample 需要
        gen_trim = generated_ids[0][len(inputs["input_ids"][0]):]
        output_text = processor.batch_decode(
            gen_trim.unsqueeze(0), skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()
        pred = output_text.lower()[:1] if output_text else ""

        out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_id)[0]}.npz")
        np.savez_compressed(
            out_path,
            image_path=image_path,
            image_id=image_id,
            question=question,
            ground_truth=gt,
            model_output=output_text,
            kappa_map=kappa_map.cpu().numpy(),
            gradcam_map=cam_map.cpu().numpy(),
            mean_kappa=float(kappa.mean().item()),
            model="qwen",
        )
        print(f"  saved {os.path.basename(out_path)}  "
              f"pred={pred} gt={gt} {'OK' if pred == gt else 'XX'}  "
              f"mean_kappa={float(kappa.mean().item()):.5f}")

    print(f"\nDone. NPZ files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
