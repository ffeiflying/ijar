# CF 反传热力图可视化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 STA 排版攻击图像，产出两张热力图——CF 值热力图（逐图像 token 的 kappa）和 CF 梯度热力图（从 CF 反传的 GradCAM saliency），三格并排可视化。

**Architecture:** 五个小模块。三个纯函数模块（可微 kappa、token→网格映射、GradCAM 公式）各有单元测试；一个 GPU 计算脚本产出 NPZ；一个绘图脚本把 NPZ 渲染成三格图。计算与绘图分离，方便调图。

**Tech Stack:** torch, transformers (qwen 环境，已打补丁返回 `all_input_down_proj`), pandas, matplotlib, PIL, numpy。

## Global Constraints

- **不用 git**：本项目不是 git 仓库，任务结尾用"运行验证命令"代替 commit。
- **统一用 qwen 环境 python**：`/data/liuxiaofei/miniconda3/envs/qwen/bin/python`（有 torch/transformers/pandas/matplotlib/scipy；**没有 pytest**，测试用纯 python 断言 + `if __name__ == "__main__"`）。
- **不修改任何已有文件、不修改任何环境**（conda、site-packages、transformers）。约束来自用户："先别改我的环境，要改环境的时候让我看一下改的哪里"。
- 所有新文件在 `ablation/cf_heatmap/` 下；文件必须存于 `/data/liuxiaofei`（用户记忆约束）。
- 模型层固定 LAYER_IDX=11（Qwen 最优层）；down-proj 权重 `/data/liuxiaofei/uncertainty/Qwen2.5-VL/weight/QWen_MLP_down_proj_11_weight.pth`。
- STA 数据：`datasets/STA/sta_mini.csv`（列 `index/question/answer`，500 行），图像 `datasets/STA/images/{index}`。

---

### Task 1: 可微 kappa 模块 `diff_cf.py`

**Files:**
- Create: `ablation/cf_heatmap/diff_cf.py`
- Test: `ablation/cf_heatmap/tests/test_diff_cf.py`

**Interfaces:**
- Produces: `differentiable_kappa(X: Tensor[K, T], B: Tensor[J, K]) -> Tensor[T]` — 逐 token 的 DST 冲突因子 kappa，可微，与 `envidence.EvidenceModel` 逐 token 循环数值等价。

背景：现有 `EvidenceModel.get_evidence_weights` 内含 `torch.no_grad()` 和 `torch.nonzero`（不可微），无法对 CF 反传。本任务实现闭式可微版本。对单 token 的 down-proj 输入 x（K 维）、固定权重 B（形状 [J, K]）：

- w = B̃ᵀx / J（B̃ = B − mean(B, axis=1)），u = Σⱼ relu(w)ⱼ，v = Σⱼ relu(−w)ⱼ
- eta_pos = 1/(K·(eᵘ−1) + 1)，eta_neg = 1/(1 − (1−e⁻ᵛ)^K)，kappa = K·eta_pos·(eᵘ−1)·(1 − eta_neg·e⁻ᵛ)

用 `expm1`/`log1p` 做数值稳定（fp32 下 `1−(1−e⁻ᵛ)^K` 会下溢为 1 导致 inf）。

- [ ] **Step 1: 写测试**

`ablation/cf_heatmap/tests/test_diff_cf.py`：

```python
import sys
import os
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # IJAR
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # cf_heatmap

from envidence import EvidenceModel
from diff_cf import differentiable_kappa


def reference_kappa(X, B):
    """逐 token 循环现有（不可微）EvidenceModel 作为参考值。"""
    ev = EvidenceModel(B.detach())
    out = []
    for t in range(X.shape[1]):
        ev.get_evidence_weights(X[:, t:t + 1].detach())
        out.append(ev.get_evidence_conflict().item())
    return np.array(out)


def test_equivalence():
    torch.manual_seed(0)
    J, K, T = 64, 128, 20
    B = torch.randn(J, K)
    X = torch.randn(K, T)
    got = differentiable_kappa(X, B).detach().numpy()
    ref = reference_kappa(X, B)
    rel = np.abs(got - ref) / (np.abs(ref) + 1e-9)
    assert np.all(rel < 1e-2), f"max rel diff {rel.max():.2e} (got {got[:3]}, ref {ref[:3]})"
    print(f"equivalence OK  max rel diff = {rel.max():.2e}")


def test_shape():
    torch.manual_seed(1)
    B = torch.randn(32, 64)
    X = torch.randn(64, 7)
    k = differentiable_kappa(X, B)
    assert tuple(k.shape) == (7,)
    print("shape OK", tuple(k.shape))


def test_gradient():
    torch.manual_seed(2)
    B = torch.randn(32, 64)
    X = torch.randn(64, 5, requires_grad=True)
    s = differentiable_kappa(X, B).sum()
    s.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
    print(f"gradient OK  mean|grad| = {X.grad.abs().mean().item():.4f}")


if __name__ == "__main__":
    test_equivalence()
    test_shape()
    test_gradient()
    print("ALL PASS")
```

- [ ] **Step 2: 运行验证失败**

运行：`/data/liuxiaofei/miniconda3/envs/qwen/bin/python ablation/cf_heatmap/tests/test_diff_cf.py`
期望：FAIL，`ModuleNotFoundError: No module named 'diff_cf'`

- [ ] **Step 3: 实现**

`ablation/cf_heatmap/diff_cf.py`：

```python
"""可微闭式 kappa（DST 冲突因子）——与 envidence.EvidenceModel 数值等价但可反传。

参考 envidence.py 的 _calculate_basic_terms：对单 token 的 down-proj 输入 x：
  w = B̃ᵀx / J（B̃ = B - mean(B, dim=1)）
  u = Σⱼ relu(w)ⱼ,  v = Σⱼ relu(-w)ⱼ
  eta_pos = 1/(K*(e^u - 1) + 1)
  eta_neg = 1/(1 - (1 - e^-v)^K)
  kappa   = K * eta_pos * (e^u - 1) * (1 - eta_neg * e^-v)
用 expm1/log1p 保证 fp32 数值稳定。
"""
import torch


def differentiable_kappa(X, B):
    """Per-token DST conflict factor (kappa) from MLP down-proj inputs.

    Args:
        X: tensor [K, T] — T tokens, each a K-dim down-proj input
           (K = LLM intermediate_size).
        B: tensor [J, K] — pretrained down-proj weight (fixed, no grad needed).

    Returns:
        kappa: tensor [T] — per-token conflict factor (float32, differentiable).
    """
    J, K = B.shape
    Bs = B - B.mean(dim=1, keepdim=True)            # [J, K]
    W = Bs @ X / J                                   # [J, T]
    u = torch.relu(W).sum(dim=0)                     # [T]
    v = torch.relu(-W).sum(dim=0)                    # [T]
    Kf = float(K)
    expm1u = torch.expm1(u)                          # [T]
    eta_pos = 1.0 / (Kf * expm1u + 1.0)              # [T]
    # eta_neg = 1/(1 - (1-e^-v)^K)；用 expm1/log1p 避免 fp32 下溢为 1
    eta_neg = -1.0 / torch.expm1(Kf * torch.log1p(-torch.exp(-v)))  # [T]
    kappa = Kf * eta_pos * expm1u * (1.0 - eta_neg * torch.exp(-v))  # [T]
    return kappa
```

- [ ] **Step 4: 运行验证通过**

运行：`/data/liuxiaofei/miniconda3/envs/qwen/bin/python ablation/cf_heatmap/tests/test_diff_cf.py`
期望：三行 `OK` + `ALL PASS`。若 equivalence 差超过 1e-2，检查 float32 vs float64 精度或数值稳定形式。

---

### Task 2: token→网格映射 `token_mapping.py`

**Files:**
- Create: `ablation/cf_heatmap/token_mapping.py`
- Test: `ablation/cf_heatmap/tests/test_token_mapping.py`

**Interfaces:**
- Produces:
  - `image_token_indices(input_ids: Tensor[L], image_token_id: int) -> Tensor` — prompt 中图像 token 的行位置（`input_ids == image_token_id`）。
  - `values_to_grid(values: Tensor[n], grid_thw: tuple(int,int,int)) -> Tensor` — 按行主序 reshape 成 (h, w)（单图 t==1）。

依据：Qwen2.5-VL 的图像 token 在序列中就是 `input_ids == image_token_id` 的连续段；`image_grid_thw` 给出 (t, h, w) 且 t·h·w = 图像 token 数；视觉塔输出（merger 后 + reverse_indices 恢复）为行主序。

- [ ] **Step 1: 写测试**

`ablation/cf_heatmap/tests/test_token_mapping.py`：

```python
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from token_mapping import image_token_indices, values_to_grid


def test_image_token_indices():
    input_ids = torch.tensor([1, 151655, 151655, 151655, 2])  # image_token_id = 151655
    idx = image_token_indices(input_ids, 151655)
    assert idx.tolist() == [1, 2, 3]
    print("indices OK")


def test_values_to_grid():
    grid_thw = (1, 16, 16)
    v = torch.arange(256.0)
    g = values_to_grid(v, grid_thw)
    assert tuple(g.shape) == (16, 16)
    assert g[0, 0].item() == 0.0 and g[15, 15].item() == 255.0
    print("grid OK")


def test_mismatch_raises():
    try:
        values_to_grid(torch.ones(100), (1, 16, 16))
        raise SystemExit("should have raised AssertionError")
    except AssertionError:
        print("mismatch raises OK")


if __name__ == "__main__":
    test_image_token_indices()
    test_values_to_grid()
    test_mismatch_raises()
    print("ALL PASS")
```

- [ ] **Step 2: 运行验证失败**

运行：`/data/liuxiaofei/miniconda3/envs/qwen/bin/python ablation/cf_heatmap/tests/test_token_mapping.py`
期望：FAIL，`ModuleNotFoundError: No module named 'token_mapping'`

- [ ] **Step 3: 实现**

`ablation/cf_heatmap/token_mapping.py`：

```python
"""把展平的 LLM token 位置映射回图像空间网格（Qwen2.5-VL）。"""
import torch


def image_token_indices(input_ids, image_token_id):
    """Row positions in the flattened prompt occupied by image tokens."""
    return torch.argwhere(input_ids == image_token_id).squeeze(1)


def values_to_grid(values, grid_thw):
    """Reshape per-image-token values [n] to a spatial grid.

    Args:
        values: tensor [n] — per-image-token values, row-major (vision output order).
        grid_thw: sequence of 3 ints (t, h, w) with t*h*w == n.

    Returns:
        grid: tensor [h, w] (single image, t == 1), or [t, h, w] (video).
    """
    t, h, w = (int(x) for x in grid_thw)
    n = t * h * w
    assert values.numel() == n, f"{values.numel()} values but grid is {n}"
    grid = values.reshape(t, h, w)
    if t == 1:
        return grid[0]
    return grid
```

- [ ] **Step 4: 运行验证通过**

运行：`/data/liuxiaofei/miniconda3/envs/qwen/bin/python ablation/cf_heatmap/tests/test_token_mapping.py`
期望：三行 `OK` + `ALL PASS`

---

### Task 3: GradCAM 公式 `gradcam.py`

**Files:**
- Create: `ablation/cf_heatmap/gradcam.py`
- Test: `ablation/cf_heatmap/tests/test_gradcam.py`

**Interfaces:**
- Produces: `gradcam_saliency(features: Tensor[S, C], grads: Tensor[S, C]) -> Tensor[S]` — 标准 GradCAM：`alpha = grads.mean(dim=0)`，`cam = relu((features * alpha).sum(dim=1))`。

- [ ] **Step 1: 写测试**

`ablation/cf_heatmap/tests/test_gradcam.py`：

```python
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gradcam import gradcam_saliency


def test_gradcam():
    torch.manual_seed(0)
    S, C = 12, 5
    features = torch.randn(S, C)
    grads = torch.randn(S, C)
    cam = gradcam_saliency(features, grads)
    assert tuple(cam.shape) == (S,)
    assert (cam >= 0).all()
    alpha = grads.mean(dim=0)
    ref = torch.relu((features * alpha).sum(dim=1))
    assert torch.allclose(cam, ref)
    print("gradcam OK")


if __name__ == "__main__":
    test_gradcam()
    print("ALL PASS")
```

- [ ] **Step 2: 运行验证失败**

运行：`/data/liuxiaofei/miniconda3/envs/qwen/bin/python ablation/cf_heatmap/tests/test_gradcam.py`
期望：FAIL，`ModuleNotFoundError: No module named 'gradcam'`

- [ ] **Step 3: 实现**

`ablation/cf_heatmap/gradcam.py`：

```python
"""GradCAM 空间显著性（标准公式）。"""
import torch


def gradcam_saliency(features, grads):
    """GradCAM-style spatial saliency.

    Args:
        features: tensor [S, C] — target-layer activations (S spatial positions).
        grads: tensor [S, C] — gradients of the scored scalar wrt features.

    Returns:
        cam: tensor [S] — non-negative per-position saliency.
    """
    alpha = grads.mean(dim=0)              # [C]
    cam = (features * alpha).sum(dim=1)    # [S]
    return torch.relu(cam)
```

- [ ] **Step 4: 运行验证通过**

运行：`/data/liuxiaofei/miniconda3/envs/qwen/bin/python ablation/cf_heatmap/tests/test_gradcam.py`
期望：`gradcam OK` + `ALL PASS`

---

### Task 4: GPU 计算脚本 `run_cf_heatmap_qwen.py`

**Files:**
- Create: `ablation/cf_heatmap/run_cf_heatmap_qwen.py`

**Interfaces:**
- Consumes: `differentiable_kappa`（Task 1）、`image_token_indices`/`values_to_grid`（Task 2）、`gradcam_saliency`（Task 3）。
- Produces: 每张采样图像一个 NPZ：`results/cf_heatmap/npz/{image_id_stem}.npz`，含键 `image_path, image_id, question, ground_truth, model_output, kappa_map, gradcam_map, mean_kappa`。`kappa_map`/`gradcam_map` 都是 16×16 的 numpy 数组。

关键机制（已在建模文件里核实）：
- Qwen 环境的 transformers 已补丁：`model(**inputs)` 返回 `(outputs, all_down_proj)`，`all_down_proj` 是逐层 MLP down-proj 输入张量的列表，第 0 层起每层形状 `[1, seq, intermediate]`。
- 视觉塔 `model.visual` 的 forward 返回 merger 后、已按 `reverse_indices` 恢复行主序的合并 token（448px → 256 个 = 16×16 网格），可直接 `register_forward_hook` 捕获，无需处理 window 重排。
- 模型以 fp32 加载，保证梯度稳定（H800 80GB，7B fp32 约 28GB，足够）。

- [ ] **Step 1: 写脚本**

`ablation/cf_heatmap/run_cf_heatmap_qwen.py`：

```python
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

STA_CSV = os.path.join(_IJAR, "datasets/STA/sta_mini.csv")
STA_IMAGE_DIR = os.path.join(_IJAR, "datasets/STA/images")
OUTPUT_DIR = os.path.join(_IJAR, "results/cf_heatmap/npz")
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
        MODEL_NAME, torch_dtype=torch.float32, device_map="cuda:0",
        cache_dir=MODEL_CACHE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
    print("  model loaded (fp32)")

    wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=LAYER_IDX))
    B = torch.load(wp, map_location="cuda:0").float().requires_grad_(False)  # [J, K]
    print(f"  down-proj weight L{LAYER_IDX}: {tuple(B.shape)}")

    df = pd.read_csv(STA_CSV).sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="samples"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        gt = str(row["answer"]).strip().lower()[:1]
        image_path = os.path.join(STA_IMAGE_DIR, image_id)

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
        inputs["pixel_values"] = inputs["pixel_values"].to("cuda:0").float().requires_grad_(True)
        inputs["input_ids"] = inputs["input_ids"].to("cuda:0")
        inputs["attention_mask"] = inputs["attention_mask"].to("cuda:0")
        inputs["image_grid_thw"] = inputs["image_grid_thw"].to("cuda:0")

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
        kappa_map = values_to_grid(kappa, inputs["image_grid_thw"][0]).detach()

        # GradCAM：S = mean(图像 token kappa)，反传到视觉塔输出
        s = kappa.mean()
        s.backward()
        feat = vision_out["feat"]
        cam = gradcam_saliency(feat, feat.grad)              # [n_merged]
        cam_map = values_to_grid(cam, inputs["image_grid_thw"][0]).detach()

        # 无梯度生成答案（对/错标注）
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=64, do_sample=False)[0]
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
        )
        print(f"  saved {os.path.basename(out_path)}  "
              f"pred={pred} gt={gt} {'OK' if pred == gt else 'XX'}  "
              f"mean_kappa={float(kappa.mean().item()):.5f}")

    print(f"\nDone. NPZ files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行验证（GPU）**

先确认 GPU 空闲，然后：

运行：`/data/liuxiaofei/miniconda3/envs/qwen/bin/python ablation/cf_heatmap/run_cf_heatmap_qwen.py 1`
期望：脚本打印模型加载、`down-proj weight L11: (3584, 8192)`，然后 1 张图像：
- 打印 `saved {id}.npz  pred=a gt=a OK  mean_kappa=0.xxxxx`
- `results/cf_heatmap/npz/` 下出现 1 个 `.npz` 文件

（若显存不足或报错，检查是否有其他任务占 GPU，重新在空闲时运行。）

---

### Task 5: 绘图脚本 `plot_cf_heatmap.py`

**Files:**
- Create: `ablation/cf_heatmap/plot_cf_heatmap.py`

**Interfaces:**
- Consumes: Task 4 产出的 NPZ。
- Produces: 每张样本三格并排 PNG/PDF：`results/cf_heatmap/figs/{image_id_stem}.png/.pdf`。三格 = 原图 | CF 梯度 saliency（`YlOrRd`，alpha 0.55，bilinear）| CF 值图（`RdBu_r`，alpha 0.55，bilinear），各带 colorbar。

- [ ] **Step 1: 写脚本**

`ablation/cf_heatmap/plot_cf_heatmap.py`：

```python
#!/usr/bin/env python
"""
把 run_cf_heatmap_qwen.py 产出的 NPZ 渲染成三格热力图。
Usage: python ablation/cf_heatmap/plot_cf_heatmap.py [N]   (N=前 N 个 npz, 0=全部)
"""
import sys
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_IJAR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPZ_DIR = os.path.join(_IJAR, "results/cf_heatmap/npz")
OUT_DIR = os.path.join(_IJAR, "results/cf_heatmap/figs")
LAYER = 11


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = [f for f in sorted(os.listdir(NPZ_DIR)) if f.endswith(".npz")]
    if len(sys.argv) > 1 and int(sys.argv[1]) > 0:
        files = files[:int(sys.argv[1])]

    for fn in files:
        d = np.load(os.path.join(NPZ_DIR, fn), allow_pickle=True)
        img = np.asarray(Image.open(str(d["image_path"])).convert("RGB"))
        H, W = img.shape[:2]
        kappa_map = d["kappa_map"].astype(float)
        cam_map = d["gradcam_map"].astype(float)
        gt = str(d["ground_truth"])
        pred = str(d["model_output"]).strip().lower()[:1]
        ok = (pred == gt)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].axis("off")
        axes[0].set_title(f"{d['image_id']}\nGT={gt} pred={pred} "
                          f"{'OK' if ok else 'XX'}", fontsize=10)

        axes[1].imshow(img)
        c1 = axes[1].imshow(cam_map, cmap="YlOrRd", alpha=0.55,
                            extent=[0, W, H, 0], interpolation="bilinear",
                            vmin=0, vmax=max(float(cam_map.max()), 1e-9))
        axes[1].set_title("CF-gradient saliency", fontsize=11)
        axes[1].axis("off")
        fig.colorbar(c1, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(img)
        c2 = axes[2].imshow(kappa_map, cmap="RdBu_r", alpha=0.55,
                            extent=[0, W, H, 0], interpolation="bilinear",
                            vmin=float(kappa_map.min()),
                            vmax=float(kappa_map.max()))
        axes[2].set_title(f"CF-value (mean={float(d['mean_kappa']):.4f})", fontsize=11)
        axes[2].axis("off")
        fig.colorbar(c2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(f"Qwen2.5-VL-7B · Layer L{LAYER} · {d['image_id']}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        stem = fn[:-4]
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.png"), dpi=200,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.pdf"), bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"  saved {OUT_DIR}/{stem}.png/.pdf")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行验证**

运行：`/data/liuxiaofei/miniconda3/envs/qwen/bin/python ablation/cf_heatmap/plot_cf_heatmap.py`
期望：对每个 npz 打印 `saved .../figs/{stem}.png/.pdf`；打开 PNG 检查：三格并排、两张热力图叠加在图像上、有 colorbar、热力图大致集中在图像中央排版文字区域。

- [ ] **Step 3: 全量跑 + 人眼检查**

运行：`/data/liuxiaofei/miniconda3/envs/qwen/bin/python ablation/cf_heatmap/run_cf_heatmap_qwen.py 4`
然后：`/data/liuxiaofei/miniconda3/envs/qwen/bin/python ablation/cf_heatmap/plot_cf_heatmap.py`
期望：4 张三格图生成，肉眼检查排版文字区域是否出现高冲突/高梯度热点。把结果图路径报给用户确认。
