# Experiment Log — ACF for VLM Misbehavior Detection

> 本文档记录所有实验的设置、参数选择、设计决策、踩坑过程及关键结果。
> 写论文时参考此文档还原实验细节。

---

## 0. 通用设置

### 0.1 核心方法：Evidence-based ACF

所有实验统一使用 DST（Dempster-Shafer Theory）证据分解计算 **Attention Conflict Factor (ACF)**。

- **计算位置**：LLM 某层 MLP down-projection 的输入 hidden states
- **Evidence 模型**：`IJAR/envidence.py` — `EvidenceModel` 类
- **Label 定义**：Positive = 错误回答 (choice ≠ 'a'), Negative = 正确回答 (choice == 'a')
- **评估指标**：AUROC, AUPR（<0.5 时 flip）, Accuracy, ASR

### 0.2 四个模型

| 模型 | 参数量 | 层数 | 视觉编码器 | 路径 |
|------|--------|------|-----------|------|
| MoF-Models-7B | ~13B (vicuna) | 40 | CLIP-L + DINOv2-L | `/data/public/models/MoF_Models/MoF_Models/MoF_Models` |
| InternVL2_5-8B | 8B | 32 | InternViT (CLIP-based) | `/data/public/models/InternVL/InternVL2_5-8B` |
| Qwen2.5-VL-7B | 7B | 28 | Qwen Vision (CLIP-based) | `/data/public/models/Qwen` |
| DeepSeek-VL2-Tiny | ~3B (MoE) | 12 | SigLIP-SO400M | `/data/public/models/deepseek/models--deepseek-ai--deepseek-vl2-tiny/...` |

### 0.3 数据集

| 数据集 | 样本数 | 用途 |
|--------|--------|------|
| STA-5K-Mini | 500 | 主测试集，结构理解攻击 |
| RTA-100 | 100 | 拒绝回答攻击 |
| Typo-parameter (19个) | 500/个 | 排版参数攻击，5组变量 |

### 0.4 通用超参

```python
RANDOM_SEED = 42
MAX_NEW_TOKENS = 128
# ACF 计算层: LAYER_IDX = 0（非 layerwise 实验固定第 0 层）
```

---

## 1. Layer-wise ACF 分析

**论文位置**: Section 4 / Figure 5  
**目录**: `IJAR/layerwise/`  
**结果**: `IJAR/results/layerwise_acf/`

### 1.1 实验目的

验证 ACF 在不同 transformer 层的表现差异——是否深层比浅层更有区分度。

### 1.2 实验设置

```
模型: 全部 4 个 (MoF, InternVL, Qwen, DeepSeek)
数据集: STA-5K-Mini (100 samples), RTA-100 (100 samples)
层范围: 全部层 (MoF=40, InternVL=32, Qwen=28, DeepSeek=12)
生成策略: greedy (do_sample=False)，确保可复现
每层计算: conflict + ignorance + 逐 token ACF
```

### 1.3 参数细节

| 模型 | do_sample | temperature | 备注 |
|------|-----------|-------------|------|
| MoF | True | 0.2 | `top_p=None, num_beams=1` |
| InternVL | False | - | greedy |
| DeepSeek | False | - | greedy |
| Qwen | default | - | 模型默认配置 |

### 1.4 设计决策

- **为什么用 greedy**：layerwise 实验关注 ACF 随层的变化规律，不是生成多样性。greedy 消除随机性，确保逐层对比的可复现性。
- **MoF 例外用 t=0.2**：历史原因，MoF 脚本最先写，沿用了 `do_sample=True` 的设置。
- **采样 100 张**：全量 500 张计算量太大（逐层逐 token），100 张足够看出趋势。

### 1.5 结果文件

```
STA_layerwise_acf_{model}.json
RTA_layerwise_acf_{model}.json
```

---

## 2. Temperature 敏感性分析

**论文位置**: Section 4 / Figure 6  
**目录**: `IJAR/temperature/`  
**结果**: `IJAR/results/temperature/`

### 2.1 实验目的

验证生成 temperature 对 ACF 区分度的影响——ACF 是否在合理 temperature 范围内保持稳定。

### 2.2 实验设置

```
模型: 全部 4 个
数据集: STA-5K-Mini (full 500), RTA-100 (full 100)
ACF 层: Layer 0 (固定)
Temperature 范围: [0.5, 1.0, 1.5, 2.0, 5.0]
do_sample: True
```

### 2.3 设计决策

- **为什么固定 Layer 0**：temperature 实验关注生成随机性的影响，不是层间差异。固定层消除层变量。
- **Temperature 范围选择**：[0.5, 5.0] 覆盖从接近 greedy 到高随机性的范围。大多数 VLM 默认 t=1.0。
- **全量测试**：每张图在每个 temperature 下推理一次，500*5=2500 次推理。

### 2.4 关键发现

（待填入具体数据）

---

## 3. Typographic Parameter 攻击分析

**论文位置**: Section 5 / Figure 7  
**目录**: `IJAR/typo-parameter/`  
**结果**: `IJAR/results/typo_parameter/`

### 3.1 实验目的

分析排版攻击的各个参数（字体大小、颜色、透明度、位置、字体族）对 ACF 检测效果的影响。

### 3.2 数据集构建

**19 个数据集，5 组变量**：

| 变量组 | 条件 | 文件数 |
|--------|------|--------|
| Font Size | 16px, 20px, 24px, 28px, 32px | 500×5 |
| Color Contrast | low, medium, high | 500×3 |
| Opacity | 30%, 50%, 70%, 90% | 500×4 |
| Position | center, corner, random | 500×3 |
| Font Family | serif, sans-serif, monospace, decorative | 500×4 |

**Clean baseline**：与 typo 图片相同但无排版文字覆盖的干净图片（500 张）。

### 3.3 实验设置

```
模型: 全部 4 个
ACF 层: Layer 0 (固定)
Temperature: 1.0 (固定)
do_sample: True
```

**两个子实验**：

| 子实验 | 样本数 | 说明 |
|--------|--------|------|
| Typo-only | 500/条件 | 只在 typo 图片上评估 ACF 区分正确/错误回答的能力 |
| Typo+Clean | 1000/条件 (500 typo + 500 clean) | 同时评估 typo 和 clean，检验 ACF 是否在 clean 上虚高 |

### 3.4 设计决策

- **为什么固定 t=1.0**：typo 实验关注攻击参数的影响，不是生成参数。固定 temperature 消除干扰。
- **为什么加 clean baseline**：纯 typo 数据上 AUROC 高可能只是模型在 typo 图片上整体更不确定。加入 clean 图片可以区分「ACF 对 typo 敏感」还是「ACF 对所有图片都高」。

---

## 4. MoF DINO 特征消融

**论文位置**: Section 6.2  
**目录**: `IJAR/ablation/`  
**结果**: `IJAR/results/ablation/mof_dino_ablation.json`

### 4.1 实验目的

验证 CLIP 和 DINO 视觉特征各自对 ACF 的贡献——MoF 原生同时拥有两个编码器，可以单独置零其中一个。

### 4.2 实验设置

```
模型: MoF-Models-7B  only
数据集: STA-5K-Mini (500)
ACF 层: Layer 0
Temperature: 1.0
```

**四种模式**：

| 模式 | 说明 | 实现方式 |
|------|------|---------|
| interleaved | CLIP + DINO 交织（完整模型） | 原始 forward |
| concat | 所有 CLIP → 所有 DINO（拼接） | 修改 token 排列逻辑 |
| clip_only | 只有 CLIP，DINO 输出置零 | monkey-patch `encode_images_withdino` → zeros |
| dino_only | 只有 DINO，CLIP 输出置零 | monkey-patch `encode_images_withclip` → zeros |

### 4.3 设计决策

- **为什么用 monkey-patch 而非修改源码**：不修改 `/data/liuxiaofei/uncertainty/` 下的环境代码。运行时替换方法，跑完恢复。
- **为什么 zero-out 而非删除**：保持 1152 token 的序列长度不变，避免 sequence length distribution shift。
- **concat 模式怎么加的**：MoF 原生只有 interleaved（`merged_features[0::2]=CLIP; merged_features[1::2]=DINO`），concat 改为 `torch.cat([CLIP, DINO], dim=0)`。

### 4.4 关键结果

| Mode | AUROC | AUPR | ACC | ASR |
|------|-------|------|-----|-----|
| interleaved | 0.9947 | 0.9995 | 0.080 | 0.920 |
| concat | 0.9926 | 0.9989 | 0.134 | 0.866 |
| clip_only | 0.984 | 0.9978 | 0.112 | 0.888 |
| dino_only | 0.985 | 0.9850 | 0.526 | 0.474 |

### 4.5 分析

- **AUROC 都很高（>0.98）**：说明无论哪种特征组合，ACF 都能有效区分正确/错误回答。但需要注意 ASR 的差异。
- **interleaved ASR 最高（0.92）**：模型在交织模式下最容易出错。
- **dino_only ACC 最高（0.526）**：只有 DINO 特征时模型正确率最高，但 ACF 的 AUPR 下降明显，说明区分度变弱。
- **CLIP vs DINO 贡献接近**：clip_only 和 dino_only 的 AUROC 接近（0.984 vs 0.985），说明两个编码器对 ACF 的贡献相当。

---

## 5. DINO 特征注入训练（Interleave / Fused）

**论文位置**: Section 6.1  
**目录**: `IJAR/ablation/interleave/`  
**结果**: `IJAR/results/interleave/`

### 5.1 实验目的

给 InternVL（原生只有 CLIP）和 DeepSeek（原生只有 SigLIP）额外注入 DINOv2 特征，训练投影器使 DINO 特征与原生视觉特征融合，看对 ACF 的影响。

### 5.2 架构设计与 Loss Function

#### 5.2.1 InternVL2_5-8B — Token 级交织

```
                            ┌────────────────────────────────────────────────────┐
                           │                   ❄️ 冻结模块                        │
                            │                                                    │
  输入图片 (448×448)  ──────▶  InternViT (CLIP-based)  ──▶  pixel_shuffle       │
                            │  [InternVisionModel]           downsample          │
                            │                                → 256 tokens        │
                            │                                [B, 256, 3584]      │
                            │                                                    │
                            │  DINOv2 ViT-L/14 (fp32)  ──▶  adaptive_pool       │
                            │  [dinov2_vitl14]              576 → 256 patches    │
                            │  ❄️ 冻结                       → [B, 256, 1024]    │
                            └────────────────────────────────────────────────────┘

                            ┌────────────────────────────────────────────────────┐
                            │                   🔥 可训练模块                      │
                            │                                                    │
  DINO pooled features ────▶  DINO Projector                                    │
  [B, 256, 1024]             ┌──────────────────────────────────────────────┐    │
                             │  nn.LayerNorm(1024)                          │    │
                             │  nn.Linear(1024 → 3584)    ← 随机初始化       │    │
                             │  nn.GELU()                                    │    │
                             │  nn.Linear(3584 → 3584)    ← 🔥零初始化!      │    │
                             └──────────────────────────────────────────────┘    │
                             → dino_residual [B, 256, 3584]                      │
                             + <IMG_CONTEXT> token embedding (dino base)         │
                             → dino_embeds [B, 256, 3584]                        │
                             (无 gate，纯残差)                                    │
                            └────────────────────────────────────────────────────┘

  ┌─ Token 排列 ─────────────────────────────────────────────────────────────────┐
  │                                                                              │
  │  concat:                                                                     │
  │    [CLIP_0 ... CLIP_255 | DINO_0 ... DINO_255]    512 tokens                 │
  │                                                                              │
  │  interleaved:                                                                │
  │    [CLIP_0, DINO_0, CLIP_1, DINO_1, ... CLIP_255, DINO_255]  512 tokens     │
  └──────────────────────────────────────────────────────────────────────────────┘

                            ┌────────────────────────────────────────────────────┐
                            │               ❄️ 冻结：InternLM2-8B                 │
                            │                                                    │
  [text_embeds + visual_tokens] ──▶  language_model.forward()                   │
                                     → hidden_states [B, seq_len, 3584]          │
                                     → lm_head                                   │
                                     → logits [B, seq_len, vocab_size]           │
                            └────────────────────────────────────────────────────┘

  Loss:
    shift_logits  = logits[:, :-1, :]     # 每个位置预测下一个 token
    shift_labels  = labels[:, 1:]          # 图像位置 = -100（忽略）
    L = CrossEntropyLoss()(shift_logits, shift_labels)
      = -1/|T_text| · Σ_{t ∈ text_positions} log P(target_token_t | image, prefix_{<t})
```

**可训练参数**: `dino_projector` (4 层，约 13.8M)  
**无 gate**：gate=0 + 零初始化 last layer 导致 gate 梯度恒为 0 → 去掉 gate

---

#### 5.2.2 DeepSeek-VL2-Tiny — Channel 级残差融合

```
                            ┌────────────────────────────────────────────────────┐
                            │                   ❄️ 冻结模块                        │
                            │                                                    │
  输入图片 (384×384)  ──────▶  SigLIP-SO400M  ──▶  MlpProjector                 │
                            │  [VisionTransformer]     (downsample_mlp_gelu)     │
                            │  576 patches             576 → 196 tokens          │
                            │  → [2 tiles, 576, 1152]  → [2 tiles, 196, 1280]   │
                            │                                                    │
                            │  DINOv2 ViT-L/14 (fp32)                            │
                            │  ❄️ 冻结                                           │
                            │  384→336 resize → 576 patches → [2 tiles, 576, 1024]│
                            └────────────────────────────────────────────────────┘

                            ┌────────────────────────────────────────────────────┐
                            │                   🔥 可训练模块                      │
                            │                                                    │
  DINO patches ──────────▶  adaptive_avg_pool2d  576 → 196 (24×24 → 14×14)     │
  [2 tiles, 576, 1024]                                                          │
                                │                                                │
                                ▼                                                │
                            DINO Projector (fp32)                                │
                            ┌──────────────────────────────────────────────┐    │
                            │  nn.LayerNorm(1024)                          │    │
                            │  nn.Linear(1024 → 1280)    ← 随机初始化       │    │
                            │  nn.GELU()                                    │    │
                            │  nn.Linear(1280 → 1280)    ← 🔥零初始化!      │    │
                            └──────────────────────────────────────────────┘    │
                            → dino_residual [2 tiles, 196, 1280]                 │
                                                                                 │
  ┌─ 融合（保持 196 tokens，不破坏 2D 网格）──────────────────────────────────┐  │
  │                                                                          │  │
  │  dino_gate ∈ R  (可学习标量，初始 0.01)                                   │  │
  │                                                                          │  │
  │  fused = native_embeds + dino_gate · dino_residual                       │  │
  │        = images_embeds + 0.01 · dino_projector(dino_feats)               │  │
  │                                                                          │  │
  │  初始时 dino_residual ≈ 0 (最后一层零初始化)                              │  │
  │  → fused ≈ native_embeds → 行为等价于 native_only                        │  │
  └──────────────────────────────────────────────────────────────────────────┘  │
                            └────────────────────────────────────────────────────┘

  ┌─ 2D 格式化（保持 14×14 网格 + 每行换行 token）──────────────────────────────┐
  │  1 global tile: 196 tokens → 14×14 + newline_col → 210 tokens              │
  │  + 1 view_separator token                                                   │
  │  + 1 local tile:  196 tokens → 14×14 + newline_col → 210 tokens            │
  │  = 421 image tokens 写入 input_embeds (masked_scatter_)                     │
  │  + text_tokens                                                              │
  └──────────────────────────────────────────────────────────────────────────────┘

                            ┌────────────────────────────────────────────────────┐
                            │          ❄️ 冻结：DeepSeek-V2 (MoE)                 │
                            │                                                    │
  [input_embeds] ──▶  DeepseekV2ForCausalLM.forward()                           │
                      → outputs.loss  (内部 shift + CrossEntropyLoss)            │
                            └────────────────────────────────────────────────────┘

  Loss:
    labels: 图像位置 = -100（忽略），文本位置 = 原始 token id
    model.language 内部:
      shift_logits = logits[:, :-1, :]
      shift_labels = labels[:, 1:]
      L = CrossEntropyLoss()(shift_logits, shift_labels)
        = -1/|T_text| · Σ_{t ∈ text_positions} log P(target_token_t | image, prefix_{<t})
```

**可训练参数**: `dino_projector` (4 层，约 3.9M) + `dino_gate` (1 个标量)  
**保留 gate**：`native + gate * residual` 中 native 有梯度 → gate 可收到梯度，不会死锁

---

#### 5.2.3 Qwen2.5-VL-7B — Token 级交织

```
    ┌─ 与 InternVL 类似，token 级 concat/interleaved ─┐
    │                                                  │
    │  原生: Qwen ViT → 视觉 tokens (数量随分辨率变化)   │
    │  DINO: DINOv2 ViT-L/14 → DINO Projector          │
    │        → dino_tokens (等量扩展)                    │
    │                                                  │
    │  Loss: 同 InternVL — CrossEntropyLoss            │
    │  可训练: dino_projector (4 层，约 13.8M)          │
    │  无 gate (和 InternVL 相同原因)                    │
    │                                                  │
    │  ⚠ 环境问题暂停，未跑训练                           │
    └──────────────────────────────────────────────────┘
```

---

#### 5.2.4 MoF-Models-7B — 不需要训练

```
    MoF 原生拥有 CLIP + DINO 双编码器，投影器已联合训练好。
    只做推理消融，不训练任何参数。
    Loss 不适用（纯推理）。
```

---

#### 5.2.5 Loss Function 汇总

| 模型 | Loss 函数 | 公式 | 可训练参数 | 参数量 |
|------|----------|------|-----------|--------|
| InternVL | `CrossEntropyLoss` | `-1/N · Σ log P(y_t | x, y_{<t})` | `dino_projector` | ~13.8M |
| DeepSeek | `CrossEntropyLoss` | 同上（language model 内部计算） | `dino_projector` + `dino_gate` | ~3.9M + 1 |
| Qwen | `CrossEntropyLoss` | 同上 | `dino_projector` | ~13.8M |
| MoF | 无（不需要训练） | - | 无 | 0 |

所有模型的 loss 本质相同：**Causal LM next-token prediction cross-entropy**，
图像 token 的 label 置 -100 忽略，只计算文本 token 位置上的预测 loss。

#### 5.2.6 DINO 投影器初始化策略

```
  投影器结构（三模型相同）:
    LayerNorm(1024) → Linear(1024→hidden) → GELU → Linear(hidden→hidden)

  初始化:
    for m in [LayerNorm, Linear_1, GELU, Linear_2]:
        m.reset_parameters()          # 前 3 层: 默认随机 (Kaiming uniform)
    nn.init.zeros_(Linear_2.weight)   # 最后一层: weight = 0
    nn.init.zeros_(Linear_2.bias)     # 最后一层: bias = 0

  效果:
    初始时 dino_residual = 0 → fused = native → loss = native baseline
    训练中 投影器逐渐学习 → dino_residual 从 0 增长 → DINO 特征逐步融入
```

### 5.3 训练设置

```python
训练数据:    ImageNet-val 随机 2000 张（排除 STA 图片）
模板:        ["This is a photo of a {class_name}.", ...] 4 种随机选
任务:        Next-token prediction
BATCH_SIZE:  2
GRAD_ACCUM:  8                  # effective batch = 16
LEARNING_RATE: 2e-3             # AdamW
NUM_EPOCHS:  3
Optimizer:   torch.optim.AdamW

冻结:  LLM, 原生视觉编码器, 原生投影器, DINOv2 编码器
可训练: dino_projector (+ dino_gate for DeepSeek only)
```

### 5.4 踩坑记录

#### 坑 1: InternVL gate 梯度死锁

**现象**：`dino_gate` 初始 0，梯度始终为 0，loss 不变。  
**原因**：gate=0 × 零初始化 last layer = 残差恒为 0，gate 在乘法中系数为 0 → 梯度为 0。  
**解决**：去掉 gate，直接 residual addition。

#### 坑 2: DeepSeek processor 阻断梯度

**现象**：`input_embeds.grad ≠ 0`，`images_feature.grad ≠ 0`，但 `dino_projector.grad = 0`。  
**排查过程**（`debug_deepseek_grad.py` / `debug_deepseek_grad2.py`）：
- 逐 tensor `.retain_grad()` 追踪，发现梯度在 processor 的 conversation formatting 链条中消失
- 单独测试 `masked_scatter_` 确认梯度能正常通过
- 手动构造 input 后梯度正常 → 确认是 processor 的问题  
**解决**：手动构造 `[BOS]+[PAD×421]+text_tokens+[EOS]`，只用 processor 的 tokenizer 和图像预处理。

#### 坑 3: DeepSeek 投影器随机初始化导致 loss 不降

**现象**：loss 很平稳，没有下降趋势。  
**原因**：`reset_dino_projector()` 对所有层都随机初始化，最后一层的随机噪声加到 native_embeds 上 → loss 被噪声抬高 → 需要同时「压低 gate」和「学到有用特征」两个矛盾目标。  
**解决**：最后一层零初始化，确保初始 residual = 0。

#### 坑 4: InternVL 推理时 `_init_dino` 覆盖 checkpoint

**现象**：推理加载 checkpoint 后，第一次 forward 时投影器权重被重置。  
**原因**：`_init_dino()` 中包含了 reset 逻辑。  
**解决**：分离 `_init_dino()`（只加载 DINO 编码器）和 `reset_dino_projector()`（训练前调用）。

### 5.5 已跑模式

| 模型 | 已跑模式 | 状态 |
|------|---------|------|
| InternVL | native_only, concat, interleaved | ✅ 训练+推理完成 |
| DeepSeek | native_only | ✅ 推理完成（无需训练） |
| DeepSeek | fused | 🔄 训练中（待验证 loss 是否下降） |
| Qwen | native_only, concat, interleaved | ⏸ 暂停（环境问题） |

### 5.6 InternVL 结果

| Mode | AUROC | AUPR | ACC | ASR |
|------|-------|------|-----|-----|
| native_only | 0.285 | 0.399 | 0.492 | 0.508 |
| concat | 0.429 | 0.128 | 0.856 | 0.144 |
| interleaved | 0.774 | 0.163 | 0.946 | 0.054 |

**分析**：
- native_only 的 AUROC≈0.285（<0.5 flip 后）→ 说明纯 CLIP 特征的 ACF 区分度差
- interleaved 的 AUROC=0.774 明显优于 concat=0.429 → 交织排列让 CLIP 和 DINO 特征在相邻位置更好地交互
- interleaved 的 ASR=0.054 最低 → 加入 DINO 后模型更不容易被攻击

### 5.7 MoF interleaved vs concat

| Mode | AUROC | AUPR | ACC | ASR |
|------|-------|------|-----|-----|
| interleaved | 0.995 | 1.000 | 0.080 | 0.920 |
| concat | 0.993 | 0.999 | 0.134 | 0.866 |

**分析**：
- 两个模式的 AUROC 都极高（>0.99），说明 MoF 的 ACF 本身就很强
- concat 的 ACC 更高（0.134 vs 0.080）但 AUROC 略低 → concat 排列下模型更准确但 ACF 区分度略差
- MoF 的结果与 InternVL 不同（MoF 两个模式差距小）→ MoF 的 CLIP 和 DINO 投影器是联合训练的，排列方式影响小；InternVL 的 DINO 投影器是后训练的，排列方式影响大

---

## 6. 代码修改约束

所有实验修改均在 `IJAR/` 目录下完成，遵循以下规则：

1. **不修改源码**：`/data/liuxiaofei/uncertainty/` 和 `/data/public/models/` 下的文件不动
2. **需要改源码时**：复制到 IJAR 对应子目录，在复制版上修改
3. **不修改环境**：不 `pip install` 新包，不修改 conda 环境
4. **运行时路径**：通过 `sys.path.insert` 引用 IJAR 下的修改版代码
5. **HF 镜像**：下载模型时用 `HF_ENDPOINT=https://hf-mirror.com`

---

## 7. 待完成

- [ ] DeepSeek fused 模式训练（修复零初始化后重跑）
- [ ] DeepSeek fused 模式推理（`run_interleave_deepseek.py`）
- [ ] Qwen concat/interleaved 训练+推理（环境问题解决后）
- [ ] 所有模型的 interleave 结果汇总画图

---

## 8. 跨模态冲突有效性实验（2026-09-05）⚠️ 发现答案字母混杂

**目的**：检验 ACF 是否真的捕捉「文本↔视觉冲突」。4 个条件看 ACF 区分度：
纯文本（无图）、纯图像（+通用描述提示词）、干净图像+问题（一致）、攻击图像+问题（冲突）。

**新增数据（qwen，n=200/条件，全 28 层）**：
- `results/crossmodal/text_only_{sta,cln}_layerwise_qwen.json`（纯文本，答案 2 token）
- `results/crossmodal/image_only_{att,cln}_layerwise_qwen.json`（纯图像，描述 65 token）
- 干净双模态=`clean_layerauroc_qwen.json`、攻击双模态=`STA_layerauroc_qwen.json`（n=500→对齐 200）
- 脚本 `ablation/crossmodal/run_crossmodal4cond_qwen.py` / `analyze_crossmodal4cond.py`
- 汇总 `results/crossmodal/crossmodal4cond_summary.json`

**L0（当前论文 qwen 固定层）均值 ACF**：
`纯文本 0.0455 / 纯图像攻击 0.0625 / 纯图像干净 0.0618 / 干净双模态 0.0432 / 攻击双模态 0.0450`
→ **朴素假设不成立**：攻击双模态不是最高，甚至低于纯图像/纯文本；绝对水平被输出内容/长度主导（描述任务 65 token >> 字母任务 2 token）。

**关键混杂：答案字母效应（纯文本对照暴露）** ⚠️
纯文本（无图、无攻击）下回答 `b` 的 L0 ACF 显著高于回答 `a`（d≈+4.7，STA 与 clean 问题同向）。由于 STA 攻击图中被愚弄=答 `b`（291/500），干净图几乎全答 `a`（493/500），**L0 的攻击/干净区分在很大程度是「是否输出 b」的代偿，而非文本-视觉冲突**。

逐层看（qwen）：
| 层 | letterEff T(b-a) | membEff D-C | 控制字母后 D_a-C_a |
|----|----|----|----|
| L0 | +4.86 | +2.32 | +1.53 (AUROC .883) |
| L8 | +6.53 | +1.40 | +0.42 |
| L26-27 | ~0 | +3.9~4.3 | +3.8~4.6 (AUROC .99) |

四个模型在其选定的层（qwen L0 / internvl L1 / mof L32 / deepseek L4）均存在强字母效应（qwen b>a、internvl L1 b<<a d≈−9.6、mof L32 b<a、deepseek L4 b>a），方向随模型/层变化。**逐层 AUROC 高值因此与「输出选项字母」强纠缠**。

**控制字母后的残余攻击印记**（攻击但答对 a 的图像 vs 干净 a）：
- qwen L0 +1.53、deepseek L4 +1.60（中）；internvl L1 +0.49（弱）；mof L32 −1.55（n=19 不稳）
- 深层普遍强正（qwen L27 +4.6 / internvl L31 +2.3 / deepseek L11 +2.3）——是真「攻击图像印记」，但**不随是否被愚弄变化**，与先前 qwen 深层 membership 难迁移到 wrongness（0.83）一致。

**结论（暂）**：ACF 在固定层能区分攻击/干净，但早期层主要由「模型将输出第 2 个选项字母 b」驱动（纯文本即存在），不能作为「捕捉文本-视觉冲突」的干净证据；深层存在与答案无关的攻击印记但对 misbehavior 预测弱。STA 的 GT 恒为 a 使「被愚弄 ≡ 答 b」，任何检测都会与 token 身份纠缠——需要在 GT 多样化 / 正确选项位置轮换的基准上重新验证，或改述机制为隐藏态读出的答案倾向而非冲突。**待用户定方向。**

---

## 附录：文件索引

### 脚本

| 功能 | 路径 |
|------|------|
| 核心 evidence 模型 | `IJAR/envidence.py` |
| Layerwise ACF | `IJAR/layerwise/layerwise_acf_{model}.py` |
| Temperature sweep | `IJAR/temperature/temperature_acf_{model}.py` |
| Typo parameter | `IJAR/typo-parameter/run_typo_{model}.py` |
| MoF DINO ablation | `IJAR/ablation/run_mof_dino_ablation.py` |
| DINO projector 训练 | `IJAR/ablation/interleave/train_dino_projector.py` |
| Interleave 推理 | `IJAR/ablation/interleave/run_interleave_{model}.py` |

### 修改版模型代码

| 模型 | 路径 |
|------|------|
| InternVL + DINO | `IJAR/ablation/interleave/internvl/modeling_internvl_chat_dino.py` |
| DeepSeek + DINO | `IJAR/ablation/interleave/deepseek/modeling_deepseek_vl_v2_dino.py` |
| Qwen + DINO | `IJAR/ablation/interleave/qwen/qwen_dino_wrapper.py` |

### 权重文件

| 模型 | 路径 |
|------|------|
| Evidence weights (4 models) | `/data/liuxiaofei/uncertainty/{Model}/weight/` |
| DINO projector checkpoints | `IJAR/results/interleave/{model}_dino_projector_{mode}.pth` |
| DINOv2 pretrained | `/data/liuxiaofei/torch_cache/hub/checkpoints/dinov2_vitl14_pretrain.pth` |
