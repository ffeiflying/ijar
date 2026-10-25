# IJAR Extended Paper — Experiment Plan & Timeline

**Target**: First draft by 2026-07-25 | **Experiment deadline**: 2026-07-20

---

## Current Status

| Item | Status |
|------|--------|
| Main results (Table 1) | ✅ Done |
| Figures 1–3 (attack examples, framework, datasets) | ✅ PNG files added |
| All table/figure placeholders in LaTeX | ✅ Ready |
| All ablation data tables | ⬜ Empty |

---

## Phase 1: Core Ablations (target: 2026-07-12)

### 1.1 Layer-wise ACF Analysis

**Two metrics, two trends (预期相反)**:

| 指标 | 定��� | 随层加深的预期趋势 | 原因 |
|------|------|:---:|------|
| **ACF mean** | 每层 ACF 分数的绝对值 | ↑ 上升 | 深层表征更抽象，clean 和 attacked 的"绝对冲突量"都变大 |
| **AUROC** | 用该层 ACF 区分 clean/attacked 的能力 | ↓ 下降 | 深层 clean 样本也有语义性的"自然冲突"，区分度被稀释 |

**核心假设**：
> 如果 ACF ↑ 但 AUROC ↓，说明早期层的冲突更"纯"（只反映 typo 引入的外来冲突），
> 深层冲突混入了抽象语义的自然不确定性，导致 clean/attacked 区分不开。
> **这证明了检测 typographic attack 应该用早期层的 ACF。**

- **Code needed**: Extract ACF independently at every transformer block (not just block 1)
- **Models**: Qwen2.5-VL-7B, MoF-Models-7B
- **Datasets**: STA-5K-Mini (1,000 samples), RTA-100 (1,000 samples)
- **Deliverables**:
  - Fill `tab:layer_wise` — per block: ACF_mean_clean, ACF_mean_attacked, ACF diff, AUROC, AUPR
  - Plot `fig5_layerwise_acf.pdf`:
    - (a) **ACF value vs layer**: 两条曲线（clean vs attacked），预期都上升但 attacked 始终高于 clean
    - (b) **AUROC vs layer**: 一条曲线，预期单调下降
    - 两图并排，ACF↑ 但 AUROC↓ → 一眼看出"稀释效应"
  - Write analysis: 解释为什么 ACF 绝对值上升但区分度下降，讨论"纯冲突"vs"语义噪声"

### 1.2 Temperature Analysis
- **Code needed**: Modify attention softmax temperature during inference (scale logits before softmax)
- **Models**: Qwen2.5-VL-7B, MoF-Models-7B
- **Dataset**: STA-5K-Mini
- **Temperatures**: 0.5, 1.0 (default), 1.5, 2.0, 5.0
- **Deliverables**:
  - Fill `tab:temperature` (AUROC/AUPR at each temperature)
  - Plot `fig6_temperature_auroc.pdf`: AUROC vs temperature curve for both models
  - Write analysis: does sharper attention amplify conflict? where's the sweet spot?

---

## Phase 2: Typographic Variables (target: 2026-07-15)

### 2.1 Sensitivity to Attack Parameters
- **Code needed**: Generate STA-5K-Mini subsets stratified by each variable
- **Model**: Qwen2.5-VL-7B (representative)
- **Variables to test**:
  - Font size: 16, 20, 24, 28, 32 px
  - Color contrast: Low (<20 ∆E), Medium (20–50), High (≥50)
  - Text opacity: 30%, 50%, 70%, 90%
  - Overlay position: Center, Corner, Random
  - Font family: Serif, Sans-serif, Monospace, Decorative
- **Deliverables**:
  - Fill `tab:typo_variables` (AUROC per condition)
  - Plot `fig7_typo_variables_auroc.pdf`: 5-panel bar chart
  - Write analysis: which variables affect ACF most? detection boundary conditions?

---

## Phase 3: MoF & Generalization (target: 2026-07-18)

### 3.1 MoF Feature Interleaving
- **Code needed**: Extract CLIP + DINOv2 features separately, simulate interleaved token arrangement
- **Models**: LLaVA-NeXT-7B, Qwen2.5-VL-7B (augmented with DINOv2), MoF-Models-7B (native)
- **Dataset**: STA-5K-Mini
- **Deliverables**:
  - Fill `tab:mof_interleave`
  - Write analysis: does interleaving boost other models?

### 3.2 MoF DINO Ablation
- **Code needed**: Inference with CLIP-only / DINO-only token subsets (mask or re-extract)
- **Model**: MoF-Models-7B
- **Dataset**: STA-5K-Mini
- **Note**: If full re-extraction requires retraining, use token masking at inference time
- **Deliverables**:
  - Fill `tab:mof_dino`
  - Write analysis: is DINO essential or complementary?

### 3.3 Misbehavior Generalization
- **Code needed**: Run ACF on non-typographic datasets
- **Model**: Qwen2.5-VL-7B
- **Datasets**: MM-SafetyBench (visual jailbreak subset), visual hallucination data (Huang et al. 2025)
- **Deliverables**:
  - Fill `tab:misbehavior`
  - Write analysis: does ACF generalize? compare conflict distributions across misbehavior types

---

## Phase 4: Method-Data Affinity Framework (target: 2026-07-18)

**核心问题**: 每种不确定性检测方法在什么类型的数据上强/弱？给出明确的判定准则，让后续研究者知道什么时候该用什么方法。

### 4.1 方法分类

| 类别 | 方法 | 原理 | 操作空间 |
|------|------|------|----------|
| **表征层 / 结构方法** | Conflict (ACF) | 检测视觉 token 间的内部表征冲突 | hidden states |
| **表征层 / 结构方法** | Entropy | 检测预测分布的高熵/平坦性 | logits |
| **表征层 / 结构方法** | Ignorance (OOD) | 检测输入偏离训练分布的密度 | hidden states / features |
| **输出层 / 语义方法** | Verbalize (口头化) | 让模型显式输出 confidence/uncertainty | token generation |
| **输出层 / 语义方法** | Semantic Entropy | 对语义等价 cluster 的熵 | token generation + clustering |

### 4.2 数据类型划分维度

按以下维度对样本进行分桶，观察每种方法在各桶上的表现差异：

#### 维度 1: Epistemic vs Aleatoric Uncertainty
| 类型 | 定义 | 数据特征 | 预期强的方法 |
|------|------|----------|:----------:|
| **Aleatoric (数据固有不确定性)** | 图像本身模糊、歧义、多义 | 低质量图像、一个图可对应多个类 | Entropy, Ignorance |
| **Epistemic (模型认知不确定性)** | 模型知识不足、表征冲突 | OOD 样本、对抗攻击、分布外 | Conflict, Ignorance |

#### 维度 2: Known-Unknown vs Unknown-Unknown
| 类型 | 定义 | 数据特征 | 预期强的方法 |
|------|------|----------|:----------:|
| **Known-Unknown** | 模型"知道"自己不确定（预测熵高） | 困难样本但仍在分布内 | Entropy, Verbalize |
| **Unknown-Unknown** | 模型"不知道"自己错了（高置信度错误） | 对抗样本、typo 攻击 | **Conflict (ACF)** |

#### 维度 3: 表征冲突 vs 输出平滑
| 类型 | 定义 | 数据特征 | 预期强的方法 |
|------|------|----------|:----------:|
| **表征冲突型** | 视觉 encoder 内部 token 间存在不一致 | typo 攻击（图像+文字冲突）、多模态不一致 | **Conflict (ACF)** |
| **输出平滑型** | 预测概率分布均匀但表征无冲突 | 细粒度分类、相似类别歧义 | Entropy |
| **表征平滑输出也平滑** | 模型完全不确定 | OOD、噪声图 | Ignorance, Entropy |
| **表征平滑输出尖锐** | 模型过拟合/记忆化，但实际上是错的 | 某些幻觉场景 | 难检测（需外部知识） |

#### 维度 4: 可通过自然语言表达 vs 不可表达
| 类型 | 定义 | 数据特征 | 预期强的方法 |
|------|------|----------|:----------:|
| **语言可表达** | 模型可以通过语言表达其不确定性 | 清晰的二选一、可推理的歧义 | Verbalize, Semantic Entropy |
| **语言不可表达** | 模型无法正确口头化其内部状态 | 内部表征冲突（但 logits 正常） | **Conflict (ACF)**, Entropy |

### 4.3 实证验证方案

对 STA-5K-Mini 样本按上述维度进行分桶，计算每种方法在各桶上的 AUROC/漏检率：

#### 桶划分策略
1. **按预测熵** (高熵 / 中熵 / 低熵) — 桶内比较 Conflict vs Entropy vs Ignorance
2. **按 logit gap** (gap 大 / 中 / 小) — logit gap 小 = 模型不确定，大 = 模型有信心
3. **按 typo 攻击强度** (强攻击 / 弱攻击 / 无攻击) — Conflict 在强攻击下是否更有效？
4. **按图像-文本冲突程度** (高冲突 / 低冲突 / 一致) — 人工标注或按 CLIP score 分
5. **按 OOD 程度** (in-distribution / near-OOD / far-OOD) — Ignorance 在 OOD 上是否最优？

#### 预期结论（待验证）
```
┌─────────────────────────────────────────────────────────┐
│  高冲突 + 低熵 (typo攻击典型场景)                          │
│  → Conflict (ACF) 最强                                   │
│  → Entropy 可能失效（模型对攻击文本"信心十足"）              │
├─────────────────────────────────────────────────────────┤
│  低冲突 + 低熵 (困难但自然样本)                            │
│  → Entropy 或 Semantic Entropy 最佳                       │
│  → Conflict 检测不到（无内部表征冲突）                      │
├─────────────────────────────────────────────────────────┤
│  高冲突 + 高熵 (OOD + 攻击叠加)                            │
│  → Ignorance 或 Conflict 均可                            │
├─────────────────────────────────────────────────────────┤
│  低冲突 + 高熵 (OOD/噪声)                                 │
│  → Entropy / Ignorance 最佳                               │
│  → Conflict 可能无效                                      │
└─────────────────────────────────────────────────────────┘
```

### 4.4 Deliverables

- **Table**: `tab:method_data_affinity` — 每种方法在各数据分桶上的 AUROC/AUPR
- **Figure**: `fig8_affinity_heatmap.pdf` — 热力图 (方法 × 数据桶 × AUROC)
- **Figure**: `fig9_affinity_radar.pdf` — 雷达图，每种方法在不同维度上的"能力 profile"
- **Analysis text**: 讨论每种方法的适用边界，给出方法选择指南

### 4.5 方法选择决策树（输出）

```
需要检测模型何时出错？
├─ 是否有明确的视觉-语义冲突？
│   ├─ 是 → Conflict (ACF)
│   └─ 否 → 继续
│       ├─ 模型预测有高熵？→ Entropy / Semantic Entropy
│       ├─ 样本是 OOD？→ Ignorance (density-based)
│       └─ 可以用自然语言表达不确定性？→ Verbalize
```

---

## Phase 5: System & Polish (target: 2026-07-20)

### 5.1 Runtime Benchmark
- **Code needed**: Profile forward pass + post-processing time for all 6 methods
- **Hardware**: NVIDIA A100-80GB, CPU post-processing
- **Metrics**: Forward pass (ms), Post-processing (ms), Total (ms)
- **Deliverables**: Fill `tab:runtime`

### 5.2 ACF Distribution Plot
- **Code needed**: Collect ACF scores for all clean/attacked samples
- **Models**: All 5 (MoF-Models-7B for main figure, others for supplementary)
- **Datasets**: STA-5K, RTA-100
- **Deliverables**: Plot `fig4_acf_distribution.pdf` (histogram/KDE overlay)

### 5.3 Writing Polish
- Expand all `Analysis to be completed` sections with real findings
- Refine Introduction and Related Work language
- Polish Discussion and Conclusion
- Add figure/table cross-references in analysis text
- Check all citations are correctly placed
- Proofread for language quality

---

## Timeline Summary

```
Jul 07 ─┬─ Phase 1 start: Layer-wise + Temperature
        │
Jul 12 ─┼─ Phase 1 done ─── Phase 2 start: Typographic Variables
        │
Jul 15 ─┼─ Phase 2 done ─── Phase 3 start: MoF + Generalization
        │
Jul 18 ─┼─ Phase 3 done ─── Phase 4 start: Method-Data Affinity
        │
Jul 20 ─┼─ Phase 4 done ─── Phase 5 start: Runtime + Distribution + Polish
        │
Jul 22 ─┼─ ALL EXPERIMENTS DONE
        │
Jul 20─25  Writing polish, proofread, finalize first draft
        │
Jul 25 ─┴─ FIRST DRAFT READY
```

---

## Checklist

### Tables (8 to fill)
- [ ] `tab:layer_wise` — Layer-wise AUROC/AUPR
- [ ] `tab:temperature` — Temperature vs ACF
- [ ] `tab:typo_variables` — Attack variable sensitivity
- [ ] `tab:mof_interleave` — Interleaving effect
- [ ] `tab:mof_dino` — DINO ablation
- [ ] `tab:misbehavior` — Generalization to other tasks
- [ ] `tab:runtime` — Runtime comparison
- [ ] `tab:method_data_affinity` — Method × Data bucket AUROC/AUPR

### Figures (6 to generate)
- [ ] `fig4_acf_distribution.pdf` — ACF score histogram
- [ ] `fig5_layerwise_acf.pdf` — Layer-wise ACF + AUROC
- [ ] `fig6_temperature_auroc.pdf` — Temperature curve
- [ ] `fig7_typo_variables_auroc.pdf` — Variable sensitivity panels
- [ ] `fig8_affinity_heatmap.pdf` — Method × Data bucket heatmap
- [ ] `fig9_affinity_radar.pdf` — Method capability radar profile

### Analysis Text (6 sections to write)
- [ ] Layer-wise ACF analysis
- [ ] Temperature analysis
- [ ] Typographic variable analysis
- [ ] MoF effectiveness analysis (2 hypotheses)
- [ ] Misbehavior generalization analysis
- [ ] Method-data affinity analysis (适用边界 + 方法选择指南)

### Writing Polish
- [ ] Introduction refinement
- [ ] Related Work coherence check
- [ ] Discussion & Limitations review
- [ ] Conclusion alignment with contributions
- [ ] Full proofread
