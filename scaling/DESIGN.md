# 模型能力 vs 不确定性量化能力（ACF/AUROC 缩放实验）

**日期**: 2026-10-12
**目标**: 验证"模型能力越强（按参数量），第 0 层 ACF 的不确定性量化能力（AUROC）越强"。

## 方法（与现有 IJAR 实验完全一致）

1. 模型对每条样本（图片 + 二选一问题）做贪心生成。
2. 逐生成步抓取**每一层文本解码器 MLP `down_proj` 的输入**（激活后向量，维度 = intermediate_size），取最后一个位置。
3. 用该层 `down_proj` 权重矩阵 `B (J=hidden, K=intermediate)` 配合 `envidence.py` 的 DST 证据分解算冲突 κ（ACF）与 ignorance。
4. 层内对生成步取均值 → `per_layer_conflict`。
5. 标签：`答错 = 1`；分数：该层 ACF → 逐层 AUROC / AUPR。**同时记录 accuracy**（供后续配对/预算可控分析）。

## 数据

- `datasets/STA/sta_mini.csv`（500 条）+ `datasets/RTA/rta_mini.csv`（500 条），与现有 4 个模型的跑法一致。

## 环境分工

| env | transformers | `qwen2_5_vl` | `qwen3_vl` | `generate` 补丁 |
|---|---|---|---|---|
| `qwen` | 4.51.1 | ✅ 已打补丁 | ❌ 模块不存在 | ✅ |
| `EUQ` | 5.3.0 | ✅ 干净 | ✅ 干净 | ❌ |

- **Qwen2.5 系列 → `qwen` env**：原版 `layerauroc/run_qwen.py` 可用（依赖该补丁）。
- **Qwen3-VL 8B/32B → `EUQ` env**：transformers 干净但自带 `qwen3_vl`；用 **forward hook** 抓 `down_proj` 输入，**不修改任何 conda 环境**。

## 抓取方案（forward hook + 正常 `model.generate`）

在每层 `layers[i].mlp.down_proj` 上挂 `register_forward_pre_hook`。`generate` 内部
prefill 一次 + 每生成一个 token 一次 forward，hook 每次都触发，取 `args[0][:, -1, :]`。
语义与原补丁（取 `layer_outputs[-1]` 再 `[:, -1:, :]`）一致，但零环境改动、零生成循环重写。

## 权重实时读取

不再产出 `.pth`，直接从 `down_proj.weight` 取。Qwen3-VL 路径 `model.model.language_model.layers[i]`，
Qwen2.5-VL 路径 `model.model.layers[i]`（脚本自动探测）。
验证阶段顺带比对实时权重与已有 `.pth` 是否逐位相同。

## 性能修正（对 32B 必需）

原 `EvidenceModel.get_evidence_weights` 里
`W = ((B_star @ M) / J).expand(J, K).to(float64)` 会把 (J,K) 实体化成 float64：

- 7B: 3584×18944×8B ≈ 28MB/张量，×3 个/层/token，尚可忍受；
- **32B: 5120×25600×8B ≈ 1GB/张量 ×3 ×64 层 —— 照搬跑不动。**

观察：`W[t, j, k] = w[t, j]` 与 `k` 无关（K 列完全相同），且下游只用行和，故可等价化简为
O(J) 闭式。设 `w[j,t] = (B_star[j] · Phi[:,t]) / J`，`P[t] = Σ_j relu(w[j,t])`，
`N[t] = Σ_j relu(-w[j,t])`，`Ki = K`：

- `w_pos1[t,k] = P[t]`（对所有 k），`w_neg1[t,k] = N[t]`
- `eta_pos[t] = 1 / (Ki·e^{P[t]} - Ki + 1)`
- `eta_neg[t] = 1 / (1 - (1 - e^{-N[t]})^Ki)`
- `P'[t] = P[t]`（若 P==0 则原实现会 IndexError；这里退化取 0 → κ=0）
- `N'[t] = N[t]`（若 N==0 则原实现取 1e-6）
- `kappa[t] = Ki · eta_pos[t] · (e^{P'[t]} - 1) · (1 - eta_neg[t]·e^{-N[t]})`
- `eta_temp[t] = 1 / (1 - kappa[t])`
- `ign[t] = Ki · e^{-N'[t]} · eta_temp[t] · eta_pos[t] · eta_neg[t]`

matvec 在权重 dtype（fp16）下做，随后转 float64 —— 与原实现一致（原实现也是 fp16 算 `B_star@M`，
之后才 `.to(float64)`）。为省内存不实体化 `B_star`，用
`B_star @ Phi == B @ Phi - m ⊗ colsum(Phi)`（`m = B.mean(dim=1)`，加载时预计算一次）。

## 验证顺序（先验证再上 32B）

1. **EUQ + hook 版跑 Qwen2.5-VL-7B** → 第 0 层 AUROC 对齐现有 `results/layerauroc/STA_layerauroc_qwen.json`；
   同时比对实时权重 ≡ 已有 `.pth`。→ 证明 hook 抓取 + 闭式化简都忠实。
2. **qwen env + 原版 `run_qwen.py` 重跑 Qwen2.5-VL-7B**（输出重定向到本目录，**不覆盖**已有 JSON）
   → 证明原管线本身可复现。
3. 都过了再跑 Qwen3-VL-8B → Qwen3-VL-32B。

## 产出

- 每模型×数据集一个 JSON，沿用原 schema（`per_layer_conflict` / `per_layer_ignorance` /
  `per_token_conflict` / `model_output` / `ground_truth` / `n_tokens`），另加 `pred` 与 `correct`。
- 每模型一个 `*_meta.json`：模型名、参数量、层数、dtype、生成配置、样本数、时间戳。
- 汇总表：参数量 / acc / 第 0 层 AUROC·AUPR / 逐层曲线图。

## 已知风险

- 32B fp16 ≈ 63GB，单卡 79GB 余量小；必要时改 bf16 或双卡切分。
- RTA 上模型本就 ~90% 准，强模型错误样本更少，500 条下 AUROC 方差大 → 存 acc 支撑后续分析。
- a/b 选项抽取（`_extract_choice` 原样复用）若与强模型输出格式不匹配，会把正确回答误判成错 →
  需抽查 32B 原始输出。

## 边界

- 本目录内新建文件；**不改动 `IJAR/` 下任何现有文件**。
- 结果写到 `scaling/results/`，避免覆盖 `results/layerauroc/` 下的已有 JSON。
