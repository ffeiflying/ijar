# CF 反传热力图可视化设计（IJAR）

日期：2026-08-31
状态：已批准（用户已确认设计）

## 目标

把算出的 ACF/冲突因子（CF = kappa，来自 DST 证据分解）反传到排版攻击图像上，观察模型"主要关注哪个区域"、冲突集中在图像的哪块区域。每张样本图输出三格并排：

1. 原图（排版攻击图像）
2. CF 梯度热力图（saliency，从 CF 值反传）——"模型认为哪块区域对冲突贡献最大"
3. CF 值热力图（逐图像 token 的 kappa）——"哪块图像区域的实际冲突值最高"

两张图互相印证：若都集中在排版文字区域，则支持"ACF 确实捕捉到图像中的攻击文字"这一论点。

## 范围

- 仅 Qwen2.5-VL-7B（管线最完整，patch 过的 generate 直接返回逐层 down-proj 输入）。
- 仅第一个实验版本；后续是否迁移到 InternVL/MoF/DeepSeek 另定。
- 不修改现有代码（envidence.py、typo-parameter、selectlayer 均不动）。
- 不修改任何环境（conda env、site-packages、transformers）。约束来自用户："先别改我的环境，要改环境的时候让我看一下改的哪里"。

## 数据流

全部复用现有管线：

1. 用 Qwen env 中 patch 过的 `model.generate` 拿到 `all_input_down_proj`（生成每一步、每层 MLP down-proj 输入张量）。取 **prefill 步（第 0 步）**，该步序列含完整 prompt：张量形状 `[prompt_len, intermediate]`。
2. **逐 token CF**：对选中层 L11 的 down-proj 输入张量的每一行（每个 token 位置）计算 kappa。图像 token 位置用 `input_ids == image_token_id` 定位（Qwen 中图像 token 为连续的 `<|image_pad|>` 段）。
3. **token→空间网格**：`image_grid_thw` 给出 (grid_h, grid_w)，单图时 grid_t=1。图像 token 的 CF 按行主序 reshape 成 (grid_h, grid_w)，双线性上采样到原图分辨率叠加。448px 图像 → 16×16=256 个图像 token。

## 可微 CF（GradCAM 反传的关键）

现有 `EvidenceModel.get_evidence_weights` 内含 `torch.no_grad()` 与 `torch.nonzero`（不可微），无法直接对 CF 反传。方案：在新脚本里写一个**等价的、可微的闭式版本**，不触碰 envidence.py。

数学化简：对单个 token 的 down-proj 输入 x（K 维），固定权重 B（形状 [J, K] 的 down-proj 权重），有

- w = B̃ᵀx / J（B̃ = B − mean(B, axis=1)）
- u = Σⱼ relu(w)ⱼ,  v = Σⱼ relu(−w)ⱼ
- eta_pos = 1/(K·exp(u) − K + 1)
- eta_neg = 1/(1 − (1 − exp(−v))^K)
- kappa = K · eta_pos · (exp(u) − 1) · (1 − eta_neg·exp(−v))

全由 relu/exp/sum 组成，可微。对一整批 token 可用矩阵运算一次性算出（每个 token 的 kappa），与逐 token 调用现有 EvidenceModel 数值一致（zero-substitution 在连续隐状态下几乎不触发，可安全省略或加极小 epsilon）。

## GradCAM 反传流程

1. 普通前向 `model(...)`（开 grad），pixel_values 需梯度。
2. forward hook 挂在视觉塔最后一个 block 的输出（patch 级特征，32×32，在 merger 之前），得到目标激活图。
3. 从前向返回的 `all_input_down_proj[L11]` 提取图像 token 行，用可微 CF 计算每个图像 token 的 kappa。
4. 标量 `S = mean(图像 token 的 kappa)`，`S.backward()`。
5. `dS/d(vision_features)` 按 GradCAM 公式（全局平均梯度 × 激活）加权 → 空间显著性图 → 上采样到原图。

## 默认参数

- 模型：Qwen/Qwen2.5-VL-7B-Instruct（cache `/data/public/models/Qwen`）
- 层：L11（selectlayer 选出的最优层）
- 样本：STA 攻击集采 4~6 张（可配置，随机种子固定；含答对/答错样本）
- 配色：CF 值热力图用蓝→红发散色（低冲突→高冲突）；CF 梯度热力图用顺序色；均叠加在图像上、带 colorbar
- 输出目录：`results/cf_heatmap/`，每张样本输出 PNG/PDF（三格并排）

## 文件

新增 `ablation/cf_heatmap/run_cf_heatmap_qwen.py`（单文件）：
- 可微 CF 计算（独立函数，等价于 EvidenceModel 的 kappa）
- GradCAM（视觉塔特征 hook + S.backward）
- 逐 token CF 值热力图
- 三格并排出图 + colorbar

输入数据路径：
- STA 图像：`datasets/STA/images/`
- STA 索引：`datasets/STA/sta_mini.csv`
- down-proj 权重：`/data/liuxiaofei/uncertainty/Qwen2.5-VL/weight/QWen_MLP_down_proj_{layer}_weight.pth`

## 验收

- 能对任意一张 STA 攻击图像生成三格图。
- CF 值热力图能按图像 token 网格上采样对齐到图像（排版文字区域对应高/低冲突可视化）。
- CF 梯度热力图从 CF 反传（非 logit 反传）。
- 输出 PNG/PDF 保存到 results/cf_heatmap/。
