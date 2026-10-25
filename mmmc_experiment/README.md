# MMMC modality-conflict experiments

所有任务均按 `code/`、`results/`、`logs/`、`notes/` 分开。

| 任务目录 | 内容 | 主要结果 |
|---|---|---|
| tasks/01_dataset | 下载、校验、样本划分 | results/；共享样本在 shared/data/ |
| tasks/02_layer0 | 第0层实验；新运行使用 evidence_v2 | results/RESULTS.md（历史v1） |
| tasks/03_last_layer | 最后一层和两层对比 | results/RESULTS.md（历史v1） |
| tasks/04_auroc_improvement | 历史取层、提示和融合探索，对照归档 | results/*_metrics.json |
| tasks/05_cf_correction | 公式审查、新版 evidence、正确性验证 | code/evidence_v2.py；results/AUDIT.md |
| tasks/06_organization | 目录整理与原始文件完整性核对 | results/verification.json |
| tasks/07_attention_acf | 后续核心 ACF 实验 | results/ |

后续主方法保留 attention 后 FFN 特征上的 ACF，不采用 LM/A-B 输出头替代。新的计算统一使用 `tasks/05_cf_correction/code/evidence_v2.py`。项目根目录原 `envidence.py` 保留不改，历史 v1 结果只作审计，不与 v2 混报。A/B/提示集成仅是已完成的对照实验。

`shared/data/MMMC/` 保存校验过的官方测试集；`shared/data/splits/` 保存按图片互斥的 dev/validation/final 划分；`shared/cache/` 保存模型动态代码与绘图缓存。指定的 `/data/datasets` 仍受系统目录权限限制，数据暂存于本目录。

整理时未重写历史原始结果和配置，配置里可能含原路径。`shared/relocation_manifest.json` 记录路径迁移；`shared/layout.py` 负责运行时解析旧路径。不要用旧日志里的路径直接启动新任务。

使用现有 `/data/liuxiaofei/miniconda3/envs/internvl/bin/python -s`，未安装、升级、卸载任何包。查看各任务 README/notes 获取实验细节。

最新任务：
- `tasks/08_feature_positions/`：第0层attention输出投影、FFN支路与原位置对比；结果 `results/layer0_baseline.json`。默认仅第0层，跨block扫描已停止。
- `tasks/09_layer0_modality/`：第0层attention视觉/文本来源分解与跨来源CF；理论说明 `notes/THEORY.md`。

当前用户约束优先：固定第0层；后续汇报只返回第0层值。历史其他层结果只归档，不作为选择依据。
