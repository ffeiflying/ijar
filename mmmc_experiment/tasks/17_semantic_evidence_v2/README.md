# 内部语义证据 V2：目标 AUROC 约0.85

本任务独立保存代码、数据、结果、日志和研究笔记，不修改旧任务结果，也不修改 internvl 环境。

- [旧版完整定义](notes/METHOD_V1.md)：独立确认集 CF AUROC0.740；它不是已验证的创新贡献。
- [创新性对照与待证条件](notes/NOVELTY_AUDIT.md)：单前向、内部特征、DST、视觉logit lens均有先例，需要更具体的贡献和消融。
- [新版候选及公式](notes/METHOD_V2.md)：命题级质量分配，保留支持、反对、未知；同时测试普通分类器。
- `results/v1_archive/`：旧版只读快照ZIP及逐文件SHA256。
- `results/atomic_evidence/`：开发集冻结配置与工作验证集结果。当前新版无训练CF0.722，并未改善旧版。
- `data/train_protocol.json`：新训练/校准划分、来源与排除全部测试图片的校验。
- `code/train_mil.py`：仅用官方train训练和选择的证据MIL、普通MIL、全局探针及纯文本控制。

0.85是目标，不是已达到的成绩。后续结果写入本任务；不使用旧600条确认集挑选新版。
