# 核心任务：保留 attention-level ACF，提高 MMMC AUROC

用户明确约束：不使用 LM 或 A/B 输出头替换核心；新版 evidence 保留潜在假设框架。旧 envidence.py 保留，后续新计算不再使用。

1. complete: 建立 evidence_v2，统一[K假设,J特征]方向、逐项证据、固定校准均值、稳定DST公式，保留零证据；60例独立数学验证通过。
2. complete: 本次目录分为7个任务，均有code/results/logs/notes，原始数据迁移SHA验证通过。
3. in_progress: 用原始问题和图像提取全部32层、attention后FFN投影输入，不使用输出logit或冲突判别提示。
4. pending: 固定开发集均值，对比无均值/开发集均值标定、层位置和全局证据温度；独立验证集报告AUROC，原始CF与数值等价logscore分别标注。
5. pending: 选择ACF改进配置并在未参与选择的新样本上检验；给出结果及旧版错误分析。

已有结果应谨慎解释：新版标定后，原始问题的L0首token AUC0.51456、L31为0.50169（400条验证问题）。A/B分类头CF0.65261属于额外对照，用户明确排除为主方法。提示集成0.83430仅归档对照，不视为核心ACF的改进成功。

Implementation status: extraction completed for dev200 and validation400 records, all32 post-attention FFN layers. Canonical evidence_v2 includes shared stable combine_evidence used by all new sweeps. Started32-layer × zero/dev-mean calibration × temperatures[.25,.5,1,2,4,8] scan. Both raw CF atT1 and monotone -log(1-CF) rankings retained; scale selection only on dev, validation used for checking generalization. No class-head scores enter this scan.

用户最新约束：固定第0层作baseline，不再做block选择；之后所有汇报仅给第0层。停止跨层扫描，保留历史结果；新位置脚本默认layers=0。第0层zero/T1验证400条：wo CF0.49269，w1 CF0.48396，w3 CF0.48479，w2 CF0.48011。
