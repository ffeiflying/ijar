# Single-forward internal evidential conflict

新任务，2026-10-25。保留旧工作全部代码和结果；本任务仅写当前目录。环境为原 internvl，不安装或更换依赖。

## 实验

原始图片+问题，InternVL2.5-8B；block从0编号，候选0/8/16/20/24/27/31。每个输入只做一次prefill，在最后一个被检查的block退出，不生成回答。所有层分数在同一次forward中读取，实际部署可在选定层退出。

开发200条按图片与验证400条互斥。用开发问题的词汇token建立固定候选frame，不读取answer、label、key_component。此frame是词汇读出类别的近似，不是已经验证的语义等价类。

## 候选

1. `feature_DS`：对attention后视觉残差做冻结归一化词汇读出，使用新版Denoeux式线性特征正/负证据分解。由attention对patch统计加权，或top3 patch混合。
2. `source_DS`：重建同一query的视觉/text attention贡献，经W_O和共享query残差，读出到同一frame，各自建立证据后计算精确跨来源DS冲突。两来源相关，因此不能仅凭公式正确宣称独立证据成立。
3. `source_discounted`：来源词汇分布以1−归一化熵折扣，未分配质量留给Theta，计算单元素交叉冲突。
4. `head_conflict`：不同head关注区域在同一frame上的折扣BPA，统计平均两两冲突；不把head自动视为独立。
5. `lexical_grounding`：词汇前提与关注区域折扣BPA的冲突，属于内部锚定代理，不能证明全图对象不存在。
6. `matching_baseline`：视觉内部词表对问题词的负对数支持，不使用证据理论，用于核查CF是否提供额外价值。

高分统一表示更可能冲突；不根据验证集翻转符号。`cross_cf`和`acf`主排序分数为稳定−log(1−CF)，不是原始CF。折扣质量仍是启发式，不能声称经过语义校准。

## 产物

- `results/frame.json`：冻结词汇frame。
- `results/features/`：初版4条smoke，保留。
- `results/features_v2/`：正式候选特征；新增v2，不覆盖初版。
- `results/math_verification.json`：独立小frame计算检查。
- `results/frozen_dev_selection.json`：完整开发集选型，验证前冻结。
- `results/validation_frozen_metrics.json`、`results/RESULTS.md`：冻结验证结果。
- `notes/`：方案、发现、进度。

全部读出仅使用模型内部特征，复用冻结unembedding解释中间状态；不以最终回答的输出分布作为检测器。
