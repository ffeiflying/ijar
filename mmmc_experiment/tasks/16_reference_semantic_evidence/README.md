# 单次前向：背景参照的内部语义CF

当前实验候选，固定配置从开发集选出：InternVL2.5-8B，block24，attention后视觉残差，冻结归一化词表读出，开发图像背景参照A1。不生成回答，不做多次采样，不修改模型权重或环境。

## 方法

对英文问题中的有效词片段，使用其query在block24的attention定位视觉patch，并对视觉隐藏状态读出的词概率加权。将当前图像支持与100张开发图像的背景概率比较，得到平均log支持比ell。

二元简单支持质量：ell>=0时，m(P)=1−exp(−ell)，m(Theta)=exp(−ell)；ell<0时，m(notP)=1−exp(ell)，m(Theta)=exp(ell)。文字前提表示期望图像提供支持，对应m_text(P)=1，因此冲突CF=m_visual(notP)。记录稳定排序分数−log(1−CF)，同时返回无知质量和支持质量。

P目前是“当前图片给出的词汇支持超出一般背景”的近似假设，不等价于已经验证的对象存在命题。支持比依赖近似背景模型和词片段，尚未建立完整语义逻辑假设空间。CF不是最终回答概率。

## 离线与在线成本

离线：100张开发图片估计背景分布，开发集选择配置；不使用回答生成或采样，背景估计不读类别标签。

在线：一张图片+一个问题，一次视觉编码和一次语言prefill，在block24 attention后、FFN前退出。冻结unembedding只解释视觉中间状态，没有生成最终回答。`detect.py`审计前向计数。

```bash
/data/liuxiaofei/miniconda3/envs/internvl/bin/python -s \
  mmmc_experiment/tasks/16_reference_semantic_evidence/code/detect.py \
  --records /path/to/records.json \
  --output mmmc_experiment/tasks/16_reference_semantic_evidence/results/my_predictions.jsonl \
  --gpu 0
```

records为JSON列表，每条只需row_id、image_id、image_path、question，无需标签或答案。输出文件支持跳过已完成row_id；既有配置不一致时拒绝覆盖。

## 结果

- `results/RESULTS.md`：迭代验证结果，开发选定CF验证AUROC0.7216。
- `results/deployment_verification.json`：400条单次前向与离线分数一致性及同配置层对照。
- `results/RESEARCH_STATUS.md`：此前证据分配失败、同层基线与限制。
- `results/confirmation/protocol.json`：600条全新图片确认协议，预测前冻结。
- `results/confirmation/RESULTS.md`：主要确认结果；300张此前未使用图片、600条问题，block24 CF AUROC0.7400，同配置block0为0.5910。

只写本任务目录。task14/15用于对照和历史尝试，全部保留；其他工作的results未改动。尚不声称优于所有匹配基线或具有已验证的ICCV新颖性。
