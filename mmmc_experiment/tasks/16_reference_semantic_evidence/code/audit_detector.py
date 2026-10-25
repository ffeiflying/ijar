"""Parity, forward-count and fixed-configuration layer comparison audit."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
TASK=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('reference_evaluation',TASK/'code/evaluate.py')
reference=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
from transformers import AutoTokenizer

rows,features=reference.load('validation')
tok=AutoTokenizer.from_pretrained(reference.MODEL,trust_remote_code=True,use_fast=False,local_files_only=True)
with np.load(TASK/'results/background.npz') as z:
    background={k:z[k] for k in z.files}
scores,masses=reference.make_scores(rows,features,tok,background)
frozen=json.loads((TASK/'results/frozen_dev_selection.json').read_text())
predictions={r['row_id']:r for r in map(json.loads,(TASK/'results/detector_validation.jsonl').read_text().splitlines())}
assert len(predictions)==len(rows)==400
assert all(r['language_forwards']==r['vision_forwards']==1 for r in predictions.values())
actual=np.array([predictions[r['row_id']]['cf_logscore'] for r in rows])
expected=scores[frozen['best_cf']]
error=np.max(np.abs(actual-expected))
assert error<1e-4,('Deployment/offline score mismatch',error)
y=np.array([r['label'] for r in rows])
times=np.array([predictions[r['row_id']]['seconds'] for r in rows])
comparison={}
for layer in reference.LAYERS:
    key=f'L{layer}/attention/A1/cf'
    comparison[str(layer)]={'key':key,**reference.metrics(rows,scores[key],bootstrap=layer in [0,24])}
groups={}
for i,row in enumerate(rows):
    groups.setdefault(row['image_id'],[]).append(i)
indices=list(groups.values())
rng=np.random.default_rng(42)
deltas=[]
for _ in range(1000):
    idx=np.concatenate([indices[i] for i in rng.integers(0,len(indices),len(indices))])
    deltas.append(roc_auc_score(y[idx],scores['L24/attention/A1/cf'][idx])-roc_auc_score(y[idx],scores['L0/attention/A1/cf'][idx]))
report={'rows':len(rows),'selected':frozen['best_cf'],'deployment_auroc':float(roc_auc_score(y,actual)),'max_abs_offline_logscore_error':float(error),'vision_and_language_forwards_per_input':1,'answer_tokens_generated':0,'early_exit':'block24 after attention, before FFN','median_seconds_excluding_first':float(np.median(times[1:])),'mean_seconds_excluding_first':float(times[1:].mean()),'p95_seconds_excluding_first':float(np.quantile(times[1:],.95)),'timing_note':'current machine/load, includes image loading/transform, prompt formatting, vision, language prefill and score; excludes model load and offline reference collection','fixed_configuration_comparison':comparison,'layer24_minus0_auc95ci':np.quantile(deltas,[.025,.975]).tolist(),'limitations':'iterative validation; approximate lexical-grounding hypotheses, no independent final confirmation'}
(TASK/'results/deployment_verification.json').write_text(json.dumps(report,indent=2))
lines=['# 当前研究检查点','', '中后层的内部语义读出比第0层更有检测信号。直接熵折扣CF泛化差，改成开发图像背景参照的简单支持质量后，CF改善。尚未证明比简单语义匹配更有判别优势，也未证明新颖性。','', '## 同配置层对照','', '所有层固定 attention加权读出、背景温度A1、相同二元简单支持质量；下面只改变block。开发集选中的主配置仍为block24。','', '| block（从0编号） | 验证CF AUROC |','|---|---|']
for layer,result in comparison.items():
    lines.append(f'| {layer} | {result["auroc"]:.4f} |')
lines+=['',f'主配置部署复算AUROC：{report["deployment_auroc"]:.4f}；离线排序分数最大误差{error:.3g}。400条每条均核验一次视觉编码和一次语言前向，未生成回答，在block24 attention后退出。',f'当前负载单条中位耗时{report["median_seconds_excluding_first"]:.3f}秒，不含模型加载和离线参考构建；未与sampling方法实测比较，因此不声称倍数加速。','', '## 结果解释','', '- 主结果为CF，稳定−log(1−CF)只用于避免饱和排序；同时保留无知Theta。','- 该二元假设近似区分当前图像的词汇支持与一般视觉背景支持，不能把背景支持自动当成逻辑否定或对象不存在。','- 纯内部匹配基线在验证集0.7327；本方法CF约0.7216，不能声称证据理论已超过匹配。','- 小型8问题/8图均衡机制对照证明分数会随图像匹配改变，但使用开发图片，不能代替独立验证。','- 本轮多次迭代使用验证集，最终结论必须在新的图片互斥数据上确认。','- 原有results与环境未修改，所有新产物保存在tasks/14、15、16中。']
(TASK/'results/RESEARCH_STATUS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'auroc':report['deployment_auroc'],'score_error':float(error),'median_seconds':report['median_seconds_excluding_first'],'layer_table':{k:round(v['auroc'],4) for k,v in comparison.items()},'auc24minus0_95ci':report['layer24_minus0_auc95ci']},indent=2))
