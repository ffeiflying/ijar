"""Evaluate frozen methods on previously unused images; no model selection."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score,roc_curve
TASK=Path(__file__).resolve().parent.parent
dest=TASK/'results/confirmation'
spec=importlib.util.spec_from_file_location('reference_evaluation',TASK/'code/evaluate.py')
ref=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
from transformers import AutoTokenizer
protocol=json.loads((dest/'protocol.json').read_text())
records_path=dest/'confirmation_records.json'
assert protocol['records_sha256']==hashlib.sha256(records_path.read_bytes()).hexdigest()
assert protocol['frozen_selection_sha256']==hashlib.sha256((TASK/'results/frozen_dev_selection.json').read_bytes()).hexdigest()
rows=json.loads(records_path.read_text())
pred={}
for layer in [0,24]:
    path=dest/f'block{layer}_predictions.jsonl'
    loaded={r['row_id']:r for r in map(json.loads,path.read_text().splitlines())}
    assert len(loaded)==len(rows)==600
    assert set(loaded)=={r['row_id'] for r in rows}
    assert all(r['layer']==layer and r['language_forwards']==r['vision_forwards']==1 for r in loaded.values())
    assert all(abs(r['cf']+r['ignorance']+r['support_premise']-1)<1e-9 for r in loaded.values())
    pred[layer]=loaded
tok=AutoTokenizer.from_pretrained(ref.MODEL,trust_remote_code=True,use_fast=False,local_files_only=True)
with np.load(TASK/'results/background.npz') as z:
    prior=np.log(np.maximum(z['L24'],1e-300))
    prior-=logsumexp(prior)
offset=[]
for row in rows:
    ids=tok.encode(row['question'],add_special_tokens=False)
    valid=[i for i in ids if ref.lexical(tok,i)]
    offset.append(prior[valid if valid else ids].mean())
score0=np.array([pred[0][r['row_id']]['cf_logscore'] for r in rows])
score24=np.array([pred[24][r['row_id']]['cf_logscore'] for r in rows])
contrast24=np.array([pred[24][r['row_id']]['contrast'] for r in rows])
nll24=contrast24-np.array(offset)
methods={'block0_CF':score0,'block24_CF':score24,'block24_plain_matching':nll24,'block24_unclipped_contrast':contrast24}
overall={name:ref.metrics(rows,s,bootstrap=True) for name,s in methods.items()}
y=np.array([r['label'] for r in rows])
groups={}
for i,row in enumerate(rows):
    groups.setdefault(row['image_id'],[]).append(i)
rng=np.random.default_rng(42)
group_indices=list(groups.values())
deltas={'CF24_minus_CF0':[],'CF24_minus_plain_matching24':[]}
for _ in range(1000):
    ix=np.concatenate([group_indices[i] for i in rng.integers(0,len(group_indices),len(group_indices))])
    auc=roc_auc_score(y[ix],score24[ix])
    deltas['CF24_minus_CF0'].append(auc-roc_auc_score(y[ix],score0[ix]))
    deltas['CF24_minus_plain_matching24'].append(auc-roc_auc_score(y[ix],nll24[ix]))
subgroups={}
for kind in ['object','attribute','relation']:
    selected={r['image_id'] for r in rows if r['conflict_type']==kind}
    ix=[i for i,r in enumerate(rows) if r['image_id'] in selected]
    subgroups[kind]={'images':len(selected),'rows':len(ix),'block0_CF':ref.metrics([rows[i] for i in ix],score0[ix]),'block24_CF':ref.metrics([rows[i] for i in ix],score24[ix])}
timings={}
for layer,loaded in pred.items():
    times=np.array([loaded[r['row_id']]['seconds'] for r in rows])
    timings[str(layer)]={'median_seconds_excluding_first':float(np.median(times[1:])),'mean_seconds_excluding_first':float(times[1:].mean()),'p95_seconds_excluding_first':float(np.quantile(times[1:],.95))}
report={'fresh_confirmation_images':300,'rows':600,'excludes_all_prior900_images':True,'frozen_before_predictions':True,'overall':overall,'paired_image_auc_difference95ci':{k:np.quantile(v,[.025,.975]).tolist() for k,v in deltas.items()},'by_conflict_type':subgroups,'timing':timings,'single_vision_and_language_forward_per_input_verified':True,'answer_generation':False,'interpretation':'plain matching and contrast are ablations at same block, not alternative test-selected methods; no further tuning on this confirmation set','limitations':['single model and dataset','lexical grounding is approximate semantic evidence','CF does not outperform unclipped contrast by construction in general','performance does not establish novelty or evidence-specific gains']}
(dest/'metrics.json').write_text(json.dumps(report,indent=2))
os.environ['MPLCONFIGDIR']=str(TASK/'results/plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
for name,score in [('Block 0 CF',score0),('Block 24 CF',score24)]:
    fpr,tpr,_=roc_curve(y,score)
    ax.plot(fpr,tpr,label=f'{name}: {roc_auc_score(y,score):.3f}')
ax.plot([0,1],[0,1],color='gray',linestyle='--',linewidth=1)
ax.set(xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1),title='Fresh confirmation: 300 image pairs')
ax.legend(loc='lower right')
fig.savefig(dest/'roc.png',dpi=180)
plt.close(fig)
lines=['# 全新图片确认：冻结的内部语义CF','', '从官方数据剩余1100张未使用图片中，以固定seed20261025抽取300张、600条配对问题。与之前开发、验证、历史final/acf_test共900张图片互斥。确认前冻结主方案block24和同配置block0；不再用确认集选层或调参。','', '| 方法 | AUROC | 95%图片bootstrap区间 |','|---|---|---|']
for name,result in overall.items():
    lo,hi=result['image_bootstrap_auroc95ci']
    lines.append(f'| {name} | {result["auroc"]:.4f} | [{lo:.4f}, {hi:.4f}] |')
lines+=['', '## 冲突类型','', '| 类型 | 图片对 | block0 CF | block24 CF |','|---|---|---|---|']
for kind,result in subgroups.items():
    lines.append(f'| {kind} | {result["images"]} | {result["block0_CF"]["auroc"]:.4f} | {result["block24_CF"]["auroc"]:.4f} |')
lines+=['', '每条均核验一次视觉编码、一次语言模型前向，无回答生成。耗时不含模型加载及离线参考构建，未与sampling方案做同机时延对比。当前结果不证明ICCV新颖性，也不意味着质量映射本身提供了超出对比值的分类增益。旧工作results未修改。','', '![ROC](roc.png)']
(dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'overall':overall,'difference95ci':report['paired_image_auc_difference95ci'],'by_conflict_type':{k:{'images':v['images'],'CF0':v['block0_CF']['auroc'],'CF24':v['block24_CF']['auroc']} for k,v in subgroups.items()},'timing':timings},indent=2))
