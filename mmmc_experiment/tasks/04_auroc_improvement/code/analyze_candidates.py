"""Fit candidates on dev only; evaluate predefined methods on disjoint validation/final."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT=TASK4 / "results"
p=argparse.ArgumentParser();p.add_argument('--split',default='dev',choices=['dev','validation','final']);a=p.parse_args()
MODES=['plain','verify','grounded','reverse','verify4','blind']
def load(split):
 data={}
 for line in (ROOT/f'{split}_candidates.jsonl').read_text().splitlines():
  r=json.loads(line);data.setdefault(r['row_id'],{})[r['mode']]=r
 source=json.loads((SPLITS/f'{split}.json').read_text())
 assert set(data)=={r['row_id'] for r in source},'Missing records'
 assert all(set(v)==set(MODES) for v in data.values()),'Missing modes'
 y=np.array([r['label'] for r in source]);features={}
 for key in data[source[0]['row_id']]['plain']['features']:
  features[key]=np.array([data[r['row_id']]['plain']['features'][key] for r in source])
 for m in MODES:
  for k in ['decision_margin','entropy','negative_max_prob']:
   features[f'{m}_{k}']=np.array([data[r['row_id']][m][k] for r in source])
 assert all(np.isfinite(v).all() for v in features.values())
 return source,y,features
source_d,yd,fd=load('dev');source,y,f=load(a.split)
score={'verify':f['verify_decision_margin'],'grounded':f['grounded_decision_margin'],'reverse':f['reverse_decision_margin'],'verify4':f['verify4_decision_margin'],'blind_control':f['blind_decision_margin'],'visual_delta':f['verify_decision_margin']-f['blind_decision_margin'],'baseline_L0_first':f['cf_L0'],'baseline_L31_first':f['cf_L31']}
selection={}
for prefix,name in [('cf_L','selected_original_cf'),('elementwise_cf_L','selected_elementwise_cf'),('activation_norm_L','selected_activation_norm')]:
 choices=[(roc_auc_score(yd,sign*v),k,sign) for k,v in fd.items() if k.startswith(prefix) for sign in [1,-1]]
 auc,key,sign=max(choices)
 selection[name]={'feature':key,'sign_selected_on_dev':sign,'dev_auroc':auc}
 score[name]=sign*f[key]
keys=[f'{m}_decision_margin' for m in ['verify','grounded','reverse']]
score['prompt_ensemble']=sum((f[k]-fd[k].mean())/max(fd[k].std(),1e-8) for k in keys)/len(keys)
# Frozen small supervised heads; no LVLM weight training. Fixed regularization, fit dev only.
for name,keys in [('cf_logistic',[k for k in fd if k.startswith('cf_L')]),('elementwise_logistic',[k for k in fd if k.startswith('elementwise_cf_L')]),('fusion_logistic',[k for k in fd if k.startswith(('cf_L','elementwise_cf_L'))]+[f'{m}_decision_margin' for m in ['verify','grounded','reverse','verify4']])]:
 X=np.array([fd[k] for k in keys]).T;Xt=np.array([f[k] for k in keys]).T
 scaler=StandardScaler().fit(X)
 model=LogisticRegression(C=.1,max_iter=1000,random_state=42).fit(scaler.transform(X),yd)
 score[name]=model.decision_function(scaler.transform(Xt))
 selection[name]={'features':keys,'C':.1,'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist()}
if a.split=='dev':
 (ROOT/'dev_frozen_candidates.json').write_text(json.dumps(selection,indent=2))
else:
 assert json.loads((ROOT/'dev_frozen_candidates.json').read_text())==selection,'Dev method definition changed'
report={k:{'auroc':float(roc_auc_score(y,v)),'auprc':float(average_precision_score(y,v))} for k,v in score.items()}
(ROOT/f'{a.split}_metrics.json').write_text(json.dumps(report,indent=2))
(ROOT/f'{a.split}_scores.json').write_text(json.dumps([{'row_id':r['row_id'],'image_id':r['image_id'],'label':r['label'],'scores':{k:float(v[i]) for k,v in score.items()}} for i,r in enumerate(source)]))
print(json.dumps(dict(sorted(report.items(),key=lambda kv:-kv[1]['auroc'])),indent=2))
