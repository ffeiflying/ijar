"""Compare feature locations with shared evidence_v2; no classifier-head signals."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import SPLITS,TASK5,EXPERIMENT
sys.path.insert(0,str(TASK5/'code'))
import torch,numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from evidence_v2 import EvidenceModel,combine_evidence,binary_channel_evidence
TASK=EXPERIMENT/'tasks/08_feature_positions';OLD=EXPERIMENT/'tasks/07_attention_acf'
p=argparse.ArgumentParser();p.add_argument('--splits',default='dev,validation');p.add_argument('--layers',default='0');args=p.parse_args()
sources={s:json.loads((SPLITS/f'{s}.json').read_text()) for s in args.splits.split(',')}
results={s:[{'row_id':r['row_id'],'image_id':r['image_id'],'label':r['label'],'scores':{}} for r in rows] for s,rows in sources.items()}
for split,rows in results.items():
 saved=TASK/f'results/{split}_scores.json'
 if saved.exists():
  previous=json.loads(saved.read_text())
  assert [r['row_id'] for r in previous]==[r['row_id'] for r in rows]
  results[split]=previous
for layer in map(int,args.layers.split(',')):
 for site in ['attn_wo','ffn_w1','ffn_w3','ffn_w2']:
  key=f'L{layer}_{site}'
  if all(all(f'{key}_dev_mean_T4_ig' in r['scores'] for r in rs) for rs in results.values()):
   print('RESUME skip',key,flush=True);continue
  featurekey=f'L{layer}' if site=='ffn_w2' else key
  parent=OLD if site=='ffn_w2' else TASK
  xs={s:np.array([np.load(parent/f'results/features/{s}/{r["row_id"]}.npz')[featurekey] for r in rows],dtype=np.float32) for s,rows in sources.items()}
  wp=Path(f'/data/liuxiaofei/uncertainty/InternVL/weight/internvl_MLP_Layer{layer}_down_proj.pth') if site=='ffn_w2' else TASK/f'results/weights/{key}.pt'
  weight=torch.load(wp,map_location='cuda')
  for calibration,mu in [('zero',None),('dev_mean',xs['dev'].mean(0,dtype=np.float64))]:
   evidence=EvidenceModel(weight,feature_mean=mu)
   for split,X in xs.items():
    plus=[];minus=[]
    for x in X:
     positive,negative=evidence.get_evidence_totals(torch.as_tensor(x,device='cuda'))
     plus.append(positive);minus.append(negative)
    a=torch.stack(plus);d=torch.stack(minus)
    for temp in [.25,1.,4.]:
     multiclass=combine_evidence(a,d,temp);binary=binary_channel_evidence(a,d,temp)
     b=binary['cf_logscore'];count=max(1,b.shape[-1]//10)
     stats={'global':multiclass['cf_logscore'],'channel_mean':b.mean(-1),'channel_top10':b.topk(count,dim=-1).values.mean(-1),'channel_max':b.max(-1).values,'negative_log_ig':-binary['log_ignorance'].mean(-1),'ig':binary['log_ignorance'].exp().mean(-1)}
     for name,v in stats.items():
      values=v.cpu().numpy();assert np.isfinite(values).all()
      skey=f'{key}_{calibration}_T{temp:g}_{name}'
      for i,r in enumerate(results[split]):r['scores'][skey]=float(values[i])
    del a,d,plus,minus
   del evidence
  del weight
  print('DONE',key,flush=True)
  for split,rows in results.items():(TASK/f'results/{split}_scores.json').write_text(json.dumps(rows))
for split,rows in results.items():
 y=[r['label'] for r in rows];metrics={k:{'auroc':float(roc_auc_score(y,[r['scores'][k] for r in rows])),'auprc':float(average_precision_score(y,[r['scores'][k] for r in rows]))} for k in rows[0]['scores']}
 (TASK/f'results/{split}_metrics.json').write_text(json.dumps(metrics,indent=2))
print('DONE all',flush=True)
