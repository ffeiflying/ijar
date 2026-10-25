"""Core ACF layer/calibration/scale sweep. No LM scores or rewritten questions."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import SPLITS,TASK5,EXPERIMENT
sys.path.insert(0,str(TASK5/'code'))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from evidence_v2 import EvidenceModel,combine_evidence
TASK=EXPERIMENT/'tasks/07_attention_acf'
p=argparse.ArgumentParser();p.add_argument('--layers',default=','.join(map(str,range(32))));args=p.parse_args()
temps=[.25,.5,1.,2.,4.,8.]
sources={s:json.loads((SPLITS/f'{s}.json').read_text()) for s in ['dev','validation']}
results={s:[{'row_id':r['row_id'],'image_id':r['image_id'],'label':r['label'],'scores':{}} for r in rows] for s,rows in sources.items()}
for layer in map(int,args.layers.split(',')):
    key=f'L{layer}';features={}
    for split,rows in sources.items():
        features[split]=np.array([np.load(TASK/f'results/features/{split}/{r["row_id"]}.npz')[key] for r in rows],dtype=np.float32)
    mu=features['dev'].mean(0,dtype=np.float64)
    weight=torch.load(f'/data/liuxiaofei/uncertainty/InternVL/weight/internvl_MLP_Layer{layer}_down_proj.pth',map_location='cuda')
    for calibration,mean in [('zero',None),('dev_mean',mu)]:
        model=EvidenceModel(weight,feature_mean=mean)
        for split,xs in features.items():
            plus=[];minus=[]
            for x in xs:
                model.get_evidence_weights(torch.as_tensor(x,device='cuda'))
                plus.append(model.w_pos1[0].clone());minus.append(model.w_neg1[0].clone())
            a=torch.stack(plus);d=torch.stack(minus)
            for temp in temps:
                statistics=combine_evidence(a,d,temperature=temp)
                scores=statistics['cf_logscore'].cpu().numpy();raw=statistics['cf'].cpu().numpy()
                for i,r in enumerate(results[split]):
                    r['scores'][f'L{layer}_{calibration}_T{temp:g}']=float(scores[i])
                    if temp==1:r['scores'][f'rawCF_L{layer}_{calibration}']=float(raw[i])
            del a,d,plus,minus
        del model
    del weight
    print('DONE layer',layer,flush=True)
    for split,rows in results.items():
        (TASK/f'results/{split}_acf_scan_scores.json').write_text(json.dumps(rows))
for split,rows in results.items():
    y=[r['label'] for r in rows]
    metrics={k:{'auroc':float(roc_auc_score(y,[r['scores'][k] for r in rows])),'auprc':float(average_precision_score(y,[r['scores'][k] for r in rows]))} for k in rows[0]['scores']}
    (TASK/f'results/{split}_acf_scan_metrics.json').write_text(json.dumps(metrics,indent=2))
print('DONE all',flush=True)
