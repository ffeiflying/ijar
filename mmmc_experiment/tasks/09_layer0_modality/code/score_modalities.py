"""Layer0 cross-modal CF evaluation. Fixed raw image/question and latent polarity frame."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import SPLITS,TASK5,EXPERIMENT
sys.path.insert(0,str(TASK5/'code'))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from evidence_v2 import EvidenceModel,cross_source_conflict
TASK=EXPERIMENT/'tasks/09_layer0_modality'
p=argparse.ArgumentParser();p.add_argument('--splits',default='dev,validation');a=p.parse_args()
sources={s:json.loads((SPLITS/f'{s}.json').read_text()) for s in a.splits.split(',')}
weight=torch.load(EXPERIMENT/'tasks/08_feature_positions/results/weights/L0_attn_wo.pt',map_location='cuda')
# Shared zero-mean gauge preserves additive visual/text partition; no sample-specific offset.
evidence=EvidenceModel(weight)
records={s:[{'row_id':r['row_id'],'image_id':r['image_id'],'label':r['label'],'scores':{}} for r in rows] for s,rows in sources.items()}
reconstruction={}
for split,rows in sources.items():
 archives=[np.load(TASK/f'results/features/{split}/{r["row_id"]}.npz') for r in rows]
 reconstruction[split]={'rows':len(rows),'max_abs_error':float(max(f['reconstruction_max_abs'] for f in archives)),'question_exact_match_fraction':float(np.mean([f['question_exact'] for f in archives]))}
 for query in ['prediction','question_end','question_mean']:
  for normalized in ['', '_normalized']:
   positive=[];negative=[]
   for source in ['visual','text_context']:
    ps=[];ns=[]
    for f in archives:
     x=f[f'{query}_{source}{normalized}'].astype(np.float32)
     ap,am=evidence.get_evidence_totals(torch.as_tensor(x,device='cuda'))
     ps.append(ap);ns.append(am)
    positive.append(torch.stack(ps));negative.append(torch.stack(ns))
   for temperature in [.25,1.,4.]:
    st=cross_source_conflict(positive[0],negative[0],positive[1],negative[1],temperature)
    b=st['cf_logscore'];count=max(1,b.shape[-1]//10)
    stats={'mean_cf':st['cf'].mean(-1),'mean_logcf':b.mean(-1),'top10_logcf':b.topk(count,dim=-1).values.mean(-1),'max_logcf':b.max(-1).values,'ig':st['log_ignorance'].exp().mean(-1),'negative_log_ig':-st['log_ignorance'].mean(-1)}
    for key,value in stats.items():
     values=value.cpu().numpy();assert np.isfinite(values).all()
     name=f'{query}{normalized}_T{temperature:g}_{key}'
     for r,v in zip(records[split],values):r['scores'][name]=float(v)
   del positive,negative,ps,ns,st
   print('DONE',split,query,normalized or 'raw',flush=True)
 for f in archives:f.close()
 (TASK/f'results/{split}_scores.json').write_text(json.dumps(records[split]))
 metrics={k:{'auroc':float(roc_auc_score([r['label'] for r in records[split]],[r['scores'][k] for r in records[split]])),'auprc':float(average_precision_score([r['label'] for r in records[split]],[r['scores'][k] for r in records[split]]))} for k in records[split][0]['scores']}
 (TASK/f'results/{split}_metrics.json').write_text(json.dumps(metrics,indent=2))
(TASK/'results/reconstruction.json').write_text(json.dumps(reconstruction,indent=2))
print('DONE all',flush=True)
