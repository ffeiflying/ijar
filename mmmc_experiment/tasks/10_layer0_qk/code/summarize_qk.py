import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import SPLITS,EXPERIMENT
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
TASK=EXPERIMENT/'tasks/10_layer0_qk'
p=argparse.ArgumentParser();p.add_argument('--splits',default='dev,validation');a=p.parse_args()
reports={};allrows={}
for split in a.splits.split(','):
 source=json.loads((SPLITS/f'{split}.json').read_text());rows=[];errors=[]
 for r in source:
  with np.load(TASK/f'results/features/{split}/{r["row_id"]}.npz') as f:
   scores={k:float(f[k]) for k in f.files if '_T' in k};assert all(np.isfinite(v) for v in scores.values())
   errors.append(float(f['plausibility_reconstruction_error']))
  rows.append({'row_id':r['row_id'],'image_id':r['image_id'],'label':r['label'],'scores':scores})
 y=[r['label'] for r in rows]
 metrics={k:{'auroc':float(roc_auc_score(y,[r['scores'][k] for r in rows])),'auprc':float(average_precision_score(y,[r['scores'][k] for r in rows]))} for k in rows[0]['scores']}
 reports[split]=metrics;allrows[split]=rows
 (TASK/f'results/{split}_scores.json').write_text(json.dumps(rows));(TASK/f'results/{split}_metrics.json').write_text(json.dumps(metrics,indent=2))
 print(split,'rows',len(rows),'max reconstruction error',max(errors))
if 'dev' in reports and 'validation' in reports:
 cfkey=max((k for k in reports['dev'] if 'cf' in k),key=lambda k:reports['dev'][k]['auroc'])
 igkey=max((k for k in reports['dev'] if k.endswith('_ig')),key=lambda k:reports['dev'][k]['auroc'])
 selected={'layer':0,'CF':{'key':cfkey,'dev':reports['dev'][cfkey],'validation':reports['validation'][cfkey]},'IG':{'key':igkey,'dev':reports['dev'][igkey],'validation':reports['validation'][igkey]},'selection':'development only, higher score predicts conflict; no direction flipping'}
 # Image-pair bootstrap for selected CF. Pair order is normal/conflict as fixed split.
 scores=np.array([r['scores'][cfkey] for r in allrows['validation']]).reshape(-1,2);y=np.tile([0,1],len(scores));rng=np.random.default_rng(42)
 bs=[roc_auc_score(y,scores[rng.integers(0,len(scores),len(scores))].ravel()) for _ in range(1000)]
 selected['CF']['validation_auroc95ci']=np.quantile(bs,[.025,.975]).tolist()
 (TASK/'results/dev_selected_validation.json').write_text(json.dumps(selected,indent=2));print(json.dumps(selected,indent=2))
