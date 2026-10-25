import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import SPLITS,TASK5,EXPERIMENT
sys.path.insert(0,str(TASK5/'code'))
import torch,numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from evidence_v2 import EvidenceModel,combine_evidence
R=EXPERIMENT/'tasks/07_attention_acf/results';config=json.loads((R/'frozen_acf_selection.json').read_text());rows=json.loads((SPLITS/'acf_test.json').read_text())
weight=torch.load(f'/data/liuxiaofei/uncertainty/InternVL/weight/internvl_MLP_Layer{config["layer"]}_down_proj.pth',map_location='cuda');m=EvidenceModel(weight)
scores=[];raw=[]
for row in rows:
 phi=np.load(R/f'features/acf_test/{row["row_id"]}.npz')[f'L{config["layer"]}'].astype(np.float32)
 m.get_evidence_weights(torch.as_tensor(phi,device='cuda'))
 st=combine_evidence(m.w_pos1[0],m.w_neg1[0],config['temperature']);scores.append(st['cf_logscore'].item());raw.append(st['cf'].item())
y=[r['label'] for r in rows];pairs=np.array(scores).reshape(-1,2);rng=np.random.default_rng(42);bs=[]
for _ in range(1000):bs.append(roc_auc_score(np.tile([0,1],len(pairs)),pairs[rng.integers(0,len(pairs),len(pairs))].ravel()))
report={'config':config,'questions':len(rows),'image_pairs':len(pairs),'auroc':float(roc_auc_score(y,scores)),'auprc':float(average_precision_score(y,scores)),'auroc95ci':np.quantile(bs,[.025,.975]).tolist(),'rawCF_unique':len(set(raw)),'logscore_unique':len(set(scores)),'score':'monotone -log(1-CF), not logits','image_disjoint_from_dev_validation_and_previous_prompt_test':True}
(R/'acf_test_frozen_metrics.json').write_text(json.dumps(report,indent=2));(R/'acf_test_frozen_scores.json').write_text(json.dumps([{**r,'cf_logscore':s,'rawCF':v} for r,s,v in zip(rows,scores,raw)]));print(json.dumps(report,indent=2))
