"""Corrected DS features on fixed disjoint splits. No test-direction selection."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from denoeux_reference import evidence,summarize
from evidence_v2 import EvidenceModel
ROOT=TASK5 / "results";IMPROVE=SPLITS

dev=np.load(ROOT/'dev_head.npz');assert len(dev['row_id'])==200
weightAB=np.load(ROOT/'ab_weight.npy');muAB=dev['phi'].mean(0,dtype=np.float64)
report={};allrecords={}
for split,expected in [('dev',200),('validation',400)]:
 data=np.load(ROOT/f'{split}_head.npz');assert len(data['row_id'])==expected
 src={r['row_id']:r for r in json.loads((IMPROVE/f'{split}.json').read_text())}
 assert set(map(int,data['row_id']))==set(src)
 records=[{'row_id':int(i),'image_id':src[int(i)]['image_id'],'label':src[int(i)]['label'],'scores':{}} for i in data['row_id']]
 for phi,r in zip(data['phi'],records):
  w=evidence(weightAB,phi,mu=muAB);s=summarize(w)
  a=s['w_plus'];d=s['w_minus'];lexp=a+np.log(-np.expm1(-a));ldp=np.logaddexp.reduce(np.r_[0.,lexp]);ldm=np.log(-np.expm1(np.log(-np.expm1(-d)).sum()));lz=np.logaddexp.reduce(np.r_[ldm,lexp-d])
  r['scores']['AB_corrected_cf']=float(s['conflict']);r['scores']['AB_corrected_cf_logscore']=float(ldp+ldm-lz)
  r['scores']['AB_logit_margin']=float((weightAB[0].astype(float)-weightAB[1])@phi)
  expectedlogits=weightAB.astype(float)@phi;expectedp=np.exp(expectedlogits-np.logaddexp.reduce(expectedlogits))
  assert np.allclose(s['normalized_plausibility'],expectedp,atol=1e-10)
 for layer in [0,31]:
  weight=torch.load(f'/data/liuxiaofei/uncertainty/InternVL/weight/internvl_MLP_Layer{layer}_down_proj.pth',map_location='cuda').float()
  mu=dev[f'phi{layer}'].mean(0,dtype=np.float64)
  evidence_model=EvidenceModel(weight,feature_mean=mu)
  for phi,r in zip(data[f'phi{layer}'],records):
   evidence_model.get_evidence_weights(torch.as_tensor(phi,device='cuda'))
   s={'cf':evidence_model.get_evidence_conflict().item(),'cf_logscore':evidence_model.get_conflict_logscore().item(),'log_ignorance':evidence_model.log_ignorance.item()}
   assert all(np.isfinite(v) for v in s.values())
   for key,v in s.items():r['scores'][f'raw_L{layer}_{key}']=v
  del weight,evidence_model
 keys=records[0]['scores'];ys=[r['label'] for r in records]
 report[split]={k:{'auroc':float(roc_auc_score(ys,[r['scores'][k] for r in records])),'auprc':float(average_precision_score(ys,[r['scores'][k] for r in records])),'unique_scores':len(set(r['scores'][k] for r in records))} for k in keys}
 allrecords[split]=records
 (ROOT/f'{split}_corrected_scores.json').write_text(json.dumps(records))
 print(split,json.dumps(report[split],indent=2),flush=True)
(ROOT/'corrected_metrics.json').write_text(json.dumps(report,indent=2))
