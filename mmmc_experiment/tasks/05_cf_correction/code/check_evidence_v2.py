import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import TASK5
import numpy as np
import torch
from evidence_v2 import EvidenceModel
from denoeux_reference import evidence,summarize
rng=np.random.default_rng(42);error=0
for K in [2,3,4]:
 for _ in range(20):
  W=rng.normal(0,.2,(K,5));phi=rng.normal(size=5);mu=rng.normal(0,.2,5);bias=rng.normal(0,.2,K)
  obj=EvidenceModel(torch.tensor(W),bias=bias,feature_mean=mu)
  weights=obj.get_evidence_weights(torch.tensor(phi[:,None]))
  refw=evidence(W,phi,bias,mu);ref=summarize(refw)
  assert np.allclose(weights[0].numpy(),refw,atol=1e-12)
  assert np.allclose(obj.get_normalized_plausibility().numpy(),ref['normalized_plausibility'],atol=1e-12)
  assert abs(obj.get_evidence_ignorance().item()-ref['ignorance'])<1e-12
  error=max(error,abs(obj.get_evidence_conflict().item()-ref['conflict']))
obj=EvidenceModel(torch.zeros(3,5,dtype=torch.float64));obj.get_evidence_weights(torch.zeros(5))
assert obj.get_evidence_conflict().item()==0 and obj.get_evidence_ignorance().item()==1
# Strong balanced opposing contributions should be represented stably even near CF1.
obj=EvidenceModel(torch.tensor([[100.,-100.],[-100.,100.]],dtype=torch.float64));obj.get_evidence_weights(torch.ones(2))
assert torch.isfinite(obj.get_conflict_logscore()).all()
report={'version':EvidenceModel.VERSION,'random_cases':60,'feature_class_evidence_matches_reference':True,'conflict_ignorance_plausibility_passed':True,'vacuous_zero_case_passed':True,'strong_evidence_logscore_finite':True,'maximum_CF_error':error}
(TASK5/'results/evidence_v2_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
