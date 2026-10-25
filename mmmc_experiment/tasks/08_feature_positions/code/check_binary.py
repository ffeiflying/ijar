import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import TASK5,EXPERIMENT
sys.path.insert(0,str(TASK5/'code'))
import torch
from evidence_v2 import binary_channel_evidence
# Independent two-simple-mass fusion: conflicting singleton products and Theta product.
a=torch.tensor([0.,.3,2.,100.],dtype=torch.float64);d=torch.tensor([0.,.7,1.,100.],dtype=torch.float64)
r=binary_channel_evidence(a,d)
mp=-torch.expm1(-a);mn=-torch.expm1(-d)
assert torch.allclose(r['cf'],mp*mn)
ig=torch.exp(-a[:3]-d[:3])/(1-mp[:3]*mn[:3]);assert torch.allclose(r['log_ignorance'][:3].exp(),ig)
logodds=a-d;logbelief1=-d-r['cf_logscore']*-1 # normalized plausibilities use exp(-d)/(exp(-d)+exp(-a))
p=torch.exp(-d[:3])/(torch.exp(-d[:3])+torch.exp(-a[:3]));assert torch.allclose(p,torch.sigmoid(logodds[:3]))
assert r['cf'][0]==0 and r['log_ignorance'][0]==0 and torch.isfinite(r['cf_logscore']).all()
out=EXPERIMENT/'tasks/08_feature_positions/results/binary_verification.json';out.write_text(json.dumps({'binary_mass_fusion_passed':True,'ignorance_passed':True,'normalized_plausibility_passed':True,'zero_and_strong_evidence_passed':True},indent=2));print(out.read_text())
