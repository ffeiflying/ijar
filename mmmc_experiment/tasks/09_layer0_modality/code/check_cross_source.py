import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import TASK5,EXPERIMENT
sys.path.insert(0,str(TASK5/'code'))
import numpy as np
import torch
from evidence_v2 import cross_source_conflict
rng=np.random.default_rng(123);worst=0
for _ in range(100):
 av,dv,at,dt=rng.uniform(0,5,4)
 def mass(a,d):
  raw={1:(1-np.exp(-a))*np.exp(-d),2:(1-np.exp(-d))*np.exp(-a),3:np.exp(-a-d)}
  z=sum(raw.values());return {k:v/z for k,v in raw.items()}
 V=mass(av,dv);T=mass(at,dt);raw={}
 for i,v in V.items():
  for j,t in T.items():raw[i&j]=raw.get(i&j,0)+v*t
 st=cross_source_conflict(*[torch.tensor(x,dtype=torch.float64) for x in [av,dv,at,dt]])
 assert abs(st['cf'].item()-raw.get(0,0))<1e-12
 assert abs(st['log_ignorance'].exp().item()-raw[3]/(1-raw.get(0,0)))<1e-12
 worst=max(worst,abs(st['cf'].item()-raw.get(0,0)))
z=torch.tensor(0.,dtype=torch.float64);h=torch.tensor(500.,dtype=torch.float64)
st=cross_source_conflict(h,z,z,h);assert torch.isfinite(st['cf_logscore'])
st=cross_source_conflict(z,z,h,z);assert st['cf']==0
report={'independent_binary_frame_enumerations':100,'cf_and_ignorance_match':True,'maximum_error':worst,'strong_opposition_and_vacuous_edges_passed':True}
p=EXPERIMENT/'tasks/09_layer0_modality/results/math_verification.json';p.write_text(json.dumps(report,indent=2));print(p.read_text())
