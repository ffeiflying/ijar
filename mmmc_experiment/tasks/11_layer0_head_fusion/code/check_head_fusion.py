import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import TASK5,EXPERIMENT
sys.path.insert(0,str(TASK5/'code'))
import torch,numpy as np
from evidence_v2 import fuse_evidence_sources

def combine(m,n):
 raw={}
 for i,v in m.items():
  for j,w in n.items():raw[i&j]=raw.get(i&j,0)+v*w
 k=raw.pop(0,0);return {i:v/(1-k) for i,v in raw.items()},k
rng=np.random.default_rng(12);worst=0
for _ in range(30):
 K=3;theta=(1<<K)-1;a=rng.uniform(0,.8,(3,K));d=rng.uniform(0,.8,(3,K));sources=[]
 for head in range(3):
  m={theta:1.}
  for k in range(K):
   m,_=combine(m,{1<<k:1-np.exp(-a[head,k]),theta:np.exp(-a[head,k])})
   m,_=combine(m,{theta^(1<<k):1-np.exp(-d[head,k]),theta:np.exp(-d[head,k])})
  sources.append(m)
 m={theta:1.};nonconflict=1.
 for s in sources:m,k=combine(m,s);nonconflict*=1-k
 st=fuse_evidence_sources(torch.tensor(a),torch.tensor(d))
 worst=max(worst,abs(st['cf'].item()-(1-nonconflict)))
 assert abs(st['cf'].item()-(1-nonconflict))<1e-12
 assert abs(st['log_ignorance'].exp().item()-m.get(theta,0))<1e-12
report={'independent_multisource_mass_enumerations':30,'same_frame_conflict_and_ignorance_passed':True,'max_error':worst}
p=EXPERIMENT/'tasks/11_layer0_head_fusion/results/math_verification.json';p.write_text(json.dumps(report,indent=2));print(p.read_text())
