"""Independent small-frame checks for new vectorized evidence and BPA conflict."""
import itertools
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'05_cf_correction/code'))
from evidence_v2 import combine_evidence, fuse_evidence_sources

torch.manual_seed(42)
max_total=max_plaus=max_bpa=max_head=0.
for _ in range(30):
    B=torch.randn(4,7,dtype=torch.float64)
    B-=B.mean(0,keepdim=True)
    h=torch.randn(3,7,dtype=torch.float64)
    w=h[:,None,:]*B[None,:,:]
    s=h@B.T
    absolute=h.abs()@B.abs().T
    a=(absolute+s)/2
    d=(absolute-s)/2
    max_total=max(max_total,(a-w.relu().sum(-1)).abs().max().item(),(d-(-w).relu().sum(-1)).abs().max().item())
    stat=combine_evidence(a,d)
    max_plaus=max(max_plaus,(stat['normalized_plausibility']-s.softmax(-1)).abs().max().item())
    p=torch.rand(5,4,dtype=torch.float64)
    p/=p.sum(-1,keepdim=True)
    r=torch.rand(5,dtype=torch.float64)
    masses=r[:,None]*p
    explicit=[]
    for i,j in itertools.combinations(range(5),2):
        value=sum((masses[i,k]*masses[j,l]).item() for k in range(4) for l in range(4) if k!=l)
        formula=(r[i]*r[j]-(masses[i]*masses[j]).sum()).item()
        max_bpa=max(max_bpa,abs(value-formula))
        explicit.append(value)
    total=(r.sum().square()-r.square().sum())-(masses.sum(0).square().sum()-masses.square().sum())
    max_head=max(max_head,abs((total/20).item()-sum(explicit)/len(explicit)))
    fused=fuse_evidence_sources(a,d,source_dim=0)
    assert 0<=fused['cf'].item()<=1
assert max(max_total,max_plaus,max_bpa,max_head)<1e-12
result={'cases':30,'matrix_identity_error':max_total,'plausibility_error':max_plaus,'singleton_BPA_conflict_error':max_bpa,'head_pair_average_error':max_head,'scope':'numerical identities only; does not validate semantic hypotheses or source independence'}
path=Path(__file__).resolve().parent.parent/'results/math_verification.json'
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
