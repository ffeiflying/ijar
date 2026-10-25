"""Independent exhaustive DS fusion verifies reference, not merely same formulas."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from denoeux_reference import evidence,summarize

def combine(m,n):
    raw={}
    for a,u in m.items():
        for b,v in n.items():raw[a&b]=raw.get(a&b,0)+u*v
    k=raw.pop(0,0)
    return {a:v/(1-k) for a,v in raw.items()},k

def fused(w,positive):
    K=w.shape[1];theta=(1<<K)-1;m={theta:1.}
    totals=np.maximum(w if positive else -w,0).sum(0)
    for k,t in enumerate(totals):
        focal=(1<<k) if positive else theta^(1<<k)
        m,_=combine(m,{focal:-np.expm1(-t),theta:np.exp(-t)})
    return m
rng=np.random.default_rng(123);maxerr=0
for K in [2,3,4]:
    for _ in range(20):
        W=rng.normal(0,.2,(K,5));x=rng.normal(size=5);mu=rng.normal(0,.2,5);b=rng.normal(0,.2,K)
        w=evidence(W,x,b,mu);r=summarize(w)
        logits=W@x+b;expected=np.exp(logits-logits.max());expected/=expected.sum()
        assert np.allclose(r['normalized_plausibility'],expected,atol=1e-12)
        m,k=combine(fused(w,True),fused(w,False))
        assert np.isclose(k,r['conflict'],atol=1e-12)
        assert np.isclose(m.get((1<<K)-1,0),r['ignorance'],atol=1e-12)
        maxerr=max(maxerr,abs(k-r['conflict']))
# Complete ignorance must be a valid neutral case, no nonzero-evidence requirement.
r=summarize(np.zeros((5,3)));assert r['conflict']==0 and r['ignorance']==1
# Reproduce existing matrix-product+expand construction: all class columns identical.
W=np.array([[1.,-1.,.5],[-.3,.2,.8]]);x=np.array([1.,2.,-.5])
old=np.broadcast_to(((W-W.mean(1,keepdims=True))@x[:,None])/W.shape[0],W.shape)
oldp=summarize(old)['normalized_plausibility']
assert np.allclose(oldp,np.ones(W.shape[1])/W.shape[1])
report={'random_small_classifiers_checked':60,'softmax_reconstruction_passed':True,'conflict_matches_independent_exhaustive_fusion':True,'ignorance_matches_exhaustive_fusion':True,'zero_evidence_is_vacuous':True,'max_conflict_error':maxerr,'old_expand_gives_uniform_class_plausibility':oldp.tolist()}
(TASK5/'results/verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
