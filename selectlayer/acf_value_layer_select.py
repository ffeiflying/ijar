"""
Layer selection from ACF values ONLY (no AUROC in the selector).
=================================================================
Dev pool : 50 STA + 50 clean  (the only samples allowed for selection)
Groups   : by ACTUAL correctness  -- safe = output starts with 'a' (= GT),
           unsafe = otherwise. (Clean samples that the model got wrong join unsafe.)
Selector (per layer l): effect size of the ACF gap between the two groups
    score(l) = | mean_ACF(unsafe,l) - mean_ACF(safe,l) | / pooled_std(ACF,l)
    l* = argmax_l score(l);  sign = +1 if mean_unsafe>=mean_safe else -1
AUROC is computed ONLY AFTER the layer is fixed, on held-out samples.
"""
import json, numpy as np
ROOT="/data/liuxiaofei/liuxiaofei/IJAR/results"
BASE=ROOT+"/layerauroc"
MODELS={"qwen":28,"internvl":32,"mof":40,"deepseek":12}
def auroc(y,s):
    y=np.asarray(y); s=np.asarray(s,float)
    if len(set(y.tolist()))<2: return float('nan')
    o=np.argsort(s,kind='mergesort'); r=np.empty_like(o,float); r[o]=np.arange(len(s))+1
    P=int(y.sum()); N=int((1-y).sum())
    if P==0 or N==0: return float('nan')
    return (r[y==1].sum()-P*(P+1)/2)/(P*N)
data={}
for m,B in MODELS.items():
    d=json.load(open(f"{BASE}/STA_layerauroc_{m}.json"))
    X=np.array([[r["per_layer_conflict"][l] for r in d] for l in range(B)])
    y=np.array([str(r["model_output"]).strip().lower()[0]!="a" for r in d],int)
    c=json.load(open(f"{ROOT}/selectlayer/clean_layerauroc_{m}.json"))
    Xc=np.array([[r["per_layer_conflict"][l] for r in c] for l in range(B)])
    yc=np.array([str(r["model_output"]).strip().lower()[0]!="a" for r in c],int)
    data[m]=dict(B=B,X=X,y=y,Xc=Xc,yc=yc)

def pick(Xsta,y_sta, Xcln,y_cln, B):
    """Given labelled dev blocks, return (l*, sign). ACF-value selector."""
    U=np.concatenate([Xsta[:,y_sta==1], Xcln[:,y_cln==1]],1)
    S=np.concatenate([Xsta[:,y_sta==0], Xcln[:,y_cln==0]],1)
    if U.shape[1]<2 or S.shape[1]<2: return None
    mu=U.mean(1); ms=S.mean(1)
    nU=U.shape[1]; nS=S.shape[1]
    po=np.sqrt(((nU-1)*U.var(1)+(nS-1)*S.var(1))/(nU+nS-2)+1e-12)
    score=np.abs(mu-ms)/po
    l=int(np.nanargmax(score))
    return l, (1.0 if mu[l]>=ms[l] else -1.0)

# ---- stability over 60 seeds ----
from collections import Counter
print("Protocol: score(l)=|Δmean|/pooled_std, pick argmax, sign by mean dir; AUROC AFTER on held-out 900.")
print(f"{'model':<9}{'held-out mean±std':>20}{'chosen-layer dist':>34}{'sign':>8}")
stat={}
for m,dd in data.items():
    B,X,y,Xc,yc=dd["B"],dd["X"],dd["y"],dd["Xc"],dd["yc"]
    hs=[]; ls=[]; sgn=[]
    for seed in range(60):
        rng=np.random.RandomState(seed)
        ista=rng.permutation(len(y)); icl=rng.permutation(len(yc))
        sd,st=ista[:50],ista[50:]; cd,ct=icl[:50],icl[50:]
        pk=pick(X[:,sd],y[sd],Xc[:,cd],yc[cd],B)
        if pk is None: continue
        l,sg=pk
        yt=np.concatenate([y[st],yc[ct]]); Xt=np.concatenate([X[:,st],Xc[:,ct]],1)
        hs.append(auroc(yt,sg*Xt[l])); ls.append(l); sgn.append(sg)
    stat[m]=(hs,ls,sgn)
    cd=Counter(ls).most_common(3)
    cs=Counter(sgn).most_common(1)[0]
    print(f"{m:<9}{np.mean(hs):>10.3f}±{np.std(hs):.3f}{str(dict(cd)):>34}{str(cs):>8}")

print("\nCANONICAL lock (seed=42, dev=50 STA + 50 clean):")
for m,dd in data.items():
    B,X,y,Xc,yc=dd["B"],dd["X"],dd["y"],dd["Xc"],dd["yc"]
    rng=np.random.RandomState(42)
    ista=rng.permutation(len(y)); icl=rng.permutation(len(yc))
    sd,st=ista[:50],ista[50:]; cd,ct=icl[:50],icl[50:]
    l,sg=pick(X[:,sd],y[sd],Xc[:,cd],yc[cd],B)
    yt=np.concatenate([y[st],yc[ct]]); Xt=np.concatenate([X[:,st],Xc[:,ct]],1)
    print(f"  {m:<9} layer={l:<3} sign={'+'if sg>0 else '-'}   held-out AUROC = {auroc(yt,sg*Xt[l]):.4f}")
