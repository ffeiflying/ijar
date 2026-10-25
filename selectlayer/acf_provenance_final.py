"""FINAL provenance-target protocol. Detection = is input attacked (STA vs clean).
Selection (dev 50+50, ACF only, NO answer):
  score(l)=|mean_ACF(STA)-mean_ACF(clean)|/pooled_std ;  l*=argmax ; sign by mean dir.
Reported AUROC (label=STA) AFTER fixing layer on held-out ~900.
"""
import json, numpy as np
from collections import Counter
ROOT="/data/liuxiaofei/liuxiaofei/IJAR/results"; BASE=ROOT+"/layerauroc"
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
    c=json.load(open(f"{ROOT}/selectlayer/clean_layerauroc_{m}.json"))
    Xc=np.array([[r["per_layer_conflict"][l] for r in c] for l in range(B)])
    data[m]=dict(B=B,X=X,Xc=Xc)
def escore(Xs,Xc):
    mu=Xs.mean(1); ms=Xc.mean(1); nU=Xs.shape[1]; nS=Xc.shape[1]
    po=np.sqrt(((nU-1)*Xs.var(1)+(nS-1)*Xc.var(1))/(nU+nS-2)+1e-12)
    return np.abs(mu-ms)/po, np.sign(mu-ms)
print("== canonical (seed42) ==")
canon={}
for m,dd in data.items():
    B,X,Xc=dd["B"],dd["X"],dd["Xc"]
    rng=np.random.RandomState(42); ista=rng.permutation(500); icl=rng.permutation(500)
    sd,st=ista[:50],ista[50:]; cd,ct=icl[:50],icl[50:]
    es,sg=escore(X[:,sd],Xc[:,cd]); l=int(np.argmax(es))
    yt=np.concatenate([np.ones(len(st)),np.zeros(len(ct))]); Xt=np.concatenate([X[:,st],Xc[:,ct]],1)
    a=auroc(yt,sg[l]*Xt[l]); canon[m]=(l,sg[l],a)
    print(f"  {m:<9} l*={l:<2} sign={'+'if sg[l]>0 else '-'}  dev|Δ|/s pooled={es[l]:.1f}  held-out AUROC={a:.4f}")
print("\n== robustness: 80 seeds, selection+AUROC; fixed-L0 reference (target=provenance) ==")
for m,dd in data.items():
    B,X,Xc=dd["B"],dd["X"],dd["Xc"]
    hs=[];ls=[];sgc=[]
    for seed in range(80):
        rng=np.random.RandomState(seed); ista=rng.permutation(500); icl=rng.permutation(500)
        sd,st=ista[:50],ista[50:]; cd,ct=icl[:50],icl[50:]
        es,sg=escore(X[:,sd],Xc[:,cd]); l=int(np.argmax(es))
        yt=np.concatenate([np.ones(len(st)),np.zeros(len(ct))]); Xt=np.concatenate([X[:,st],Xc[:,ct]],1)
        hs.append(auroc(yt,sg[l]*Xt[l])); ls.append(l); sgc.append(sg[l])
    # fixed L0 reference: calibrate sign on dev pool (seed 0..79), eval
    ref=[]
    for seed in range(80):
        rng=np.random.RandomState(seed); ista=rng.permutation(500); icl=rng.permutation(500)
        sd,st=ista[:50],ista[50:]; cd,ct=icl[:50],icl[50:]
        s0=np.sign(X[:,sd].mean(1)[0]-Xc[:,cd].mean(1)[0]) if False else 1.0
        # L0 sign from pooled dev
        pool_s=np.concatenate([X[0,sd],Xc[0,cd]]); grp=np.array([1]*50+[0]*50)
        o=np.argsort(pool_s); rn=np.empty_like(o,float); rn[o]=np.arange(100)+1.0
        sgn=1.0 if (rn[grp==1].sum()-50*51/2)/2500>=0.5 else -1.0
        yt=np.concatenate([np.ones(len(st)),np.zeros(len(ct))]); Xt=np.concatenate([X[:,st],Xc[:,ct]],1)
        ref.append(auroc(yt,sgn*Xt[0]))
    print(f"  {m:<9} pick: {np.mean(hs):.3f}±{np.std(hs):.3f}  layer#{dict(Counter(ls).most_common(2))}  sign#{dict(Counter(sgc).most_common(2))}   |  fixed L0: {np.mean(ref):.3f}±{np.std(ref):.3f}")
print("\n== effect-size curve (canonical seed42), peaks & plateaus ==")
for m,dd in data.items():
    B,X,Xc=dd["B"],dd["X"],dd["Xc"]
    rng=np.random.RandomState(42); ista=rng.permutation(500); icl=rng.permutation(500)
    es,sg=escore(X[:,ista[:50]],Xc[:,icl[:50]])
    order=np.argsort(-es)
    print(f"  {m:<9} top-4 layers {order[:4].tolist()} es {np.round(es[order[:4]],2).tolist()} | argmax es={es.max():.2f} | es@L0={es[0]:.2f}")
