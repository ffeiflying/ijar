"""Provenance-blind selection: group dev 50 STA vs 50 clean by SOURCE (no answer peek).
Selection stat per layer = |mean ACF(STA)-mean ACF(clean)| / pooled_std.
Report held-out AUROC under two possible target definitions:
  (i)  provenance : positive = STA sample (attack detector)
  (ii) correctness: positive = model answer != 'a'  (safety/correctness detector, project label)
Selection is IDENTICAL for both targets (uses no answer info); only the eval label differs.
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
    wr=np.array([str(r["model_output"]).strip().lower()[0]!="a" for r in d],int)  # fooled by attack
    c=json.load(open(f"{ROOT}/selectlayer/clean_layerauroc_{m}.json"))
    Xc=np.array([[r["per_layer_conflict"][l] for r in c] for l in range(B)])
    wc=np.array([str(r["model_output"]).strip().lower()[0]!="a" for r in c],int)
    data[m]=dict(B=B,X=X,wr=wr,Xc=Xc,wc=wc)

def pick(src):
    """src: tuple (X_sta,X_cln). Pure source grouping, effect size."""
    Xs,Xcl=src
    mu=Xs.mean(1); ms=Xcl.mean(1)
    nU=Xs.shape[1]; nS=Xcl.shape[1]
    po=np.sqrt(((nU-1)*Xs.var(1)+(nS-1)*Xcl.var(1))/(nU+nS-2)+1e-12)
    sc=np.abs(mu-ms)/po
    l=int(np.nanargmax(sc))
    return l,(1.0 if mu[l]>=ms[l] else -1.0)

print("Selection: source grouping (STA vs clean), effect size, argmax.  Held-out AUROC:")
print(f"{'model':<9}{'→target(i)provenance':>20}{'target(ii)correctness':>24}")
for m,dd in data.items():
    B,X,wr,Xc,wc=dd["B"],dd["X"],dd["wr"],dd["Xc"],dd["wc"]
    acc_i=[]; acc_ii=[]; lay=[]; sgn=[]
    for seed in range(80):
        rng=np.random.RandomState(seed)
        ista=rng.permutation(len(wr)); icl=rng.permutation(len(wc))
        sd,st=ista[:50],ista[50:]; cd,ct=icl[:50],icl[50:]
        l,sg=pick((X[:,sd],Xc[:,cd]))
        Xt=np.concatenate([X[:,st],Xc[:,ct]],1)
        yi=np.concatenate([np.ones(len(st)),np.zeros(len(ct))])            # provenance
        yii=np.concatenate([wr[st],wc[ct]])                                  # correctness
        acc_i.append(auroc(yi,sg*Xt[l])); acc_ii.append(auroc(yii,sg*Xt[l]))
        lay.append(l); sgn.append(sg)
    from collections import Counter
    top=Counter(lay).most_common(2)
    s=Counter(sgn).most_common(1)[0]
    print(f"{m:<9}{np.mean(acc_i):>11.3f}±{np.std(acc_i):.3f}{np.mean(acc_ii):>13.3f}±{np.std(acc_ii):.3f}  "
          f"layer~{dict(top)} {s}")

# how aligned are the two targets on the FULL pools (upper bound of a provenance-blind detector)?
print("\nalignment: among STA fooled rate & clean-wrong rate (fraction of each pool, full 500):")
for m,dd in data.items():
    print(f"  {m:<9} STA fooled={dd['wr'].mean():.3f}   clean-wrong={dd['wc'].mean():.3f}   "
          f"frac of all-unsafe that are clean-wrong={dd['wc'].sum()/(dd['wr'].sum()+dd['wc'].sum()):.3f}")
