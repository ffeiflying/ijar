"""Dev-set size sensitivity for the per-model dev-argmax-|AUROC-0.5| picker.
STA pool only; label = output!='a'; sign calibrated on dev; eval on leftover.
Also tests an 'early-quarter' constraint (pick only within first 25% of layers).
"""
import json, numpy as np
BASE="/data/liuxiaofei/liuxiaofei/IJAR/results/layerauroc"
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
    data[m]=(X,y)
for devN in [20,40,60,100,150,200]:
    print(f"\n=== dev size = {devN} ===")
    for m,(X,y) in data.items():
        B=X.shape[0]
        aucs_all=[]; aucs_early=[]
        for seed in range(10):
            rng=np.random.RandomState(seed)
            idx=rng.permutation(len(y)); dev,test=idx[:devN],idx[devN:]
            yd=y[dev]
            if len(set(yd.tolist()))<2: continue
            Xd,Xt=X[:,dev],X[:,test]; yt=y[test]
            da=np.array([auroc(yd,Xd[l]) for l in range(B)])
            dd=np.nan_to_num(da,nan=0.5)
            l=int(np.nanargmax(np.abs(dd-0.5)))
            sg= 1.0 if (da[l]==da[l] and da[l]>=0.5) else -1.0
            aucs_all.append(auroc(yt,sg*Xt[l]))
            # early-quarter constraint
            q=max(1,int(round(0.25*(B-1))))
            sub=dd[:q+1]
            le=int(np.nanargmax(np.abs(sub-0.5)))
            sge= 1.0 if (da[le]==da[le] and da[le]>=0.5) else -1.0
            aucs_early.append(auroc(yt,sge*Xt[le]))
        a=np.nanmean(aucs_all) if aucs_all else float('nan')
        e=np.nanmean(aucs_early) if aucs_early else float('nan')
        print(f"  {m:<9} full-pick={a:.3f}  early-q-pick={e:.3f}")
