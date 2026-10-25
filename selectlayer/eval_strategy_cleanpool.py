"""Select layer on STA-only-dev vs STA+clean dev; evaluate on a deployment-like
pool where clean-correct are negatives (a fooling detector must not fire on benign).

Label everywhere: output != 'a'  (GT is 'a' for STA, RTA and clean alike).
Held-out eval pool = leftover STA + ALL clean (never used for selection).
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

for devN in [60,120,200]:
    print(f"\n===== dev STA size = {devN}; clean pool sizes: qwen {data['qwen']['yc'].sum()}/{500} wrong ... =====")
    for m,dd in data.items():
        B,X,y,Xc,yc=dd["B"],dd["X"],dd["y"],dd["Xc"],dd["yc"]
        out={s:[] for s in ["L0","argA-STAdev","argA-STAcleanDev"]}
        for seed in range(10):
            rng=np.random.RandomState(seed)
            idx=rng.permutation(len(y)); dev,test=idx[:devN],idx[devN:]
            yd=y[dev]
            if len(set(yd.tolist()))<2: continue
            Xd,Xt=X[:,dev],X[:,test]; yt=y[test]
            Xdevclean=Xc; ydevclean=yc          # all clean as extra dev negatives
            # eval pool: held-out STA + all clean
            Xe=np.concatenate([Xt,Xc],1); ye=np.concatenate([yt,yc])
            def eval_layer(l,sg): return auroc(ye,sg*Xe[l])
            # ---- L0 fixed: sign calibrated on dev STA (needs direction)
            da0=auroc(yd,Xd[0]); sg0= 1.0 if (da0==da0 and da0>=0.5) else -1.0
            out["L0"].append(eval_layer(0,sg0))
            # ---- select argmax|AUROC-.5| on STA dev only
            da=np.array([auroc(yd,Xd[l]) for l in range(B)])
            ddv=np.nan_to_num(da,nan=0.5); l=int(np.nanargmax(np.abs(ddv-0.5)))
            sg= 1.0 if da[l]>=0.5 else -1.0
            out["argA-STAdev"].append(eval_layer(l,sg))
            # ---- select on STA dev + clean dev (clean as negatives)
            yA=np.concatenate([yd,ydevclean]); XA=np.concatenate([Xd,Xdevclean],1)
            daA=np.array([auroc(yA,XA[l]) for l in range(B)])
            dA=np.nan_to_num(daA,nan=0.5); lA=int(np.nanargmax(np.abs(dA-0.5)))
            sgA= 1.0 if daA[lA]>=0.5 else -1.0
            out["argA-STAcleanDev"].append(eval_layer(lA,sgA))
        line="  ".join(f"{k}: {np.nanmean(v):.3f}" for k,v in out.items())
        print(f"{m:<9} deploy-poolAUROC | {line}")
