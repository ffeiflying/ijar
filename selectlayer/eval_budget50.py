"""Layer-selection under the REAL budget: dev = 50 STA + 50 clean ONLY.
Labels for dev known (GT='a' everywhere). Task: discriminate safe (answer 'a')
vs unsafe (answer != 'a'). Selection: argmax |AUROC - 0.5| on dev pool, sign from dev.
Test: disjoint remaining STA + remaining clean, deployment-like pool.
Report mean+-std over many seeds, and chosen-layer stability.
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

SDEV,CDEV,NSEEDS=50,50,60
print(f"dev budget = {SDEV} STA + {CDEV} clean per seed; test=disjoint rest; label=safe(answer=='a')")
for m,dd in data.items():
    B,X,y,Xc,yc=dd["B"],dd["X"],dd["y"],dd["Xc"],dd["yc"]
    aucs=[]; signs=[]; layers=[]
    for seed in range(NSEEDS):
        rng=np.random.RandomState(seed)
        ista=rng.permutation(len(y)); icl=rng.permutation(len(yc))
        sd,st=ista[:SDEV],ista[SDEV:]; cd,ct=icl[:CDEV],icl[CDEV:]
        yt=np.concatenate([y[st],yc[ct]]); Xt=np.concatenate([X[:,st],Xc[:,ct]],1)
        if len(set(yt.tolist()))<2: continue
        yA=np.concatenate([y[sd],yc[cd]]); XA=np.concatenate([X[:,sd],Xc[:,cd]],1)
        if len(set(yA.tolist()))<2: continue
        da=np.array([auroc(yA,XA[l]) for l in range(B)])
        ddn=np.nan_to_num(da,nan=0.5)
        l=int(np.nanargmax(np.abs(ddn-0.5)))
        sg= 1.0 if da[l]>=0.5 else -1.0
        a=auroc(yt,sg*Xt[l])
        aucs.append(a); signs.append(sg); layers.append(l)
    # chosen-layer histogram across seeds
    from collections import Counter
    cnt=Counter(layers)
    top=cnt.most_common(4)
    sdval=np.std(aucs)
    print(f"{m:<9} testAUROC={np.mean(aucs):.3f}±{sdval:.3f}  "
          f"chosen-layer top={ {l:c for l,c in top} }  sign={Counter(signs).most_common(2)}  n_test={len(yt)}")
