"""Clean-split honest comparison. Selection dev and evaluation test are disjoint:
  dev  = random 150 STA + random 250 clean   (labels known, GT='a')
  test = remaining 350 STA + remaining 250 clean
Label = output != 'a'. Candidate strategies:
  L0-fixed        (layer 0; sign from dev AUROC)
  argA-STAdev     argmax|AUROC-0.5| over layers on STA-only dev (clean ignored)
  argA-allDev     argmax|AUROC-0.5| on STA+clean dev
Report AUROC on the disjoint deployment-like test pool. 10 seeds.
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

STA_DEV, CLEAN_DEV = 150, 250
print(f"dev = {STA_DEV} STA + {CLEAN_DEV} clean; test = rest (disjoint); metric = AUROC on test pool")
print(f"{'model':<9}{'L0':>9}{'argA-STAdev':>13}{'argA-allDev':>13}{'n_test':>8}")
rows={}
for m,dd in data.items():
    B,X,y,Xc,yc=dd["B"],dd["X"],dd["y"],dd["Xc"],dd["yc"]
    acc={"L0":[],"argA-STAdev":[],"argA-allDev":[]}
    for seed in range(10):
        rng=np.random.RandomState(seed)
        ista=rng.permutation(len(y)); icl=rng.permutation(len(yc))
        sd,st=ista[:STA_DEV],ista[STA_DEV:]; cd,ct=icl[:CLEAN_DEV],icl[CLEAN_DEV:]
        # disjoint test pool
        yt=np.concatenate([y[st],yc[ct]]); Xt=np.concatenate([X[:,st],Xc[:,ct]],1)
        if len(set(yt.tolist()))<2: continue
        # --- L0
        da0=auroc(y[sd],X[0,sd]); sg0= 1.0 if da0>=0.5 else -1.0
        acc["L0"].append(auroc(yt,sg0*Xt[0]))
        # --- select on STA dev only
        da=np.array([auroc(y[sd],X[l,sd]) for l in range(B)])
        if np.all(np.isnan(da)): continue
        ddn=np.nan_to_num(da,nan=0.5); l=int(np.nanargmax(np.abs(ddn-0.5)))
        sg= 1.0 if da[l]>=0.5 else -1.0
        acc["argA-STAdev"].append(auroc(yt,sg*Xt[l]))
        # --- select on STA+clean dev
        yA=np.concatenate([y[sd],yc[cd]]); XA=np.concatenate([X[:,sd],Xc[:,cd]],1)
        daA=np.array([auroc(yA,XA[l]) for l in range(B)])
        dAn=np.nan_to_num(daA,nan=0.5); lA=int(np.nanargmax(np.abs(dAn-0.5)))
        sgA= 1.0 if daA[lA]>=0.5 else -1.0
        acc["argA-allDev"].append(auroc(yt,sgA*Xt[lA]))
    out={k:(f"{np.mean(v):.3f}±{np.std(v):.3f}" if v else "  nan") for k,v in acc.items()}
    rows[m]=acc
    print(f"{m:<9}"+ "".join(f"{out[k]:>13}" for k in ["L0","argA-STAdev","argA-allDev"])+f"{len(yt):>8}")
# average test AUROC (mean across models of the mean over seeds)
print("\nmean over models:")
for k in ["L0","argA-STAdev","argA-allDev"]:
    avg=np.mean([np.mean(rows[m][k]) for m in MODELS])
    print(f"  {k:<14} {avg:.3f}")
