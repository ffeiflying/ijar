"""Source-selected layer (dev 50 STA+50 clean), answer-based AUROC, seed-averaged.
Pools: STA-leftover+clean-leftover; RTA-500+clean-leftover.  Also flipped-sign to
test whether low cells are sign-transfer failures (|sign wrong|) or no signal.
"""
import json, numpy as np
ROOT="/data/liuxiaofei/liuxiaofei/IJAR/results"; BASE=ROOT+"/layerauroc"
MODELS={"qwen":28,"internvl":32,"mof":40,"deepseek":12}
def auroc(y,s):
    y=np.asarray(y); s=np.asarray(s,float)
    if len(set(y.tolist()))<2: return float('nan')
    o=np.argsort(s,kind='mergesort'); r=np.empty_like(o,float); r[o]=np.arange(len(s))+1
    P=int(y.sum()); N=int((1-y).sum())
    if P==0 or N==0: return float('nan')
    return (r[y==1].sum()-P*(P+1)/2)/(P*N)
def wrong(d): return np.array([str(r["model_output"]).strip().lower()[0]!="a" for r in d],int)
D={}
for m,B in MODELS.items():
    s=json.load(open(f"{BASE}/STA_layerauroc_{m}.json")); r=json.load(open(f"{BASE}/RTA_layerauroc_{m}.json"))
    c=json.load(open(f"{ROOT}/selectlayer/clean_layerauroc_{m}.json"))
    D[m]=dict(B=B,
      Xs=np.array([[x["per_layer_conflict"][l] for x in s] for l in range(B)]),ws=wrong(s),
      Xr=np.array([[x["per_layer_conflict"][l] for x in r] for l in range(B)]),wr=wrong(r),
      Xc=np.array([[x["per_layer_conflict"][l] for x in c] for l in range(B)]),wc=wrong(c))
def escore(a,b):
    mu=a.mean(1); ms=b.mean(1); nu=a.shape[1]; ns=b.shape[1]
    po=np.sqrt(((nu-1)*a.var(1)+(ns-1)*b.var(1))/(nu+ns-2)+1e-12)
    return np.abs(mu-ms)/po, np.sign(mu-ms)
print("seed-avg answer-AUROC | source-selected layer (dev STA+clean), sign from dev STA/clean means")
print(f"{'model':<9}{'STA+clean pool':>16}{'(flip)':>8}{'RTA+clean pool':>16}{'(flip)':>8}")
for m,dd in D.items():
    B=dd["B"]; A1=[];A1f=[];A2=[];A2f=[]
    for seed in range(60):
        rng=np.random.RandomState(seed); i=rng.permutation(500); sd,st=i[:50],i[50:]
        j=rng.permutation(500); cd,ct=j[:50],j[50:]
        es,sg=escore(dd["Xs"][:,sd],dd["Xc"][:,cd]); l=int(np.argmax(es)); sgn=sg[l]
        X1=np.concatenate([dd["Xs"][:,st],dd["Xc"][:,ct]],1); y1=np.concatenate([dd["ws"][st],dd["wc"][ct]])
        X2=np.concatenate([dd["Xr"],dd["Xc"][:,ct]],1); y2=np.concatenate([dd["wr"],dd["wc"][ct]])
        A1.append(auroc(y1,sgn*X1[l])); A1f.append(auroc(y1,-sgn*X1[l]))
        A2.append(auroc(y2,sgn*X2[l])); A2f.append(auroc(y2,-sgn*X2[l]))
    g=lambda v: f"{np.nanmean(v):.3f}±{np.nanstd(v):.3f}"
    print(f"{m:<9}{g(A1):>16}{g(A1f):>8}{g(A2):>16}{g(A2f):>8}")
