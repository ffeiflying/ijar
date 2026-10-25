"""User protocol: select layer on 50 clean + 50 STA by SOURCE (no answers);
evaluate AUROC on the BIG pool(s) with label = ANSWER CORRECT (output=='a').
Canonical selection = seed42 dev draw; then report answer-AUROC per pool.
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
data={}
for m,B in MODELS.items():
    sta=json.load(open(f"{BASE}/STA_layerauroc_{m}.json"))
    Xs=np.array([[r["per_layer_conflict"][l] for r in sta] for l in range(B)]); ws=wrong(sta)
    rta=json.load(open(f"{BASE}/RTA_layerauroc_{m}.json"))
    Xr=np.array([[r["per_layer_conflict"][l] for r in rta] for l in range(B)]); wr=wrong(rta)
    cln=json.load(open(f"{ROOT}/selectlayer/clean_layerauroc_{m}.json"))
    Xc=np.array([[r["per_layer_conflict"][l] for r in cln] for l in range(B)]); wc=wrong(cln)
    data[m]=dict(B=B,Xs=Xs,ws=ws,Xr=Xr,wr=wr,Xc=Xc,wc=wc)
    print(f"{m}: STA fool={ws.mean():.3f}  RTA fool={wr.mean():.3f}  clean wrong={wc.mean():.3f}")
def escore(a,b):
    mu=a.mean(1); ms=b.mean(1); nu=a.shape[1]; ns=b.shape[1]
    po=np.sqrt(((nu-1)*a.var(1)+(ns-1)*b.var(1))/(nu+ns-2)+1e-12)
    return np.abs(mu-ms)/po, np.sign(mu-ms)
print("\nCanonical source-selected layer (dev 50 STA + 50 clean, seed42), answer-based AUROC on each big pool:")
hdr=f"{'model':<9}{'l* sg':>8}"+"".join(f"{p:>16}" for p in ["STAleftover","RTA-500","clean-left","STA+clean","RTA+clean"])
print(hdr)
for m,dd in data.items():
    B=dd["B"]
    rng=np.random.RandomState(42); i=rng.permutation(500); sd,st=i[:50],i[50:]
    j=rng.permutation(500); cd,ct=j[:50],j[50:]
    es,sg=escore(dd["Xs"][:,sd],dd["Xc"][:,cd]); l=int(np.argmax(es)); sgn=sg[l]
    L=(lambda X: sgn*X[l])
    aS=auroc(dd["ws"][st],L(dd["Xs"][:,st]))                       # STA leftover vs fooled
    aR=auroc(dd["wr"],L(dd["Xr"]))                                 # RTA full vs fooled
    aC=auroc(dd["wc"][ct],L(dd["Xc"][:,ct]))                       # clean leftover vs wrong
    aSC=auroc(np.concatenate([dd["ws"][st],dd["wc"][ct]]),L(np.concatenate([dd["Xs"][:,st],dd["Xc"][:,ct]],1)))
    aRC=auroc(np.concatenate([dd["wr"],dd["wc"][ct]]),L(np.concatenate([dd["Xr"],dd["Xc"][:,ct]],1)))
    print(f"{m:<9}{str(l)+('+' if sgn>0 else '-'):>8}"+ "".join(f"{x:>16.3f}" for x in [aS,aR,aC,aSC,aRC]))
