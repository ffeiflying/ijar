"""FINAL LOCKED protocol.
SELECT (dev = 50 STA + 50 clean, ACF only, answers NEVER used):
  score(l) = |mean_ACF(STA@l)-mean_ACF(clean@l)| / pooled_std(ACF@l)
  l* = argmax_l score(l); sign = sign(mean_ACF(STA) - mean_ACF(clean))
EVALUATE (big held-out pool = STA leftover + clean leftover, ~900):
  AUROC of sign*ACF(l*) vs label = ANSWER CORRECT  (output=='a'; wrong=positive)
Pool choices fixed by user: eval pool STA+clean; qwen limits reported as-is.
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
    s=json.load(open(f"{BASE}/STA_layerauroc_{m}.json")); c=json.load(open(f"{ROOT}/selectlayer/clean_layerauroc_{m}.json"))
    D[m]=dict(B=B,
      Xs=np.array([[r["per_layer_conflict"][l] for r in s] for l in range(B)]),ws=wrong(s),
      Xc=np.array([[r["per_layer_conflict"][l] for r in c] for l in range(B)]),wc=wrong(c))
def escore(a,b):
    mu=a.mean(1); ms=b.mean(1); nu=a.shape[1]; ns=b.shape[1]
    po=np.sqrt(((nu-1)*a.var(1)+(ns-1)*b.var(1))/(nu+ns-2)+1e-12)
    return np.abs(mu-ms)/po, np.sign(mu-ms)
print("CANONICAL lock (seed42 dev draw): selected by SOURCE on 50STA+50clean")
canon={}
for m,dd in D.items():
    B=dd["B"]; rng=np.random.RandomState(42); i=rng.permutation(500); sd,st=i[:50],i[50:]
    j=rng.permutation(500); cd,ct=j[:50],j[50:]
    es,sg=escore(dd["Xs"][:,sd],dd["Xc"][:,cd]); l=int(np.argmax(es)); sgn=sg[l]
    X=np.concatenate([dd["Xs"][:,st],dd["Xc"][:,ct]],1); y=np.concatenate([dd["ws"][st],dd["wc"][ct]])
    a=auroc(y,sgn*X[l]); canon[m]=(l,sgn,a)
    n_pos=int(y.sum())
    print(f"  {m:<9} l*={l:<2} sign={'+'if sgn>0 else'-'}  es={es[l]:.1f}  answer-AUROC(STA+clean,{len(y)})= {a:.4f}  (pos={n_pos})")
print("\nROBUSTNESS 60 seeds (re-select on each dev draw, same eval pool):")
for m,dd in D.items():
    B=dd["B"]; hs=[];ls=[]
    for seed in range(60):
        rng=np.random.RandomState(seed); i=rng.permutation(500); sd,st=i[:50],i[50:]
        j=rng.permutation(500); cd,ct=j[:50],j[50:]
        es,sg=escore(dd["Xs"][:,sd],dd["Xc"][:,cd]); l=int(np.argmax(es)); sgn=sg[l]
        X=np.concatenate([dd["Xs"][:,st],dd["Xc"][:,ct]],1); y=np.concatenate([dd["ws"][st],dd["wc"][ct]])
        hs.append(auroc(y,sgn*X[l])); ls.append(l)
    from collections import Counter
    print(f"  {m:<9} {np.mean(hs):.3f}±{np.std(hs):.3f}   layer~{dict(Counter(ls).most_common(2))}")
print("\n(wrong-rate on full pools: qwen STA .58 / internvl .52 / mof .96 / deepseek .82 —"
      " explains answer-AUROC ceiling for low-hit models)")
