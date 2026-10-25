"""DoLa-style value-based premature-layer pilot, on saved per-token-per-layer ACF.

DoLa (Chuang et al., ICLR 2024) picks, per token, the "premature" layer whose
next-token distribution is most different (max JSD) from the final layer, then
contrasts mature-minus-premature.  We do NOT have per-layer vocab distributions
in the saved layerwise JSONs -- we only have per-token x per-layer ACF
(per_token_conflict[t][l]).  So this pilot transposes DoLa's selection rule to
the ACF conflict profile:

  * for a sample s, each layer l defines a distribution over its own generated
    tokens:  P_{s,l} = softmax_t( ACF_{s,l}(t) )   (where conflict is concentrated)
  * "mature" reference N = last layer; candidate l in a DoLa-like bucket set
  * premature layer M_s = argmax_l JSD(P_{s,l} || P_{s,N})   -- label-free, per sample

Questions answered (pilot, n=100 STA per model):
  Q1  Does M_s land in different layers for correct vs incorrect samples?
  Q2  Does a detector using the dynamically chosen layer beat fixing layer 0?
  Q3  Does a DoLa-like contrast score (z at M minus z at N) separate the two?

Caveats (written into output): no clean negatives in layerwise JSONs; n=100;
scores z-scored over the same pool (unsupervised, mild; true honest eval on the
500 STA + 500 clean pool requires re-extraction with clean negatives).
"""
import json, os
import numpy as np
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "layerwise_acf")
FILES = {
    "qwen":     "STA_layerwise_acf.json",           # no-model-suffix file is qwen
    "internvl": "STA_layerwise_acf_internvl.json",
    "mof":      "STA_layerwise_acf_mof.json",
    "deepseek": "STA_layerwise_acf_deepseek.json",
}
LAYERS = {"qwen": 28, "internvl": 32, "mof": 40, "deepseek": 12}


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return float('nan')
    o = np.argsort(s, kind='mergesort'); r = np.empty_like(o, float); r[o] = np.arange(len(s)) + 1
    P = int(y.sum()); N = int((1 - y).sum())
    if P == 0 or N == 0:
        return float('nan')
    a = (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)
    return a if a >= 0.5 else 1 - a   # sign-agnostic AUROC (flip if inverted)


def jsd(p, q):
    """Jensen-Shannon divergence of two discrete distributions (base-2)."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    eps = 1e-12
    kl = lambda a, b: np.sum(a * np.log2((a + eps) / (b + eps)))
    return 0.5 * (kl(p, m) + kl(q, m))


def load(model, fname, B):
    d = json.load(open(os.path.join(ROOT, fname)))
    recs = []
    for r in d:
        ptc = r["per_token_conflict"]          # [T][l]
        # token dimension may include image/answer tokens; keep only layers, mean over tokens
        C = np.array(ptc, dtype=float)          # [T, l]
        if C.ndim != 2 or C.shape[1] != B:
            # some files may be [T][l] others [l][T]? guard
            C = C.T if C.shape[0] == B else C
            if C.ndim != 2 or C.shape[1] != B:
                print(f"   !! shape {C.shape} unexpected for {fname}"); continue
        mean_by_layer = C.mean(axis=0)          # [B]  scalar ACF per layer (mean over tokens)
        # profile per layer = softmax over tokens of that layer's conflict
        # skip all-constant/zero profiles
        prof = np.exp(C - C.max(axis=0, keepdims=True))
        prof = prof / prof.sum(axis=0, keepdims=True)   # [T, B]
        out = str(r["model_output"]).strip().lower()
        recs.append(dict(
            y=int(out[:1] != "a"),              # unsafe = model did NOT answer 'a'
            mean=mean_by_layer,
            prof=prof,
            ntok=C.shape[0],
        ))
    return recs


def run(model, B, early_only):
    recs = load(model, FILES[model], B)
    if not recs:
        return
    y = np.array([r["y"] for r in recs])
    n_corr, n_wrong = int((1 - y).sum()), int(y.sum())
    N = B - 1  # mature reference layer

    M_all, jsd_vals, meanM, meanN = [], [], [], []
    for r in recs:
        P_N = r["prof"][:, N]
        # candidate layers: DoLa picks among its buckets; we use all non-final,
        # plus an early-half-restricted variant handled by 'early_only'.
        cand = range(0, B - 1) if not early_only else range(0, max(1, int(round(0.5 * B))))
        js = np.array([jsd(r["prof"][:, l], P_N) for l in cand])
        l = int(np.argmax(js))
        M_all.append(l); jsd_vals.append(js.max())
        meanM.append(r["mean"][l]); meanN.append(r["mean"][N])
    M_all = np.array(M_all); meanM = np.array(meanM); meanN = np.array(meanN)
    jsd_vals = np.array(jsd_vals)

    # unsupervised z-score over the pool (pilot only)
    def z(a):
        return (a - a.mean()) / (a.std() + 1e-12)
    zM = z(meanM); zN = z(meanN)
    contrast = zM - zN

    tag = "early-half candidates" if early_only else "all non-final candidates"
    print(f"  [{model:9s} B={B:2d}] n={len(recs)} (corr={n_corr}, wrong={n_wrong})  {tag}")
    print(f"     Q1 chosen-layer (DoLa argmax JSD vs L{B-1}):")
    for lbl, mask in (("correct ", y == 0), ("incorrect", y == 1)):
        if mask.sum():
            c = Counter(M_all[mask].tolist())
            top = dict(sorted(c.items(), key=lambda kv: -kv[1])[:5])
            print(f"        {lbl}: mean chosen {M_all[mask].mean():5.1f}  top {top}")
    print(f"     Q1b JSD value: corr {jsd_vals[y==0].mean():.4f}  wrong {jsd_vals[y==1].mean():.4f}")
    au_l0 = auroc(y, z(np.array([rr['mean'][0] for rr in recs])))
    au_dyn = auroc(y, zM)
    au_contra = auroc(y, contrast)
    print(f"     Q2/Q3 AUROC (unsupervised scores on n=100, flipped-if-inverted):")
    print(f"        L0-fixed  = {au_l0:.3f}    DoLa-dynamic M_s = {au_dyn:.3f}    contrast(zM-zN) = {au_contra:.3f}")
    return dict(model=model, n=len(recs), n_corr=n_corr, n_wrong=n_wrong,
                auroc_l0=au_l0, auroc_dyn=au_dyn, auroc_contrast=au_contra)


if __name__ == "__main__":
    for early in (False, True):
        print(f"\n========== DoLa-style premature-layer pilot, STA-layerwise (100/model) ==========\n"
              f"variant: {'EARLY-HALF candidate bucket' if early else 'ALL non-final candidate bucket'}")
        for m, B in LAYERS.items():
            try:
                run(m, B, early_only=early)
            except Exception as e:
                print(f"  [{m}] ERROR {type(e).__name__}: {e}")
