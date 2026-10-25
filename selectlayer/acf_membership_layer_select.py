"""Layer selection by ACF gap between CLEAN and ATTACK samples (membership only).

User's intended protocol:
  * At layer-selection time we do NOT know the correct answer.
  * We only know, for each dev image, whether it is CLEAN or ATTACKED.
  * Select, per layer l, the standardized ACF gap between the two membership groups:
        score(l) = |mean_ACF(attack_dev,l) - mean_ACF(clean_dev,l)| / pooled_std(l)
        l* = argmax_l score(l);  sign = +1 if mean_attack >= mean_clean else -1
  * Correctness is NEVER used in the selector.

Evaluation (after the layer is FIXED, on disjoint held-out = leftover STA + leftover clean):
  metric A  AUROC vs membership        y = is_attacked
  metric B  AUROC vs output-wrongness  y = (answer != 'a')      <- paper headline metric

STA json = attacked pool; clean json = benign pool (membership is known from the file).
"""
import json, os
import numpy as np
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
AUROC = os.path.join(ROOT, "layerauroc")
CLN = os.path.join(ROOT, "selectlayer")
FILES = {"qwen": "STA_layerauroc_qwen.json", "internvl": "STA_layerauroc_internvl.json",
         "mof": "STA_layerauroc_mof.json", "deepseek": "STA_layerauroc_deepseek.json"}
CLNF = {"qwen": "clean_layerauroc_qwen.json", "internvl": "clean_layerauroc_internvl.json",
        "mof": "clean_layerauroc_mof.json", "deepseek": "clean_layerauroc_deepseek.json"}


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return float('nan')
    o = np.argsort(s, kind='mergesort'); r = np.empty_like(o, float); r[o] = np.arange(len(s)) + 1
    P = int(y.sum()); N = int((1 - y).sum())
    if P == 0 or N == 0:
        return float('nan')
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def pick_dev(Xa, Xc):
    """Xa [B, n_attack_dev], Xc [B, n_clean_dev]. Returns (l*, sign). Membership only."""
    ma = Xa.mean(1); mc = Xc.mean(1)
    na, nc = Xa.shape[1], Xc.shape[1]
    if na < 2 or nc < 2:
        return None
    va = Xa.var(1); vc = Xc.var(1)
    po = np.sqrt(((na - 1) * va + (nc - 1) * vc) / (na + nc - 2) + 1e-12)
    score = np.abs(ma - mc) / po
    l = int(np.nanargmax(score))
    return l, (1.0 if ma[l] >= mc[l] else -1.0)


def load(BASE, fname, B):
    d = json.load(open(os.path.join(BASE, fname)))
    X = np.array([[r["per_layer_conflict"][l] for r in d] for l in range(B)])   # [B, n]
    yw = np.array([str(r["model_output"]).strip().lower()[:1] != "a" for r in d], int)  # output-wrongness
    return X, yw


N_DEV, N_SEEDS = 50, 60
print("Selector: membership groups (attack vs clean), score(l)=|Δmean|/pooled_std, argmax, sign from mean dir.")
print("Eval (held-out, disjoint): metricA=AUROC vs attacked-membership,  metricB=AUROC vs output-wrongness")
print(f"{'model':<9}{'l*':>4}{'sign':>6}{'B(memb)':>9}{'B(wrong)':>9}   chosen-layer dist / sign dist (60 seeds)")
stats = {}
for m in FILES:
    # infer B
    d0 = json.load(open(os.path.join(AUROC, FILES[m])))[0]
    B = len(d0["per_layer_conflict"])
    Xa, ywa = load(AUROC, FILES[m], B)
    Xc, ywc = load(CLN, CLNF[m], B)
    n = Xa.shape[1]
    A_mem, A_w, Ls, Sgs = [], [], [], []
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        ia = rng.permutation(n); ic = rng.permutation(n)
        ad, at = ia[:N_DEV], ia[N_DEV:]; cd, ct = ic[:N_DEV], ic[N_DEV:]
        pk = pick_dev(Xa[:, ad], Xc[:, cd])
        if pk is None:
            continue
        l, sg = pk
        Xt = np.concatenate([Xa[:, at], Xc[:, ct]], 1)                 # [B, test]
        yt_mem = np.concatenate([np.ones(len(at), int), np.zeros(len(ct), int)])
        yt_w = np.concatenate([ywa[at], ywc[ct]])
        A_mem.append(auroc(yt_mem, sg * Xt[l]))
        A_w.append(auroc(yt_w, sg * Xt[l]))
        Ls.append(l); Sgs.append(sg)
    stats[m] = (Ls, Sgs)
    cd = Counter(Ls).most_common(3)
    cs = Counter(Sgs).most_common(2)
    print(f"{m:<9}{np.mean(Ls):>4.1f}{('+' if np.mean(Sgs) >= 0 else '-'):>6}"
          f"{np.mean(A_mem):>9.3f}{np.mean(A_w):>9.3f}   layers {dict(cd)}  sign {dict(cs)}")

print("\nCANONICAL lock (seed=42):")
for m in FILES:
    d0 = json.load(open(os.path.join(AUROC, FILES[m])))[0]
    B = len(d0["per_layer_conflict"])
    Xa, ywa = load(AUROC, FILES[m], B); Xc, ywc = load(CLN, CLNF[m], B)
    n = Xa.shape[1]
    rng = np.random.RandomState(42)
    ia = rng.permutation(n); ic = rng.permutation(n)
    ad, at = ia[:N_DEV], ia[N_DEV:]; cd, ct = ic[:N_DEV], ic[N_DEV:]
    l, sg = pick_dev(Xa[:, ad], Xc[:, cd])
    Xt = np.concatenate([Xa[:, at], Xc[:, ct]], 1)
    yt_mem = np.concatenate([np.ones(len(at), int), np.zeros(len(ct), int)])
    yt_w = np.concatenate([ywa[at], ywc[ct]])
    print(f"  {m:<9} layer={l:<3} sign={'+' if sg > 0 else '-'}  "
          f"held-out AUROC memb={auroc(yt_mem, sg * Xt[l]):.4f}  wrong={auroc(yt_w, sg * Xt[l]):.4f}")
