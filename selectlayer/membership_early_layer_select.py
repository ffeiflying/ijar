"""Honest layer selection under the user's intended protocol.

Constraint (user):
  * At SELECTION time we do NOT know the correct answer; we only know which dev
    images are CLEAN and which are ATTACKED (membership).
  * Final TASK is misbehavior detection: use the fixed layer to tell whether the
    model was affected by the typographic attack, i.e. whether it answered
    correctly (answer != 'a'  ==>  misbehavior / fooled).

Method:
  1. A-priori prior from the paper's dilution finding: the conflict signal that
     discriminates fooled from not-fooled is concentrated in the EARLY layers.
     Therefore the layer search is restricted to the first half of the blocks
     (candidate set  C = {0, ..., ceil(0.5*B)-1}).
  2. Within C, on a small dev pool (N_DEV attacked + N_DEV clean, membership
     only, NO correctness), pick the layer whose ACF best separates the two
     membership groups, in standardized units:
        score(l) = |mean_ACF(attack,l) - mean_ACF(clean,l)| / pooled_std(l)
        l* = argmax_{l in C} score(l);   sign = sign(mean_attack - mean_clean)
  3. Fix (l*, sign). Evaluate ONCE, on disjoint held-out (leftover STA + leftover
     clean), with the target label = output wrongness (answer != 'a').

Reports canonical seed and 60-seed stability (AUROC mean+-std, chosen-layer hist).
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
CAP = 0.5           # a-priori: only first half of layers are candidates
N_DEV, N_SEEDS = 50, 60


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return float('nan')
    o = np.argsort(s, kind='mergesort'); r = np.empty_like(o, float); r[o] = np.arange(len(s)) + 1
    P = int(y.sum()); N = int((1 - y).sum())
    if P == 0 or N == 0:
        return float('nan')
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def pick_membership_early(Xa, Xc, B, cap):
    """(l*, sign). Membership-only, candidates restricted to first `cap*B` layers."""
    hi = max(1, int(round(cap * (B - 1))))
    ma = Xa.mean(1); mc = Xc.mean(1)
    na, nc = Xa.shape[1], Xc.shape[1]
    va = Xa.var(1); vc = Xc.var(1)
    po = np.sqrt(((na - 1) * va + (nc - 1) * vc) / (na + nc - 2) + 1e-12)
    score = np.abs(ma - mc) / po
    l = int(np.nanargmax(score[:hi + 1]))
    return l, (1.0 if ma[l] >= mc[l] else -1.0)


def load(BASE, fname):
    d = json.load(open(os.path.join(BASE, fname)))
    B = len(d[0]["per_layer_conflict"])
    X = np.array([[r["per_layer_conflict"][l] for r in d] for l in range(B)])
    yw = np.array([str(r["model_output"]).strip().lower()[:1] != "a" for r in d], int)
    return X, yw, B


print(f"Protocol: dev={N_DEV} attack + {N_DEV} clean (membership ONLY). "
      f"candidates = first {CAP:.0%} layers. target = output wrongness.")
print(f"{'model':<9}{'l*':>5}{'sign':>6}{'wrong-AUROC':>15}{'memb-AUROC':>14}   chosen-layer dist (60 seeds)")
for m in FILES:
    Xa, ywa, B = load(AUROC, FILES[m])
    Xc, ywc, _ = load(CLN, CLNF[m])
    n = Xa.shape[1]
    Aw, Am, Ls, Sgs = [], [], [], []
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        ia = rng.permutation(n); ic = rng.permutation(n)
        ad, at = ia[:N_DEV], ia[N_DEV:]; cd, ct = ic[:N_DEV], ic[N_DEV:]
        l, sg = pick_membership_early(Xa[:, ad], Xc[:, cd], B, CAP)
        Xt = np.concatenate([Xa[:, at], Xc[:, ct]], 1)
        yw = np.concatenate([ywa[at], ywc[ct]])
        ym = np.concatenate([np.ones(len(at), int), np.zeros(len(ct), int)])
        Aw.append(auroc(yw, sg * Xt[l])); Am.append(auroc(ym, sg * Xt[l]))
        Ls.append(l); Sgs.append(sg)
    cd = Counter(Ls).most_common(4)
    cs = Counter(Sgs).most_common(2)
    print(f"{m:<9}{np.mean(Ls):>5.1f}{('+' if np.mean(Sgs) >= 0 else '-'):>6}"
          f"{np.mean(Aw):>13.3f}+-{np.std(Aw):.3f}{np.mean(Am):>12.3f}   {dict(cd)}  sign {dict(cs)}")

print("\nCANONICAL (seed=42):")
for m in FILES:
    Xa, ywa, B = load(AUROC, FILES[m])
    Xc, ywc, _ = load(CLN, CLNF[m])
    n = Xa.shape[1]
    rng = np.random.RandomState(42)
    ia = rng.permutation(n); ic = rng.permutation(n)
    ad, at = ia[:N_DEV], ia[N_DEV:]; cd, ct = ic[:N_DEV], ic[N_DEV:]
    l, sg = pick_membership_early(Xa[:, ad], Xc[:, cd], B, CAP)
    Xt = np.concatenate([Xa[:, at], Xc[:, ct]], 1)
    yw = np.concatenate([ywa[at], ywc[ct]])
    ym = np.concatenate([np.ones(len(at), int), np.zeros(len(ct), int)])
    print(f"  {m:<9} layer={l:<3} sign={'+' if sg > 0 else '-'}  "
          f"wrong-AUROC={auroc(yw, sg * Xt[l]):.4f}   memb-AUROC={auroc(ym, sg * Xt[l]):.4f}")
