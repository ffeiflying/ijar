"""Control for the surviving novelty: is it the CONFLICT (kappa) component that
detects typographic-attack errors, or would ANY Dempster-Shafer quantity at the
same layer do?  The canonical json already stores both `per_layer_conflict` and
`per_layer_ignorance` per sample, so this needs no GPU.

For each model at its locked l* we report, on the held-out 900 pool:
  pooled AUROC / attack-only / clean-only, for conflict vs ignorance,
  using FOLDED AUROC (max(a,1-a)) so neither term is forced to the other's sign,
  plus a paired bootstrap on (conflict - ignorance) within the attack condition.

Usage: python analyze_acf_vs_ignorance.py
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = "/data/liuxiaofei/liuxiaofei/IJAR"
LROOT = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "results", "e4")
SEED, NBOOT = 42, 2000

# model -> (n_layers, locked l*, protocol sign)
MODELS = {"qwen": (28, 27, +1), "internvl": (32, 22, +1),
          "mof": (40, 32, -1), "deepseek": (12, 6, +1)}


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty_like(o, float)
    r[o] = np.arange(len(s)) + 1
    P = int(y.sum())
    N = int(len(y) - P)
    if P == 0 or N == 0:
        return float("nan")
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


def folded(y, s):
    a = auroc(y, s)
    return a if (np.isnan(a) or a >= 0.5) else 1 - a


def wrong(rows):
    return np.array([int(str(r["model_output"]).strip().lower()[:1] != "a") for r in rows])


def split_idx(n, seed=SEED):
    rng = np.random.RandomState(seed)
    i = rng.permutation(n)
    return i[:50], i[50:]


def paired(y, s1, s2, n=NBOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        a, b = folded(y[idx], np.asarray(s1)[idx]), folded(y[idx], np.asarray(s2)[idx])
        if not (np.isnan(a) or np.isnan(b)):
            d.append(a - b)
    if not d:
        return float("nan"), (float("nan"), float("nan"))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), (float(lo), float(hi))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for m, (B, lstar, sgn) in MODELS.items():
        s = json.load(open(f"{ROOT}/results/layerauroc/STA_layerauroc_{m}.json"))
        _, st = split_idx(len(s))                     # held-out STA (attack) 450
        As = np.array([r["per_layer_conflict"][lstar] for r in s])[st]
        Is = np.array([r["per_layer_ignorance"][lstar] for r in s])[st]
        ys = wrong(s)[st]

        for name, X in (("conflict", As), ("ignorance", Is)):
            rows.append(dict(model=m, l_star=lstar, score=name, attack_only=folded(ys, X)))
        dm, (lo, hi) = paired(ys, As, Is)
        rows[-2].update(att_d_vs_ignorance=dm, d_lo=lo, d_hi=hi)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/acf_vs_ignorance.csv", index=False)
    piv = df.pivot(index="model", columns="score", values="attack_only")
    print("=== folded AUROC within ATTACK (held-out STA 450): conflict vs ignorance ===")
    print(piv.round(4).to_string())
    print("\n=== paired (conflict - ignorance) within ATTACK, 95% CI ===")
    for m in MODELS:
        r = df[(df.model == m) & (df.score == "conflict")]
        if len(r) and not np.isnan(r.att_d_vs_ignorance.iloc[0]):
            r = r.iloc[0]
            print(f"  {m:<9} d={r.att_d_vs_ignorance:+.4f} "
                  f"[{r.d_lo:+.4f},{r.d_hi:+.4f}]")
    print(f"\n[saved] {OUT}/acf_vs_ignorance.csv")


if __name__ == "__main__":
    main()
