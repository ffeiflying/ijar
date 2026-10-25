"""Honest comparison of layer-selection strategies on offline per-layer ACF data.

Pool per model: STA-500 (all GT='a'); label y = (model_output != 'a')  [fooled].
Clean-500 provides extra y=0 negatives for the extended test pool.

Strategies (selection may use ONLY the 100-sample dev split; AUROC reported on
held-out test):
  S_L0        fixed layer 0
  S_frac{f}   fixed layer = round(f*(B-1))  (swept below)
  S_auroc     per-model: argmax over layers of dev-AUROC  (supervised dev pick)
  S_margin    per-model: argmax over layers of |mean(wrong)-mean(correct)| on dev
  oracle      argmax over layers of TEST AUROC (label-peeking, ceiling only)

Every per-layer score is sign-calibrated on dev: s = +1 if dev AUROC(l)>=0.5.
Test pools: (a) STA-only leftover 400, (b) leftover 400 + all 500 clean.
"""
import json, os
import numpy as np

MODELS = {"qwen": 28, "internvl": 32, "mof": 40, "deepseek": 12}
BASE = "/data/liuxiaofei/liuxiaofei/IJAR/results"

def first(r):
    t = str(r["model_output"]).strip().lower()
    return t[0] if t else ""

def load(name, model):
    return json.load(open(os.path.join(BASE, "layerauroc", f"{name}_layerauroc_{model}.json"))) \
        if name != "clean" else \
        json.load(open(os.path.join(BASE, "selectlayer", f"clean_layerauroc_{model}.json")))

def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(s)) + 1.0
    P = int(y.sum()); N = int((1 - y).sum())
    if P == 0 or N == 0:
        return float("nan")
    return (ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N)

def layer_scores(d):
    L = len(d[0]["per_layer_conflict"])
    return np.array([[r["per_layer_conflict"][l] for r in d] for l in range(L)])  # [L, n]

def sign_from_dev(dev_auc):
    return 1.0 if (dev_auc == dev_auc and dev_auc >= 0.5) else -1.0

def frac_layer(f, B):
    return int(round(f * (B - 1)))

def main():
    SEEDS = list(range(10))
    DEV_N = 100
    fracs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    # gather data
    data = {}
    for m, B in MODELS.items():
        sta = load("STA", m)
        cln = load("clean", m)
        X = layer_scores(sta); Xc = layer_scores(cln)
        y = np.array([first(r) != "a" for r in sta], int)
        yc = np.array([first(r) != "a" for r in cln], int)
        data[m] = dict(B=B, X=X, Xc=Xc, y=y, yc=yc)

    # Results accumulators: strategy -> per-model list of per-seed AUC_STA / AUC_ALL
    strategies = ["L0"] + [f"F{int(f*100):03d}" for f in fracs] + ["AUROC", "MARGIN"]
    acc = {s: {m: [] for m in MODELS} for s in strategies}
    acc_oracle = {m: [] for m in MODELS}
    chosen = {s: {m: [] for m in MODELS} for s in ["AUROC", "MARGIN"]}  # layer chosen per seed

    for m, d in data.items():
        B, X, Xc, y, yc = d["B"], d["X"], d["Xc"], d["y"], d["yc"]
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(y))
            dev_i, test_i = idx[:DEV_N], idx[DEV_N:]
            ydev = y[dev_i]
            if len(set(ydev.tolist())) < 2:
                continue  # dev must contain both classes for sign calibration
            Xdev, Xtest = X[:, dev_i], X[:, test_i]
            ytest = y[test_i]
            ytest_all = np.concatenate([ytest, yc])
            Xtest_all = np.concatenate([Xtest, Xc], axis=1)

            dev_auc = np.array([auroc(ydev, Xdev[l]) for l in range(B)])

            for s in strategies:
                if s == "L0":
                    l = 0
                elif s.startswith("F"):
                    l = frac_layer(int(s[1:]) / 100.0, B)
                elif s == "AUROC":
                    if np.all(np.isnan(dev_auc)):
                        continue
                    dd = np.nan_to_num(dev_auc, nan=0.0)
                    l = int(np.nanargmax(np.abs(dd - 0.5)))
                elif s == "MARGIN":
                    mw = Xdev[:, ydev == 1].mean(axis=1)
                    mc = Xdev[:, ydev == 0].mean(axis=1)
                    l = int(np.argmax(np.abs(mw - mc)))
                if s in chosen:
                    chosen[s][m].append(l)
                sg = sign_from_dev(dev_auc[l])
                a1 = auroc(ytest, sg * Xtest[l])
                a2 = auroc(ytest_all, sg * Xtest_all[l])
                acc[s][m].append((a1, a2))

            # oracle ceiling on this test split: report sign-flipped |AUROC-0.5|+0.5
            def flip_ceiling(y_, X_):
                auc = np.array([auroc(y_, X_[l]) for l in range(B)])
                dd = np.abs(np.nan_to_num(auc, nan=0.5) - 0.5)
                return float(np.nanmax(dd)) + 0.5
            acc_oracle[m].append((flip_ceiling(ytest, Xtest),
                                  flip_ceiling(ytest_all, Xtest_all)))

    def mean_auc(lst):
        return float(np.mean([x for x in lst if x == x])) if lst else float("nan")

    rows = []
    for s in strategies:
        row = {}
        for m in MODELS:
            if len(acc[s][m]) == 0:
                row[m] = (float("nan"), float("nan"))
                continue
            a1 = np.mean([p[0] for p in acc[s][m]])
            a2 = np.mean([p[1] for p in acc[s][m]])
            row[m] = (a1, a2)
        rows.append((s, row))
    # oracle row
    row = {m: (mean_auc([p[0] for p in acc_oracle[m]]), mean_auc([p[1] for p in acc_oracle[m]]))
           for m in MODELS}
    rows.append(("oracle", row))

    def avg2(tag, key):
        vals = [r[1][key][0] for r in rows if r[0] in strategies]
        return float(np.nanmean(vals)) if vals else float("nan")

    print(f"{'strat':<8}" + "".join(f"{m+' (STA|+clean)':>22}" for m in MODELS) + "   avg STA")
    for s, row in rows:
        print(f"{s:<8}", end="")
        allv = []
        for m in MODELS:
            a1, a2 = row[m]
            allv.append(a1)
            print(f"  {a1:.3f} | {a2:.3f}   ", end="")
        print(f"  {np.nanmean(allv):.3f}" if allv else "  nan")

    # per-seed variance (STA pool) for the main contenders
    print("\nper-seed STA AUROC  mean +/- std:")
    for s in ["L0", "AUROC", "MARGIN"]:
        print(f"  {s:<8}", end="")
        for m in MODELS:
            vals = [p[0] for p in acc[s][m]]
            if not vals:
                print("      nan      ", end=""); continue
            print(f"  {np.mean(vals):.3f}+-{np.std(vals):.3f}", end="")
        print()

    # chosen-layer stability across seeds for supervised picks
    print("\nchosen layer per seed (AUROC pick / MARGIN pick):")
    for m in MODELS:
        print(f"  {m:<9}", end="")
        for s in ["AUROC", "MARGIN"]:
            print(f"  {s}:{sorted(chosen[s][m])}", end="")
        print()

    # fixed-fraction sweep table: avg STA across models per fraction
    print("\nfixed-fraction avg (STA pool) and per-model best-to-worst spread:")
    for s, row in rows:
        if not s.startswith("F"):
            continue
        vals = {m: row[m][0] for m in MODELS}
        print(f"  {s}: avg={np.nanmean(list(vals.values())):.3f}  "
              f"per-model={ {m: round(v,3) for m,v in vals.items()} }")

if __name__ == "__main__":
    main()
