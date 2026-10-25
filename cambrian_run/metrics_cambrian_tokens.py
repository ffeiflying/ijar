"""Cambrian-8B: compare ACF statistics (all-token mean vs decision token).

Cambrian-8B generates ~100-token captions, so the per-sample ACF the first run
stored -- a mean of kappa over every generated token -- averages the conflict
signal away relative to the four letter-emitting models, which only generate
1-3 tokens. This script puts the three statistics side by side:

  per_layer_conflict        mean kappa over ALL generated tokens
  per_layer_conflict_first  kappa at generated token 0 (the decision token)
  per_layer_conflict_first5 mean kappa over the first 5 generated tokens

Two labels, both derived offline from the options embedded in the question
(Cambrian never emits a letter, so the paper's letter-matching label is
constant on clean):

  names_b    the response names option (b) -> the injected text landed
  not_names_a the response fails to name option (a) -> closest analogue of
             the paper's "wrong answer"

For each (statistic, label) pair: per-layer AUROC, the full-data argmax of the
selection statistic g(b), and the locked protocol (b* chosen on a 100-sample
dev set using ACF values only, evaluated on the held-out 900).

Usage: python metrics_cambrian_tokens.py
"""

import json
import re

import numpy as np

ROOT = "/data/liuxiaofei/liuxiaofei/IJAR/results/cambrian"
SEED = 42
N_DEV = 50
STATS = ["per_layer_conflict", "per_layer_conflict_first",
         "per_layer_conflict_first5"]


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty_like(o, float)
    r[o] = np.arange(len(s)) + 1
    P, N = int(y.sum()), int((1 - y).sum())
    if P == 0 or N == 0:
        return float("nan")
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def parse_options(q):
    q = q.replace("(such as a or b or c or d)", "")
    return dict(re.findall(r"\(([abcd])\)\s*([^()]+?)(?=\s*\([abcd]\)|$)", q))


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def labels(records):
    na, nb = [], []
    for r in records:
        opts = parse_options(r["question"])
        out = " " + norm(r["model_output"]) + " "
        a, b = norm(opts.get("a", "")), norm(opts.get("b", ""))
        na.append(bool(a) and f" {a} " in out)
        nb.append(bool(b) and f" {b} " in out)
    return np.array(na, int), np.array(nb, int)


def gstat(xa, xc):
    """g(b) = |mean_a - mean_c| / pooled_sd, and its sign."""
    mu, ms = xa.mean(1), xc.mean(1)
    nu, ns = xa.shape[1], xc.shape[1]
    pooled = np.sqrt(((nu - 1) * xa.var(1) + (ns - 1) * xc.var(1)) / (nu + ns - 2) + 1e-12)
    return np.abs(mu - ms) / pooled, np.sign(mu - ms)


def main():
    sta = json.load(open(f"{ROOT}/STA_layerauroc_cambrian.json"))
    clean = json.load(open(f"{ROOT}/clean_layerauroc_cambrian.json"))
    B = len(sta[0]["per_layer_conflict"])

    na_s, nb_s = labels(sta)
    na_c, nb_c = labels(clean)
    print(f"Cambrian-8B  STA N={len(sta)} clean N={len(clean)} B={B}")
    print(f"  pos rate  clean: names_b={nb_c.mean():.1%}  not_names_a={(1-na_c).mean():.1%}")
    print(f"  pos rate  STA  : names_b={nb_s.mean():.1%}  not_names_a={(1-na_s).mean():.1%}")

    rows = {}
    for stat in STATS:
        if stat not in sta[0]:
            print(f"\n!! '{stat}' not present -- the token-level run has not "
                  f"finished yet; re-run this script after it completes.")
            return
        Xs = np.array([r[stat] for r in sta]).T
        Xc = np.array([r[stat] for r in clean]).T
        for lname, ys, yc in (("names_b", nb_s, nb_c),
                              ("not_names_a", 1 - na_s, 1 - na_c)):
            y = np.concatenate([ys, yc])
            X = np.concatenate([Xs, Xc], 1)
            auc = np.array([auroc(y, X[b]) for b in range(B)])
            best = int(np.nanargmax(np.abs(np.nan_to_num(auc) - 0.5)))
            g, sg = gstat(Xs, Xc)
            b_full = int(np.argmax(g))

            rng = np.random.RandomState(SEED)
            ps, pc = rng.permutation(len(ys)), rng.permutation(len(yc))
            sd, st, cd, ct = ps[:N_DEV], ps[N_DEV:], pc[:N_DEV], pc[N_DEV:]
            gd, sgd = gstat(Xs[:, sd], Xc[:, cd])
            b_star = int(np.argmax(gd))
            y_ev = np.concatenate([ys[st], yc[ct]])
            X_ev = np.concatenate([Xs[:, st], Xc[:, ct]], 1)
            auc_locked = auroc(y_ev, sgd[b_star] * X_ev[b_star])

            rows[(stat, lname)] = (best, auc[best], b_full, g[b_full],
                                   b_star, gd[b_star], auc_locked)

    print("\n" + "=" * 84)
    print("最右列是定稿协议的 held-out AUROC（选层不看标签，只在 dev 100 样本上用 ACF）")
    print("=" * 84)
    print(f"  {'statistic':26s} {'label':12s} {'bestL':>6} {'bestAUC':>8} "
          f"{'b*(full)':>9} {'g':>6} {'b*(dev)':>8} {'g_dev':>6} {'HELD-OUT':>9}")
    for (stat, lname), (best, bauc, bf, gf, bs, gs, auc) in rows.items():
        print(f"  {stat.replace('per_layer_conflict','plc'):26s} {lname:12s} "
              f"{best:>6} {bauc:>8.4f} {bf:>9} {gf:>6.3f} {bs:>8} {gs:>6.3f} "
              f"{auc:>9.4f}")

    print("\n参考（前 4 个模型，定稿协议 held-out AUROC）: "
          "qwen 0.8225 | internvl 0.9585 | mof 0.9966 | deepseek 0.9869")


if __name__ == "__main__":
    main()
