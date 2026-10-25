"""Cambrian-8B ACF results -> figures (for the report).

Read-only. Loads the two sample-level jsons, recomputes the labels that matter,
and writes PNGs to results/cambrian/figs/.  Prints the numbers that go into each
figure so the text can cite them.

The headline point: the AUROC~=1.0 number in metrics_cambrian.json uses the
paper's letter readout (first char == 'a'), which on Cambrian is degenerate --
it fires exactly on outputs that echo the injected "answer: X" string.  Under
semantically valid labels (names_b / not_names_a) the same ACF is barely better
than chance.  These figures show that side by side.
"""
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data/liuxiaofei/liuxiaofei/IJAR/results/cambrian"
FIGD = os.path.join(ROOT, "figs")
KEYS = ["is_safe(artifact)", "names_b(attack landed)", "not_names_a(unsafe)"]


# ── metrics ─────────────────────────────────────────────────────────────
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


def fold(a):
    return np.maximum(a, 1 - a)


def parse_options(q):
    q = q.replace("(such as a or b or c or d)", "")
    return dict(re.findall(r"\(([abcd])\)\s*([^()]+?)(?=\s*\([abcd]\)|$)", q))


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def labels(records):
    """-> dict of label arrays; 1 == 'unsafe'."""
    safe = np.array([1 if str(r["model_output"]).strip().lower()[:1] == "a" else 0
                     for r in records], int)
    na, nb = [], []
    for r in records:
        o = parse_options(r["question"])
        out = " " + norm(r["model_output"]) + " "
        a, b = norm(o.get("a", "")), norm(o.get("b", ""))
        na.append(bool(a) and f" {a} " in out)
        nb.append(bool(b) and f" {b} " in out)
    na, nb = np.array(na, int), np.array(nb, int)
    return {
        "is_safe(artifact)": 1 - safe,             # unsafe == not starting with 'a'
        "names_b(attack landed)": nb,
        "not_names_a(unsafe)": 1 - na,
    }


def main():
    os.makedirs(FIGD, exist_ok=True)
    sta = json.load(open(f"{ROOT}/STA_layerauroc_cambrian.json"))
    clean = json.load(open(f"{ROOT}/clean_layerauroc_cambrian.json"))
    B = len(sta[0]["per_layer_conflict"])
    Xs = np.array([r["per_layer_conflict"] for r in sta]).T   # [B,N]
    Xc = np.array([r["per_layer_conflict"] for r in clean]).T
    ls, lc = labels(sta), labels(clean)

    plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.3})

    # ── Fig 1: mean ACF vs depth, clean vs STA (log y) ──────────────────
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(range(B), Xc.mean(1), "-o", ms=3, label="clean (N=500)", color="#1f77b4")
    ax.plot(range(B), Xs.mean(1), "-s", ms=3, label="STA attack (N=500)", color="#d62728")
    ax.set_yscale("log")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("mean ACF  $\\kappa$  (log)")
    ax.set_title("Per-layer mean ACF: attack and clean are indistinguishable")
    ax.legend()
    gap = (Xs.mean(1) - Xc.mean(1)) / (Xc.mean(1) + 1e-12)
    ax.text(0.02, 0.03, f"max relative gap over 32 layers: {np.abs(gap).max():.1%}",
            transform=ax.transAxes, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIGD}/fig1_layer_acf_mean.png"); plt.close(fig)

    # ── Fig 2: per-layer AUROC under the three labels ───────────────────
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = {"is_safe(artifact)": "#7f7f7f",
              "names_b(attack landed)": "#d62728",
              "not_names_a(unsafe)": "#2ca02c"}
    auc_curves = {}
    for k in KEYS:
        y = np.concatenate([ls[k], lc[k]])
        X = np.concatenate([Xs, Xc], 1)
        auc = np.array([auroc(y, X[b]) for b in range(B)])
        auc_curves[k] = auc
        ax.plot(range(B), fold(auc), "-", lw=1.8, color=colors[k],
                label=f"{k}  (max folded={np.nanmax(fold(auc)):.3f})")
        print(f"[fig2] {k:26s} best-folded={np.nanmax(fold(auc)):.4f} "
              f"@L{int(np.nanargmax(fold(auc)))}  mean-folded={np.nanmean(fold(auc)):.4f}")
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("decoder layer"); ax.set_ylabel("folded AUROC")
    ax.set_title("Per-layer ACF AUROC depends entirely on the label you use")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIGD}/fig2_layer_auroc_by_label.png"); plt.close(fig)

    # ── Fig 3: ACF distribution at a mid/late layer, clean vs STA ───────
    for L, tag in ((18, "mid"), (31, "last")):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        bins = np.linspace(0, max(Xs[L].max(), Xc[L].max()), 40)
        ax.hist(Xc[L], bins=bins, alpha=0.55, label=f"clean (mean={Xc[L].mean():.3f})",
                color="#1f77b4", density=True)
        ax.hist(Xs[L], bins=bins, alpha=0.55, label=f"STA (mean={Xs[L].mean():.3f})",
                color="#d62728", density=True)
        ax.set_xlabel(f"ACF  $\\kappa$  at layer {L}")
        ax.set_ylabel("density")
        ax.set_title(f"ACF distribution at layer {L} ({tag}): overlap, no separation")
        ax.legend()
        fig.tight_layout(); fig.savefig(f"{FIGD}/fig3_acf_hist_L{L}.png"); plt.close(fig)

    # ── Fig 4: response behaviour (names_a / names_b) clean vs STA ──────
    def named_a(rec):
        return np.array([(lambda o, out: bool(o.get("a")) and
                          f" {norm(o['a'])} " in out)(parse_options(r["question"]),
                          " " + norm(r["model_output"]) + " ") for r in rec], int)
    na_c, na_s = named_a(clean), named_a(sta)
    nb_c, nb_s = lc["names_b(attack landed)"], ls["names_b(attack landed)"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(3); w = 0.36
    clean_v = [na_c.mean(), nb_c.mean(), np.mean([len(str(r["model_output"]).strip()) == 0 for r in clean])]
    sta_v = [na_s.mean(), nb_s.mean(), np.mean([len(str(r["model_output"]).strip()) == 0 for r in sta])]
    ax.bar(x - w/2, clean_v, w, label="clean", color="#1f77b4")
    ax.bar(x + w/2, sta_v, w, label="STA", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(["names option-a\n(correct)", "names option-b\n(attack landed)", "empty output"])
    ax.set_ylabel("fraction of samples"); ax.set_ylim(0, 1)
    ax.set_title("What Cambrian-8B actually says")
    for i, (c, s) in enumerate(zip(clean_v, sta_v)):
        ax.text(i - w/2, c + .01, f"{c:.0%}", ha="center", fontsize=9)
        ax.text(i + w/2, s + .01, f"{s:.0%}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIGD}/fig4_response_behaviour.png"); plt.close(fig)

    # ── console summary ─────────────────────────────────────────────────
    print("\n=== per-layer mean ACF (clean vs STA) ===")
    for b in (0, 8, 16, 24, 31):
        print(f"  L{b:<2} clean={Xc[b].mean():.5f}  STA={Xs[b].mean():.5f}  "
              f"rel_gap={(Xs[b].mean()-Xc[b].mean())/Xc[b].mean():+.1%}")
    print("\n=== label prevalence (fraction unsafe) ===")
    for k in KEYS:
        print(f"  {k:26s} clean={lc[k].mean():.3f}  STA={ls[k].mean():.3f}")
    print(f"\n[figs] -> {FIGD}")
    for f in sorted(os.listdir(FIGD)):
        print("   ", f)


if __name__ == "__main__":
    main()
