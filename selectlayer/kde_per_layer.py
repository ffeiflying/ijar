"""
Per-layer KDE of per-sample ACF — four groups (clean/STA × correct/incorrect)
================================================================================
One figure per model, one subplot per layer. Each subplot shows the density
(KDE) of per-sample ACF values for the 4 groups, with the same colour /
linestyle scheme as four_line_plot_zoom.py:
    clean-correct   #2b83ba solid      clean-incorrect   #80b9dd dashed
    sta-correct     #d7191c solid      sta-incorrect     #ef9a9a dashed

The selected (best) layer is outlined in black and its title bolded.

Usage: python selectlayer/kde_per_layer.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RESULT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results"
OUT_DIR = f"{RESULT_DIR}/selectlayer/per_model_figures"

MODELS = [
    ("qwen",     "Qwen2.5-VL-7B",    28),
    ("internvl", "InternVL2.5-8B",   32),
    ("mof",      "MoF-Models-7B",    40),
    ("deepseek", "DeepSeek-VL2-Tiny",12),
]

STYLE = {
    "clean-correct":   ("#2b83ba", "solid"),
    "clean-incorrect": ("#80b9dd", (0, (5, 2))),
    "sta-correct":     ("#d7191c", "solid"),
    "sta-incorrect":   ("#ef9a9a", (0, (5, 2))),
}
ORDER = ["clean-correct", "clean-incorrect", "sta-correct", "sta-incorrect"]
MARKER = {"clean-correct": "o", "clean-incorrect": "o",
          "sta-correct": "s", "sta-incorrect": "s"}

KDE_NPTS = 400       # x grid points per subplot
X_PAD = 0.05         # relative padding of the shared x range


def load_samples(path):
    with open(path) as f:
        return json.load(f)


def collect_groups(samples, num_layers, prefix):
    """Return (groups[k][layer]=list of values, counts[k])."""
    groups = {k: [[] for _ in range(num_layers)] for k in ORDER}
    counts = {k: 0 for k in ORDER}
    for s in samples:
        pred = str(s["model_output"]).strip().lower()[:1]
        gt = str(s["ground_truth"]).strip().lower()[:1]
        correct = (pred == gt)
        key = f"{prefix}-{'correct' if correct else 'incorrect'}"
        pl = s["per_layer_conflict"][:num_layers]
        for i, v in enumerate(pl):
            groups[key][i].append(float(v))
        counts[key] += 1
    return groups, counts


def draw_model_figure(key, name, num_layers):
    clean = load_samples(f"{RESULT_DIR}/selectlayer/clean_layerauroc_{key}.json")
    sta = load_samples(f"{RESULT_DIR}/layerauroc/STA_layerauroc_{key}.json")
    clean_g, clean_c = collect_groups(clean, num_layers, "clean")
    sta_g, sta_c = collect_groups(sta, num_layers, "sta")
    groups = {k: clean_g[k] if k.startswith("clean") else sta_g[k] for k in ORDER}
    counts = {k: clean_c[k] if k.startswith("clean") else sta_c[k] for k in ORDER}

    sel_path = f"{RESULT_DIR}/selectlayer/select_layer_result_{key}.json"
    best_layer = None
    if os.path.exists(sel_path):
        best_layer = json.load(open(sel_path)).get("best_layer")

    ncols = int(np.ceil(np.sqrt(num_layers)))
    nrows = int(np.ceil(num_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for layer in range(num_layers):
        ax = axes[layer]
        vals_all = np.concatenate([groups[k][layer] for k in ORDER]) if any(
            len(groups[k][layer]) for k in ORDER) else np.array([0.0])
        if vals_all.size == 0:
            ax.axis("off")
            continue
        lo, hi = float(vals_all.min()), float(vals_all.max())
        pad = (hi - lo) * X_PAD if hi > lo else 0.05
        x = np.linspace(lo - pad, hi + pad, KDE_NPTS)

        ymax = 0.0
        for k in ORDER:
            vals = np.asarray(groups[k][layer], dtype=float)
            if vals.size < 2:
                continue
            try:
                kde = stats.gaussian_kde(vals)
                y = kde(x)
            except Exception:
                continue
            ymax = max(ymax, float(np.nanmax(y)))
            c, ls = STYLE[k]
            ax.plot(x, y, linestyle=ls, color=c, linewidth=1.6, zorder=3)

        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        is_best = (best_layer is not None and layer == best_layer)
        ax.set_title(f"L{layer}", fontsize=10,
                     fontweight="bold" if is_best else "normal")
        if is_best:
            for sp in ax.spines.values():
                sp.set_edgecolor("black")
                sp.set_linewidth(2.2)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, lw=0.5)
        if layer % ncols != 0:
            ax.set_yticklabels([])
        if layer // ncols != nrows - 1:
            ax.set_xticklabels([])

    # legend on the best-layer subplot (or first)
    leg_ax = axes[best_layer] if best_layer is not None else axes[0]
    handles = [plt.Line2D([0], [0], color=STYLE[k][0], linestyle=STYLE[k][1],
                          lw=2, label=f"{k} (n={counts[k]})") for k in ORDER]
    leg_ax.legend(handles=handles, fontsize=8, loc="best",
                  framealpha=0.9, title="group (per-sample count)")

    fig.suptitle(f"{name} ({num_layers} layers) — per-sample ACF density by group",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT_DIR}/kde_{key}.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(f"{OUT_DIR}/kde_{key}.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  saved kde_{key}.png/.pdf   (best L{best_layer}, n="
          + ", ".join(f"{k}:{counts[k]}" for k in ORDER) + ")")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for key, name, num_layers in MODELS:
        cl = f"{RESULT_DIR}/selectlayer/clean_layerauroc_{key}.json"
        if not os.path.exists(cl):
            print(f"[skip] {name}: missing {cl}")
            continue
        draw_model_figure(key, name, num_layers)


if __name__ == "__main__":
    main()
