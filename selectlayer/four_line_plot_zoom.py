"""
Per-model four-line ACF plot — one figure, 1x4 row, tall panels
================================================================
Four models in a single row. Each panel keeps the same tall
figure height (long y-axis) and the zoomed informative layer
band [0, best_layer+TAIL], so the small clean/STA x
correct/incorrect differences are clearly visible.

Also saves each model's individual tall figure (four_line_{model}.*)
for single-model reference.

Usage: python selectlayer/four_line_plot_zoom.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results"
OUT_DIR = f"{RESULT_DIR}/selectlayer/per_model_figures"

MODELS = [
    ("Qwen2.5-VL-7B",   28),
    ("InternVL2.5-8B",  32),
    ("MoF-Models-7B",   40),
    ("DeepSeek-VL2-Tiny", 12),
]
MODEL_KEY = {"Qwen2.5-VL-7B": "qwen", "InternVL2.5-8B": "internvl",
             "MoF-Models-7B": "mof", "DeepSeek-VL2-Tiny": "deepseek"}

STYLE = {
    "clean-correct":    ("#2b83ba", "solid"),
    "clean-incorrect":  ("#80b9dd", (0, (5, 2))),
    "sta-correct":      ("#d7191c", "solid"),
    "sta-incorrect":    ("#ef9a9a", (0, (5, 2))),
}
ORDER = ["clean-correct", "clean-incorrect", "sta-correct", "sta-incorrect"]
MARKER = {"clean-correct": "o", "clean-incorrect": "o",
          "sta-correct": "s", "sta-incorrect": "s"}

TAIL = 4  # extra layers after best_layer to show in the zoomed view
PANEL_W, PANEL_H = 8, 12  # each panel keeps the tall y-axis


def load_samples(path):
    with open(path) as f:
        return json.load(f)


def group_stats(samples, num_layers, prefix):
    groups = {k: [] for k in ORDER}
    for s in samples:
        pred = str(s["model_output"]).strip().lower()[:1]
        gt = str(s["ground_truth"]).strip().lower()[:1]
        correct = (pred == gt)
        pl = np.array(s["per_layer_conflict"][:num_layers])
        if pl.size < num_layers:
            pl = np.pad(pl, (0, num_layers - pl.size), constant_values=np.nan)
        groups[f"{prefix}-{'correct' if correct else 'incorrect'}"].append(pl)
    means, counts = {}, {}
    for k in ORDER:
        arr = np.array(groups[k])
        counts[k] = len(arr)
        means[k] = np.nanmean(arr, axis=0) if len(arr) else np.full(num_layers, np.nan)
    return means, counts


def draw_model(ax, key, num_layers, show_legend, show_ylabel):
    """Draw one model's four-line zoomed view on ax."""
    clean_path = f"{RESULT_DIR}/selectlayer/clean_layerauroc_{key}.json"
    sta_path = f"{RESULT_DIR}/layerauroc/STA_layerauroc_{key}.json"
    clean_means, clean_counts = group_stats(load_samples(clean_path), num_layers, "clean")
    sta_means, sta_counts = group_stats(load_samples(sta_path), num_layers, "sta")
    layers = np.arange(num_layers)

    sel_path = f"{RESULT_DIR}/selectlayer/select_layer_result_{key}.json"
    best_layer = None
    if os.path.exists(sel_path):
        best_layer = json.load(open(sel_path)).get("best_layer")
    xmax = num_layers - 1 if best_layer is None else min(best_layer + TAIL, num_layers - 1)

    vals = np.stack([(clean_means[k] if k.startswith("clean") else sta_means[k])
                     for k in ORDER], axis=0)  # (4, num_layers)
    band_max = np.nanmax(vals[:, :xmax + 1])
    ymax = band_max * 1.15 if band_max > 0 else 1.0

    for k in ORDER:
        c, ls = STYLE[k]
        mk = MARKER[k]
        cnt = clean_counts[k] if k.startswith("clean") else sta_counts[k]
        y = clean_means[k] if k.startswith("clean") else sta_means[k]
        ax.plot(layers[:xmax + 1], y[:xmax + 1], marker=mk, linestyle=ls,
                color=c, markersize=4.5, linewidth=1.8,
                label=f"{k} (n={cnt})", zorder=3)

    if best_layer is not None:
        ax.axvline(x=best_layer, color="black", linestyle="--", linewidth=1.3,
                   alpha=0.55, zorder=2)
        ax.text(best_layer + 0.15, ymax * 0.96, f"sel L{best_layer}",
                fontsize=9, color="black", ha="left")

    ax.set_xlim(-0.5, xmax + 0.5)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Layer index", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Mean ACF", fontsize=11)
    ax.set_title(f"{key}", fontsize=12, fontweight="bold")
    if show_legend:
        ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    spread_band = np.nanmax(vals[:, :xmax + 1], axis=0) - np.nanmin(vals[:, :xmax + 1], axis=0)
    imax = int(np.nanargmax(spread_band))
    ax.annotate(f"max gap L{imax}\n({spread_band[imax]:.4f})",
                xy=(imax, np.nanmax(vals[:, imax])),
                xytext=(imax, np.nanmax(vals[:, imax]) + ymax * 0.08),
                fontsize=8, ha="center", color="dimgray",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="gray"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # single combined 1x4 figure
    fig, axes = plt.subplots(1, 4, figsize=(4 * PANEL_W, PANEL_H))
    for ax, (name, num_layers) in zip(axes, MODELS):
        key = MODEL_KEY[name]
        if not os.path.exists(f"{RESULT_DIR}/selectlayer/clean_layerauroc_{key}.json"):
            ax.set_title(f"{key}\n(missing clean data)", fontsize=11)
            ax.axis("off")
            continue
        draw_model(ax, key, num_layers,
                   show_legend=(ax is axes[0]),
                   show_ylabel=(ax is axes[0]))

    fig.suptitle("Per-Layer Mean ACF by Group (clean/STA × correct/incorrect)",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(f"{OUT_DIR}/four_line_row.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(f"{OUT_DIR}/four_line_row.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/four_line_row.png/.pdf")

    # individual tall figures as well
    for name, num_layers in MODELS:
        key = MODEL_KEY[name]
        if not os.path.exists(f"{RESULT_DIR}/selectlayer/clean_layerauroc_{key}.json"):
            continue
        fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
        draw_model(ax, key, num_layers, show_legend=True, show_ylabel=True)
        fig.savefig(f"{OUT_DIR}/four_line_{key}.png", dpi=200, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        fig.savefig(f"{OUT_DIR}/four_line_{key}.pdf", bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
    print(f"Saved individual tall figures: four_line_{{model}}.png/.pdf")


if __name__ == "__main__":
    main()
