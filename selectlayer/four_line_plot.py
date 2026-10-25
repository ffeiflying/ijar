"""
Four-line per-layer ACF plot
============================
Per model, per transformer block, draw 4 mean-ACF curves:
  clean-correct, clean-incorrect, sta-correct, sta-incorrect

Data sources:
  - Clean: results/selectlayer/clean_layerauroc_{model}.json
           (per-sample {ground_truth, model_output, per_layer_conflict})
  - STA:   results/layerauroc/STA_layerauroc_{model}.json
           (per-sample {ground_truth, model_output, per_layer_conflict})

Usage: python selectlayer/four_line_plot.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results"

MODELS = [
    ("Qwen2.5-VL-7B",   28),
    ("InternVL2.5-8B",  32),
    ("MoF-Models-7B",   40),
    ("DeepSeek-VL2-Tiny", 12),
]
MODEL_KEY = {"Qwen2.5-VL-7B": "qwen", "InternVL2.5-8B": "internvl",
             "MoF-Models-7B": "mof", "DeepSeek-VL2-Tiny": "deepseek"}

# Colors: blue = clean, red = STA; solid = correct, dashed = incorrect
STYLE = {
    "clean-correct":    ("#2b83ba", "solid"),
    "clean-incorrect":  ("#80b9dd", (0, (4, 2))),
    "sta-correct":      ("#d7191c", "solid"),
    "sta-incorrect":    ("#ef9a9a", (0, (4, 2))),
}
ORDER = ["clean-correct", "clean-incorrect", "sta-correct", "sta-incorrect"]


def load_samples(path):
    with open(path) as f:
        return json.load(f)


def group_stats(samples, num_layers, prefix):
    """Split samples by correctness, return per-group per-layer mean ACF.
    prefix: 'clean' or 'sta' selects which two groups the samples belong to."""
    groups = {k: [] for k in ORDER}
    for s in samples:
        pred = str(s["model_output"]).strip().lower()[:1]
        gt = str(s["ground_truth"]).strip().lower()[:1]
        correct = (pred == gt)
        pl = np.array(s["per_layer_conflict"][:num_layers])
        if pl.size < num_layers:
            pl = np.pad(pl, (0, num_layers - pl.size), constant_values=np.nan)
        if correct:
            groups[f"{prefix}-correct"].append(pl)
        else:
            groups[f"{prefix}-incorrect"].append(pl)
    means = {}
    counts = {}
    for k in ORDER:
        arr = np.array(groups[k])
        counts[k] = len(arr)
        means[k] = np.nanmean(arr, axis=0) if len(arr) else np.full(num_layers, np.nan)
    return means, counts


def main():
    summary = {}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for ax, (name, num_layers) in zip(axes, MODELS):
        key = MODEL_KEY[name]
        clean_path = f"{RESULT_DIR}/selectlayer/clean_layerauroc_{key}.json"
        sta_path = f"{RESULT_DIR}/layerauroc/STA_layerauroc_{key}.json"

        if not os.path.exists(clean_path):
            print(f"[skip] {name}: missing clean data {clean_path}")
            ax.set_title(f"{name}\n(clean data missing)", fontsize=11, fontweight="bold")
            ax.axis("off")
            continue

        clean = load_samples(clean_path)
        sta = load_samples(sta_path)
        clean_means, clean_counts = group_stats(clean, num_layers, "clean")
        sta_means, sta_counts = group_stats(sta, num_layers, "sta")

        layers = np.arange(num_layers)
        # selected layer from selectlayer results (vertical marker)
        sel_path = f"{RESULT_DIR}/selectlayer/select_layer_result_{key}.json"
        best_layer = None
        if os.path.exists(sel_path):
            best_layer = json.load(open(sel_path)).get("best_layer")

        for k in ORDER:
            c, ls = STYLE[k]
            cnt = clean_counts[k] if k.startswith("clean") else sta_counts[k]
            y = clean_means[k] if k.startswith("clean") else sta_means[k]
            marker = "o" if k.startswith("clean") else "s"
            ax.plot(layers, y, marker=marker, linestyle=ls, color=c,
                    markersize=3, linewidth=1.6,
                    label=f"{k} (n={cnt})", zorder=3)

        if best_layer is not None:
            ax.axvline(x=best_layer, color="black", linestyle="--", linewidth=1.2,
                       alpha=0.5, zorder=2)
            ax.text(best_layer + num_layers * 0.02, ax.get_ylim()[1] * 0.96,
                    f"sel L{best_layer}", fontsize=8, color="black", ha="left")

        ax.set_xlabel("Layer index", fontsize=10)
        ax.set_ylabel("Mean ACF", fontsize=10)
        ax.set_title(f"{name}  ({num_layers} layers)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, num_layers - 0.5)

        summary[name] = {
            "n_layers": num_layers,
            "best_layer": best_layer,
            "counts": {**clean_counts, **sta_counts},
            "mean_acf": {k: clean_means[k].tolist() if k.startswith("clean")
                         else sta_means[k].tolist() for k in ORDER},
        }

        print(f"\n=== {name} (layer counts) ===")
        for k in ORDER:
            cnt = clean_counts[k] if k.startswith("clean") else sta_counts[k]
            m = clean_means[k] if k.startswith("clean") else sta_means[k]
            nz = np.nanmean(m)
            print(f"  {k:<18s} n={cnt:<4d} mean(over layers)={nz:.4f}")

    fig.suptitle("Per-Layer Mean ACF by Group (Clean/STA × Correct/Incorrect)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(f"{RESULT_DIR}/selectlayer", exist_ok=True)
    fig.savefig(f"{RESULT_DIR}/selectlayer/fig_four_line.png", dpi=200,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(f"{RESULT_DIR}/selectlayer/fig_four_line.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"\nSaved: {RESULT_DIR}/selectlayer/fig_four_line.png/.pdf")

    with open(f"{RESULT_DIR}/selectlayer/four_line_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {RESULT_DIR}/selectlayer/four_line_summary.json")


if __name__ == "__main__":
    main()
