"""
Per-layer ACF KDE distribution plot (IJAR Phase 1.1)
=====================================================
One figure per model, each subplot = one layer.
Shows ACF density of Correct vs Incorrect samples at each layer.

Output: 4 PDFs saved to results/layerauroc/
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

RESULT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/layerauroc"
OUTPUT_DIR = RESULT_DIR

MODELS = {
    "Qwen2.5-VL-7B": {
        "file": "STA_layerauroc_qwen.json",
        "color": "#4C72B0",
        "n_layers": 28,
        "grid": (7, 4),
    },
    "InternVL2_5-8B": {
        "file": "STA_layerauroc_internvl.json",
        "color": "#55A868",
        "n_layers": 32,
        "grid": (8, 4),
    },
    "MoF-Models-7B": {
        "file": "STA_layerauroc_mof.json",
        "color": "#C44E52",
        "n_layers": 40,
        "grid": (8, 5),
    },
    "DeepSeek-VL2-Tiny": {
        "file": "STA_layerauroc_deepseek.json",
        "color": "#E69F00",
        "n_layers": 12,
        "grid": (3, 4),
    },
}

KDE_LINEWIDTH = 1.2
ALPHA_FILL = 0.15
BINS = 40


def plot_one_model(model_name, cfg):
    json_path = os.path.join(RESULT_DIR, cfg["file"])
    if not os.path.exists(json_path):
        print(f"  [{model_name}] SKIP: file not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    # Determine correctness per sample
    labels = []  # 0=correct, 1=incorrect
    for d in data:
        pred = d["model_output"].strip().lower()[:1]
        gt = d["ground_truth"].strip().lower()[:1]
        labels.append(0 if pred == gt else 1)

    n_correct = sum(1 for l in labels if l == 0)
    n_incorrect = sum(1 for l in labels if l == 1)
    n_layers = cfg["n_layers"]
    rows, cols = cfg["grid"]
    color = cfg["color"]

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"  Correct: {n_correct}, Incorrect: {n_incorrect}")
    print(f"  Grid: {rows}×{cols} for {n_layers} layers")
    print(f"{'='*50}")

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows),
                              squeeze=False)

    for layer_idx in range(n_layers):
        ax = axes[layer_idx // cols][layer_idx % cols]

        # Collect ACF scores for this layer
        scores_correct = [d["per_layer_conflict"][layer_idx]
                         for d, l in zip(data, labels) if l == 0]
        scores_incorrect = [d["per_layer_conflict"][layer_idx]
                           for d, l in zip(data, labels) if l == 1]

        # KDE for correct samples
        if len(scores_correct) >= 3:
            try:
                kde_c = gaussian_kde(scores_correct)
                x_range = np.linspace(min(scores_correct), max(scores_correct), 200)
                ax.plot(x_range, kde_c(x_range), color=color, linestyle='--',
                        linewidth=KDE_LINEWIDTH, label="Correct")
                ax.fill_between(x_range, kde_c(x_range), alpha=ALPHA_FILL,
                                color=color)
            except Exception:
                ax.hist(scores_correct, bins=BINS, density=True, alpha=0.4,
                        color=color, edgecolor=color, linestyle='--')

        # KDE for incorrect samples
        if len(scores_incorrect) >= 3:
            try:
                kde_i = gaussian_kde(scores_incorrect)
                x_range = np.linspace(min(scores_incorrect), max(scores_incorrect), 200)
                ax.plot(x_range, kde_i(x_range), color=color, linestyle='-',
                        linewidth=KDE_LINEWIDTH, label="Incorrect")
                ax.fill_between(x_range, kde_i(x_range), alpha=ALPHA_FILL,
                                color=color)
            except Exception:
                ax.hist(scores_incorrect, bins=BINS, density=True, alpha=0.4,
                        color=color, edgecolor=color, linestyle='-')

        # If only one class exists
        if len(scores_correct) < 2 and len(scores_incorrect) < 2:
            all_scores = scores_correct + scores_incorrect
            if len(all_scores) >= 2:
                ax.hist(all_scores, bins=BINS, density=True, alpha=0.4,
                        color=color)

        ax.set_title(f"L{layer_idx}", fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=6)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide unused subplots
    for idx in range(n_layers, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(f"Per-Layer ACF KDE Distribution — {model_name}",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.text(0.5, 0.01, "ACF Value", ha='center', fontsize=11)
    fig.text(0.01, 0.5, "Density", va='center', rotation='vertical',
             fontsize=11)

    # Single legend for the whole figure
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc='upper right',
                   bbox_to_anchor=(0.99, 0.98), fontsize=9, ncol=2)

    plt.tight_layout(rect=[0.02, 0.03, 1, 0.97])

    safe_name = model_name.replace(" ", "_").replace(".", "-")
    for ext in ["pdf", "png"]:
        out_path = os.path.join(OUTPUT_DIR, f"fig_kde_{safe_name}.{ext}")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: fig_kde_{safe_name}.pdf")


def main():
    for model_name, cfg in MODELS.items():
        plot_one_model(model_name, cfg)
    print(f"\nDone. All KDE figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
