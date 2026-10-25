"""
Plot layer-wise ACF — DeepSeek-VL2 (IJAR Phase 1.1)
---------------------------------------------------
Simple linear axes with spline smoothing.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/layerwise_acf"
OUTPUT_DIR = RESULTS_DIR

DATASET_FILES = {
    "STA": os.path.join(RESULTS_DIR, "STA_layerwise_acf_deepseek.json"),
    "RTA": os.path.join(RESULTS_DIR, "RTA_layerwise_acf_deepseek.json"),
}
LEGEND_NAMES = {"STA": "STA-5K", "RTA": "RTA-100"}
COLORS = {"STA": "#3498db", "RTA": "#e74c3c"}
NUM_LAYERS = 12


def load_mean_acf(json_path):
    with open(json_path) as f:
        data = json.load(f)
    n_samples = len(data)
    per_layer = np.zeros(NUM_LAYERS)
    for sample in data:
        per_layer += np.array(sample["per_layer_conflict"])
    per_layer /= n_samples
    return per_layer, n_samples


def main():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(NUM_LAYERS)

    for name in ["STA", "RTA"]:
        path = DATASET_FILES[name]
        if not os.path.exists(path):
            print(f"[{name}] skipping — file not found: {path}")
            continue
        cf, n = load_mean_acf(path)
        print(f"[{name}] n={n}, range=[{cf.min():.6f}, {cf.max():.6f}]")
        x_smooth = np.linspace(0, NUM_LAYERS - 1, 200)
        spl = make_interp_spline(x, cf, k=3)
        y_smooth = spl(x_smooth)
        ax.plot(x_smooth, y_smooth, label=LEGEND_NAMES[name],
                linewidth=2.5, color=COLORS[name])

    ax.set_xlabel('Transformer Block', fontsize=18)
    ax.set_ylabel('Attention Conflict', fontsize=18)
    ax.set_xticks(range(0, NUM_LAYERS, 2))
    ax.set_xticklabels([str(i) for i in range(0, NUM_LAYERS, 2)], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper left', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_pdf = os.path.join(OUTPUT_DIR, "fig5_layerwise_acf_deepseek.pdf")
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_layerwise_acf_deepseek.png"),
                bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
