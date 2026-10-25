"""
Plot MoF DINO Ablation results (IJAR 3.2)
==========================================
Grouped bars: AUROC / AUPR per mode.
Overlaid line: ACC.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/ablation"
JSON_PATH = os.path.join(RESULTS_DIR, "mof_dino_ablation.json")
OUTPUT_DIR = RESULTS_DIR

MODE_LABELS = [
    "CLIP+DINO\ninterleaved",
    "CLIP+DINO\nconcat",
    "CLIP-only",
    "DINO-only",
]
MODE_ORDER = ["interleaved", "concat", "clip_only", "dino_only"]

AUROC_COLOR = "#AFC5E3"   # soft muted blue
AUPR_COLOR  = "#E8CDB5"   # soft muted warm
ACC_COLOR   = "#8C6B7A"   # muted mauve (line)

BAR_WIDTH = 0.30


def flip_below_half(v):
    if v is None or np.isnan(v):
        return float('nan')
    return 1.0 - v if v < 0.5 else v


def main():
    if not os.path.exists(JSON_PATH):
        print(f"[SKIP] {JSON_PATH} not found. Run run_mof_dino_ablation.py first.")
        return

    with open(JSON_PATH) as f:
        data = json.load(f)

    auroc_vals, aupr_vals, acc_vals = [], [], []
    for mode in MODE_ORDER:
        r = data.get(mode, {})
        auroc_vals.append(flip_below_half(r.get("auroc")))
        aupr_vals.append(flip_below_half(r.get("aupr")))
        acc_vals.append(r.get("accuracy", float('nan')))

    n_modes = len(MODE_ORDER)
    x = np.arange(n_modes)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Bar groups: AUROC + AUPR side by side
    ax.bar(x - BAR_WIDTH/2, auroc_vals, BAR_WIDTH,
           color=AUROC_COLOR, label="AUROC", zorder=2)
    ax.bar(x + BAR_WIDTH/2, aupr_vals, BAR_WIDTH,
           color=AUPR_COLOR, label="AUPR", zorder=2)

    # ACC as overlaid line
    ax.plot(x, acc_vals, color=ACC_COLOR, marker="D",
            linewidth=2.2, markersize=8, label="ACC", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(MODE_LABELS, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylabel("Score", fontsize=13)

    # Legend — compact, placed outside plot area
    ax.legend(fontsize=11, loc='upper left', frameon=True,
              ncol=3, handlelength=1.2, handleheight=1.0,
              columnspacing=0.6, borderpad=0.3, labelspacing=0.3)

    ax.set_title("MoF DINO Ablation (STA-5K-Mini)", fontsize=14, fontweight='bold')
    plt.tight_layout(pad=0.5)

    for ext in ["pdf", "png"]:
        out = os.path.join(OUTPUT_DIR, f"fig_mof_dino_ablation.{ext}")
        plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/fig_mof_dino_ablation.pdf")

    # Console table
    print(f"\n{'='*55}")
    print(f"  {'Mode':<18s} {'AUROC':>8s} {'AUPR':>8s} {'ACC':>8s}")
    print(f"  {'-'*42}")
    for i, mode in enumerate(MODE_ORDER):
        print(f"  {mode:<18s} {auroc_vals[i]:>8.4f} {aupr_vals[i]:>8.4f} "
              f"{acc_vals[i]:>8.4f}")
    print(f"  {'='*55}")


if __name__ == "__main__":
    main()
