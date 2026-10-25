"""
Plot cross-modal ACF ablation results (bar chart).
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/crossmodal"
INPUT_FILE = os.path.join(RESULT_DIR, "crossmodal_qwen_L11.json")

ORDER = ["clean_image_text", "typo_text_only", "typo_image_only", "typo_image_text"]
LABELS = [
    "Clean Image+Text\n(clean img + question)",
    "Typo Text Only\n(STA question)",
    "Typo Image Only\n(STA img + describe)",
    "Typo Image+Text\n(STA img + question)",
]
COLORS = ["#66c2a5", "#74add1", "#fdae61", "#d7191c"]


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    means = [data[k]["acf_mean"] for k in ORDER]
    stds = [data[k]["acf_std"] for k in ORDER]
    tokens = [data[k]["tokens_mean"] for k in ORDER]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ORDER))
    bars = ax.bar(x, means, width=0.55, color=COLORS, edgecolor="white",
                  linewidth=1.2, yerr=stds, capsize=8,
                  error_kw={"linewidth": 1.5})

    for i, (bar, m, s, t) in enumerate(zip(bars, means, stds, tokens)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.00015,
                f"{m:.6f}\n(~{t:.0f} tok)", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    it = data["typo_image_text"]["acf_mean"]
    io = data["typo_image_only"]["acf_mean"]
    to = data["typo_text_only"]["acf_mean"]
    ct = data["clean_image_text"]["acf_mean"]

    ratio_mod = it / max(io, to) if max(io, to) > 0 else float('inf')
    ratio_atk = it / ct if ct > 0 else float('inf')

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylabel("Mean per-token ACF (Layer 11)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Cross-Modal ACF Ablation — Qwen2.5-VL-7B\n"
        f"Cross-modal = {ratio_mod:.2f}×  |  Attack/Clean = {ratio_atk:.2f}×",
        fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(means) * 1.3)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        out = INPUT_FILE.replace(".json", f".{fmt}")
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
