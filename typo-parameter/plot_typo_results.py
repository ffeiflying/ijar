"""
Plot typographic-parameter results — fig7_typo_variables.pdf (IJAR 2.1)
--------------------------------------------------------------------------
4×5 grid: rows = models, columns = variable groups.
Each cell: AUROC/AUPR bars + ASR line.
Reads typo_*.json produced by run_typo_*.py.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/typo_parameter"
OUTPUT_DIR = RESULTS_DIR

MODELS = [
    ("MoF-Models-7B",     "typo_mof_L25.json"),
    ("Qwen2.5-VL-7B",     "typo_qwen_L11.json"),
    ("InternVL2_5-8B",    "typo_internvl_L20.json"),
    ("DeepSeek-VL2-Tiny", "typo_deepseek_L1.json"),
]

PANELS = [
    ("Font Size", [
        ("16", "font_size_16"), ("20", "font_size_20"),
        ("24", "font_size_24"), ("28", "font_size_28"),
        ("32", "font_size_32"),
    ]),
    ("Color Contrast", [
        ("Low", "color_contrast_low"),
        ("Medium", "color_contrast_medium"),
        ("High", "color_contrast_high"),
    ]),
    ("Text Opacity", [
        ("30%", "opacity_30"), ("50%", "opacity_50"),
        ("70%", "opacity_70"), ("90%", "opacity_90"),
    ]),
    ("Overlay Position", [
        ("Corner", "position_corner"),
        ("Random", "position_random"),
        ("Center", "position_center"),
    ]),
    ("Font Family", [
        ("Serif", "font_family_serif"),
        ("Sans", "font_family_sans_serif"),
        ("Mono", "font_family_monospace"),
        ("Decor", "font_family_decorative"),
    ]),
]

AUROC_COLOR = "#7EA7CC"
AUPR_COLOR  = "#E8B090"
ASR_COLOR   = "#CC6677"
BAR_WIDTH = 0.30


def flip_below_half(v):
    if v is None or np.isnan(v):
        return float('nan')
    return 1.0 - v if v < 0.5 else v


def main():
    n_rows, n_cols = len(MODELS), len(PANELS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols + 2, 3.2 * n_rows + 1.5),
                             sharey='row')

    for row_idx, (model_name, json_file) in enumerate(MODELS):
        json_path = os.path.join(RESULTS_DIR, json_file)
        if not os.path.exists(json_path):
            print(f"[SKIP] {model_name}: {json_file} not found")
            for col_idx in range(n_cols):
                ax = axes[row_idx][col_idx]
                ax.set_visible(False)
            continue

        with open(json_path) as f:
            data = json.load(f)

        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")

        for col_idx, (title, pairs) in enumerate(PANELS):
            ax = axes[row_idx][col_idx]
            labels = [p[0] for p in pairs]
            keys = [p[1] for p in pairs]
            x = np.arange(len(labels))

            auroc_raw, aupr_raw, asr_vals = [], [], []
            for key in keys:
                if key in data:
                    m = data[key]
                    auroc_raw.append(flip_below_half(m.get("auroc")))
                    aupr_raw.append(flip_below_half(m.get("aupr")))
                    asr = m.get("asr")
                    if asr is None:
                        incorrect = m.get("incorrect", 0)
                        total = m.get("total", 1)
                        asr = incorrect / total if total > 0 else float('nan')
                    asr_vals.append(asr)
                else:
                    auroc_raw.append(float('nan'))
                    aupr_raw.append(float('nan'))
                    asr_vals.append(float('nan'))

            # Bars
            ax.bar(x - BAR_WIDTH/2, auroc_raw, BAR_WIDTH,
                   color=AUROC_COLOR, zorder=2)
            ax.bar(x + BAR_WIDTH/2, aupr_raw, BAR_WIDTH,
                   color=AUPR_COLOR, zorder=2)

            # ASR line
            ax2 = ax.twinx()
            ax2.plot(x, asr_vals, color=ASR_COLOR, marker="D",
                     linewidth=2, markersize=6, zorder=3)

            # Axis
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9, rotation=30 if col_idx == 0 else 0)
            ax.set_ylim(0, 1.10)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
            ax.grid(True, alpha=0.3, axis='y')

            ax2.set_ylim(0, 1.10)
            ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

            # Column titles (top row)
            if row_idx == 0:
                ax.set_title(title, fontsize=14, fontweight='bold')

            # Row labels (first column)
            if col_idx == 0:
                ax.set_ylabel(model_name, fontsize=13, fontweight='bold')

            # Hide right y-axis except last column
            if col_idx != n_cols - 1:
                ax2.tick_params(axis='y', which='both', right=False,
                                labelright=False)

            # Console
            for lb, key in pairs:
                if key in data:
                    m = data[key]
                    auroc = m.get('auroc')
                    aupr = m.get('aupr')
                    asr_v = m.get('asr')
                    if asr_v is None:
                        incorrect = m.get('incorrect', 0)
                        total = m.get('total', 1)
                        asr_v = incorrect / total if total > 0 else float('nan')
                    au_s = f"AUROC={auroc:.4f}" if auroc is not None else "AUROC=N/A"
                    ap_s = f"AUPR={aupr:.4f}" if aupr is not None else "AUPR=N/A"
                    print(f"  [{title}] {lb:8s}: {au_s}, {ap_s}, ASR={asr_v:.4f}")

    # Shared legend at bottom
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=AUROC_COLOR),
        plt.Rectangle((0, 0), 1, 1, fc=AUPR_COLOR),
        plt.Line2D([], [], color=ASR_COLOR, marker="D",
                   linewidth=2, markersize=6),
    ]
    fig.legend(handles, ["AUROC", "AUPR", "ASR"],
               loc='lower center', ncol=3, fontsize=16,
               frameon=True, bbox_to_anchor=(0.5, -0.01),
               handlelength=4.0, handleheight=1.8)

    fig.text(0.05, 0.5, "AUROC / AUPR", va='center',
             rotation='vertical', fontsize=16)
    fig.text(0.97, 0.5, "ASR", va='center',
             rotation='vertical', fontsize=16, color=ASR_COLOR)

    plt.tight_layout(pad=0.5, w_pad=1.5, h_pad=1.5, rect=(0.04, 0.02, 0.96, 0.98))
    for ext in ["pdf", "png"]:
        out = os.path.join(OUTPUT_DIR, f"fig7_typo_variables.{ext}")
        plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/fig7_typo_variables.pdf")


if __name__ == "__main__":
    main()
