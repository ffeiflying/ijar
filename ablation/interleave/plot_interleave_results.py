"""
Plot Interleaving Ablation results (IJAR 3.1)
==============================================
3 subplots: AUROC / AUPR / ACC.
Each subplot: grouped by model (rows), bars per mode (cols).
ACC shown as line chart, others as bars.
Style consistent with previous plots — muted colors.
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/interleave"
OUTPUT_DIR = RESULTS_DIR

MODELS = [
    ("DeepSeek-VL2-Tiny", "interleave_deepseek.json"),
    ("Qwen2.5-VL-7B", "interleave_qwen.json"),
    ("InternVL2_5-8B", "interleave_internvl.json"),
]

# Modes displayed — first 3 are for Qwen/InternVL (token-level),
# DeepSeek uses channel fusion so has different modes
ALL_MODES = ["native_only", "concat", "interleaved", "fused"]
ALL_MODES_DISPLAY = {
    "native_only": "native-only", "concat": "concat",
    "interleaved": "interleaved", "fused": "fused",
}

# Muted, desaturated colors — one per mode
MODE_COLORS = {
    "native_only": "#8BAAB8",   # blueish
    "concat": "#B8A894",        # warm-grey
    "interleaved": "#A8B8A0",   # greenish
    "fused": "#C4A2B8",         # mauve
}


def flip_below_half(v):
    if v is None or np.isnan(v): return float('nan')
    return 1.0 - v if v < 0.5 else v


def main():
    # Gather data: per model, detect which modes are present
    all_data = {}
    for model_name, json_file in MODELS:
        json_path = os.path.join(RESULTS_DIR, json_file)
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)
            # Keep only modes that exist in the data
            existing_modes = [m for m in ALL_MODES if m in data]
            if existing_modes:
                all_data[model_name] = {"data": data, "modes": existing_modes}

    if not all_data:
        print("No results found.")
        return

    n_models = len(all_data)

    # 3 columns: AUROC, AUPR, ACC
    fig, axes = plt.subplots(n_models, 3, figsize=(9, 4 * n_models),
                             squeeze=False)

    bar_width = 0.55

    for row_idx, (model_name, info) in enumerate(all_data.items()):
        data = info["data"]
        modes = info["modes"]
        n = len(modes)
        x = np.arange(n)

        colors = [MODE_COLORS[m] for m in modes]
        labels = [ALL_MODES_DISPLAY.get(m, m) for m in modes]

        auroc_vals = [flip_below_half(data.get(m, {}).get("auroc")) for m in modes]
        aupr_vals = [flip_below_half(data.get(m, {}).get("aupr")) for m in modes]
        acc_vals = [data.get(m, {}).get("accuracy", float('nan')) for m in modes]

        # --- Col 0: AUROC (bars) ---
        ax = axes[row_idx][0]
        ax.bar(x, auroc_vals, bar_width, color=colors, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.grid(True, alpha=0.3, axis='y')
        if row_idx == 0:
            ax.set_title("AUROC", fontsize=13, fontweight='bold')
        if row_idx == n_models - 1:
            ax.set_xlabel("Mode", fontsize=10)

        # --- Col 1: AUPR (bars) ---
        ax = axes[row_idx][1]
        ax.bar(x, aupr_vals, bar_width, color=colors, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.grid(True, alpha=0.3, axis='y')
        if row_idx == 0:
            ax.set_title("AUPR", fontsize=13, fontweight='bold')
        if row_idx == n_models - 1:
            ax.set_xlabel("Mode", fontsize=10)

        # --- Col 2: ACC (line with mode-colored markers) ---
        ax = axes[row_idx][2]
        if n > 1:
            ax.plot(x, acc_vals, '-', color='#6B7B8D', linewidth=1.8, alpha=0.5, zorder=2)
        for i, (m, v) in enumerate(zip(modes, acc_vals)):
            ax.plot(i, v, 'o', color=colors[i], markersize=11, zorder=3,
                    markeredgecolor='white', markeredgewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.grid(True, alpha=0.3, axis='y')
        if row_idx == 0:
            ax.set_title("ACC", fontsize=13, fontweight='bold')
        if row_idx == n_models - 1:
            ax.set_xlabel("Mode", fontsize=10)

        # Row label
        axes[row_idx][0].set_ylabel(model_name, fontsize=12, fontweight='bold')

        # Console
        print(f"\n{model_name}:")
        print(f"  {'Mode':<14s} {'AUROC':>8s} {'AUPR':>8s} {'ACC':>8s}")
        for i, mode in enumerate(modes):
            print(f"  {mode:<14s} {auroc_vals[i]:>8.4f} {aupr_vals[i]:>8.4f} "
                  f"{acc_vals[i]:>8.4f}")

    plt.tight_layout(pad=0.5, w_pad=1.5, h_pad=1.5)
    for ext in ["pdf", "png"]:
        out = os.path.join(OUTPUT_DIR, f"fig_interleave_ablation.{ext}")
        plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/fig_interleave_ablation.pdf")


if __name__ == "__main__":
    main()
