"""
Plot per-layer ACF: Clean vs STA for all 4 models.
Shows why the selected layer is chosen — max |relative change|.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

RESULT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results"

MODELS = {
    "InternVL2.5-8B": {
        "result_file": f"{RESULT_DIR}/selectlayer/select_layer_result_internvl.json",
        "clean_file": f"{RESULT_DIR}/selectlayer/clean_layerauroc_internvl.json",
        "sta_file": f"{RESULT_DIR}/layerauroc/STA_layerauroc_internvl.json",
        "num_layers": 32,
    },
    "Qwen2.5-VL-7B": {
        "result_file": f"{RESULT_DIR}/selectlayer/select_layer_result_qwen.json",
        "num_layers": 28,
    },
    "DeepSeek-VL2-Tiny": {
        "result_file": f"{RESULT_DIR}/selectlayer/select_layer_result_deepseek.json",
        "num_layers": 12,
    },
    "MoF-Models-7B": {
        "result_file": f"{RESULT_DIR}/selectlayer/select_layer_result_mof.json",
        "num_layers": 40,
    },
}

COLOR_CLEAN = "#2b83ba"   # blue
COLOR_STA = "#d7191c"     # red


def load_data():
    """Load per-model (mean_clean, mean_sta, best_layer, sign, selected_auroc)."""
    data = {}

    for name, cfg in MODELS.items():
        with open(cfg["result_file"]) as f:
            res = json.load(f)

        best_layer = res["best_layer"]
        sign = res["sign"]
        selected_auroc = res["selected_auroc"]

        if "mean_clean" in res and "mean_sta_calib" in res:
            mean_clean = np.array(res["mean_clean"])
            mean_sta = np.array(res["mean_sta_calib"])
        else:
            # InternVL: load from separate files
            with open(cfg["clean_file"]) as f:
                clean_data = json.load(f)
            with open(cfg["sta_file"]) as f:
                sta_data = json.load(f)
            mean_clean = np.mean([d["per_layer_conflict"] for d in clean_data], axis=0)
            mean_sta = np.mean([d["per_layer_conflict"] for d in sta_data], axis=0)

        data[name] = {
            "mean_clean": mean_clean,
            "mean_sta": mean_sta,
            "best_layer": best_layer,
            "sign": sign,
            "selected_auroc": selected_auroc,
            "num_layers": len(mean_clean),
        }

    return data


def plot(data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (name, d) in zip(axes, data.items()):
        layers = np.arange(d["num_layers"])
        mean_clean = d["mean_clean"]
        mean_sta = d["mean_sta"]
        best = d["best_layer"]
        sign = d["sign"]
        auroc = d["selected_auroc"]

        # Relative change for annotation
        eps = 1e-8
        rel_change = (mean_sta - mean_clean) / (mean_clean + eps)
        direction = "↑" if sign > 0 else "↓"

        ax.plot(layers, mean_clean, 'o-', color=COLOR_CLEAN, markersize=3,
                linewidth=1.5, label="Clean (50 samples)", zorder=3)
        ax.plot(layers, mean_sta, 's-', color=COLOR_STA, markersize=3,
                linewidth=1.5, label="STA (50 samples)", zorder=3)

        # Mark best layer
        ax.axvline(x=best, color="black", linestyle="--", linewidth=1.2,
                   alpha=0.6, zorder=2)
        y_best = max(mean_clean[best], mean_sta[best])
        ax.annotate(
            f"L{best} {direction}\nAUROC={auroc:.3f}\nΔrel={rel_change[best]*100:+.1f}%",
            xy=(best, y_best),
            xytext=(best + d["num_layers"] * 0.08, y_best * 1.15),
            fontsize=8, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9),
            zorder=10,
        )

        ax.set_xlabel("Layer index", fontsize=10)
        ax.set_ylabel("Mean ACF", fontsize=10)
        ax.set_title(f"{name}  ({d['num_layers']} layers)", fontsize=11,
                     fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, d["num_layers"] - 0.5)

    fig.suptitle(
        "Per-Layer ACF: Clean vs STA (Attacked) Samples",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    out_path = f"{RESULT_DIR}/selectlayer/plot_layer_selection.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out_path}")

    out_pdf = out_path.replace(".png", ".pdf")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved: {out_pdf}")

    plt.close(fig)


def main():
    print("Loading data...")
    data = load_data()

    print("\nSummary:")
    print(f"{'Model':<22s} {'Best':<8s} {'Sign':<6s} {'AUROC':<10s} {'Layers':<8s}")
    print("-" * 56)
    for name, d in data.items():
        direction = "high→attacked" if d["sign"] > 0 else "low→attacked"
        print(f"{name:<22s} L{d['best_layer']:<7d} {direction:<14s} "
              f"{d['selected_auroc']:<10.4f} {d['num_layers']:<8d}")

    print("\nPlotting...")
    plot(data)
    print("Done.")


if __name__ == "__main__":
    main()
