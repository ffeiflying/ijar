"""
Layer-wise AUROC Analysis (IJAR Phase 1.1)
============================================
Reads existing layerwise ACF results and computes per-layer AUROC:
  - Label: 0 = model answered correctly, 1 = model got it wrong
  - Score: ACF value at that layer

Key hypothesis:
  ACF value ↑ with depth (absolute conflict grows)
  but AUROC ↓ with depth (discriminative power diluted)

Output:
  - Per-model JSON with per-layer AUROC/AUPR/ACC
  - fig_layerwise_acf_auroc.pdf: (a) ACF value vs layer, (b) AUROC vs layer
"""

import json, os, re
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _extract_choice(output_text):
    """Extract answer choice 'a' or 'b' from model output.
    Handles: 'a', '(a)', 'a)', '[a]', '(a) slipper', etc."""
    text = output_text.strip()
    if not text:
        return None
    # 1) First non-whitespace character
    first = text[0].lower()
    if first in ('a', 'b'):
        rest = text[1:3] if len(text) > 1 else ''
        if not rest or rest[0] in (' ', '.', ',', ')', ']', ':', '\n', '', '('):
            return first
    # 2) Patterns: (a) (b) a) b) [a] [b]
    for pat in [r'\(([abAB])\)', r'\[([abAB])\]',
                r'([abAB])\)', r'\(([abAB])(?:\s|$|[\.\,])',
                r'\b([abAB])[\.\,](?:\s|$)',
                r'(?:^|\s)([abAB])(?:\s|$|\.|\,)']:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1].lower()
    # 3) Standalone a/b
    m = re.search(r'\b([abAB])\b', text)
    if m:
        return m.group(1).lower()
    return None

# ── Config ──────────────────────────────────────────────────────────────
RESULT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/layerauroc"
OUTPUT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/layerauroc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "Qwen2.5-VL-7B": {
        "sta": "STA_layerauroc_qwen.json",
        "rta": "RTA_layerauroc_qwen.json",
        "color": "#4C72B0",
    },
    "InternVL2_5-8B": {
        "sta": "STA_layerauroc_internvl.json",
        "rta": "RTA_layerauroc_internvl.json",
        "color": "#55A868",
    },
    "MoF-Models-7B": {
        "sta": "STA_layerauroc_mof.json",
        "rta": "RTA_layerauroc_mof.json",
        "color": "#C44E52",
    },
    "DeepSeek-VL2-Tiny": {
        "sta": "STA_layerauroc_deepseek.json",
        "rta": "RTA_layerauroc_deepseek.json",
        "color": "#E69F00",
    },
}


def compute_layer_metrics(data):
    """
    For each layer, compute AUROC/AUPR using ACF score to predict correctness.

    Args:
        data: list of dicts with 'model_output', 'ground_truth', 'per_layer_conflict'

    Returns:
        dict with per-layer metrics
    """
    # Determine correctness per sample
    labels = []
    for d in data:
        pred = _extract_choice(d["model_output"])
        gt = d["ground_truth"].strip().lower()[:1]
        labels.append(0 if pred == gt else 1)  # 0=correct, 1=incorrect

    n_correct = sum(1 for l in labels if l == 0)
    n_incorrect = sum(1 for l in labels if l == 1)
    n_layers = len(data[0]["per_layer_conflict"])

    # Per-layer metrics
    auroc_per_layer = []
    aupr_per_layer = []
    mean_acf_correct = []
    mean_acf_incorrect = []
    mean_acf_all = []

    for layer_idx in range(n_layers):
        scores = [d["per_layer_conflict"][layer_idx] for d in data]

        mean_all = float(np.mean(scores))
        mean_acf_all.append(mean_all)

        # Mean ACF for correct vs incorrect
        scores_correct = [s for s, l in zip(scores, labels) if l == 0]
        scores_incorrect = [s for s, l in zip(scores, labels) if l == 1]
        mean_acf_correct.append(float(np.mean(scores_correct)) if scores_correct else float('nan'))
        mean_acf_incorrect.append(float(np.mean(scores_incorrect)) if scores_incorrect else float('nan'))

        # AUROC / AUPR
        if len(set(labels)) < 2:
            auroc_per_layer.append(float('nan'))
            aupr_per_layer.append(float('nan'))
        else:
            try:
                auroc = float(roc_auc_score(labels, scores))
                aupr = float(average_precision_score(labels, scores))
                auroc_per_layer.append(auroc)
                aupr_per_layer.append(aupr)
            except Exception:
                auroc_per_layer.append(float('nan'))
                aupr_per_layer.append(float('nan'))

    return {
        "n_layers": n_layers,
        "n_samples": len(data),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "accuracy": n_correct / len(data) if len(data) > 0 else float('nan'),
        "auroc_per_layer": auroc_per_layer,
        "aupr_per_layer": aupr_per_layer,
        "mean_acf_correct": mean_acf_correct,
        "mean_acf_incorrect": mean_acf_incorrect,
        "mean_acf_all": mean_acf_all,
    }


def main():
    all_results = {}

    for model_name, paths in MODELS.items():
        print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
        all_results[model_name] = {}

        for dataset in ["sta", "rta"]:
            json_path = os.path.join(RESULT_DIR, paths[dataset])
            if not os.path.exists(json_path):
                print(f"  [{dataset.upper()}] SKIP: not found")
                continue

            with open(json_path) as f:
                data = json.load(f)

            metrics = compute_layer_metrics(data)
            all_results[model_name][dataset] = metrics

            layers = range(metrics["n_layers"])
            auroc = metrics["auroc_per_layer"]
            print(f"\n  [{dataset.upper()}] N={metrics['n_samples']}, "
                  f"ACC={metrics['accuracy']:.3f}")
            print(f"  {'Layer':<8} {'AUROC':<10} {'AUPR':<10} "
                  f"{'ACF_correct':<14} {'ACF_incorrect':<14}")
            for l in sorted(set([0, metrics['n_layers']//4,
                                 metrics['n_layers']//2,
                                 3*metrics['n_layers']//4,
                                 metrics['n_layers']-1])):
                print(f"  {l:<8} {auroc[l]:<10.4f} {metrics['aupr_per_layer'][l]:<10.4f} "
                      f"{metrics['mean_acf_correct'][l]:<14.8f} "
                      f"{metrics['mean_acf_incorrect'][l]:<14.8f}")

    # Save detailed results
    out_json = os.path.join(OUTPUT_DIR, "layer_auroc_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────
    # One row per model, two columns: (left) ACF value, (right) AUROC
    # All 4 models; DeepSeek may have NaN AUROC if all samples wrong

    PLOT_MODELS = {
        "Qwen2.5-VL-7B": {"color": "#4C72B0", "sta": "STA_layerauroc_qwen.json"},
        "InternVL2_5-8B": {"color": "#55A868", "sta": "STA_layerauroc_internvl.json"},
        "MoF-Models-7B": {"color": "#C44E52", "sta": "STA_layerauroc_mof.json"},
        "DeepSeek-VL2-Tiny": {"color": "#E69F00", "sta": "STA_layerauroc_deepseek.json"},
    }
    N_MODELS = len(PLOT_MODELS)
    fig, axes = plt.subplots(N_MODELS, 2, figsize=(12, 4 * N_MODELS), squeeze=False)

    for row, (model_name, info) in enumerate(PLOT_MODELS.items()):
        json_path = os.path.join(RESULT_DIR, info["sta"])
        if not os.path.exists(json_path):
            print(f"  [{model_name}] SKIP plot: file not found ({info['sta']})")
            axes[row][0].text(0.5, 0.5, 'No data', ha='center', va='center',
                              transform=axes[row][0].transAxes, fontsize=14, color='gray')
            axes[row][1].text(0.5, 0.5, 'No data', ha='center', va='center',
                              transform=axes[row][1].transAxes, fontsize=14, color='gray')
            continue
        with open(json_path) as f:
            data = json.load(f)
        m = compute_layer_metrics(data)
        color = info["color"]

        layers = np.arange(m["n_layers"])

        # ── Left: ACF value vs layer ──
        ax = axes[row][0]
        y_correct = np.array(m["mean_acf_correct"])
        y_incorrect = np.array(m["mean_acf_incorrect"])

        has_correct = not np.all(np.isnan(y_correct))
        has_incorrect = not np.all(np.isnan(y_incorrect))

        if has_correct:
            ax.plot(layers, y_correct, color=color, linewidth=1.5,
                    linestyle='--', label="Correct")
        if has_incorrect:
            ax.plot(layers, y_incorrect, color=color, linewidth=1.5,
                    linestyle='-', label="Incorrect")

        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("ACF Value", fontsize=10)
        ax.set_title(f"ACF Value — {model_name}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

        # ── Right: AUROC vs layer ──
        ax = axes[row][1]
        y_auroc = np.array(m["auroc_per_layer"])
        valid = ~np.isnan(y_auroc)

        if valid.sum() > 0:
            y_valid = y_auroc[valid]
            layers_valid = layers[valid]
            ax.plot(layers_valid, y_valid, color=color, linewidth=1.5,
                    marker='o', markersize=4)
            ax.set_ylim(0, 1.05)
        else:
            ax.text(0.5, 0.5, 'AUROC N/A\n(no correct or no incorrect samples)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='gray', fontstyle='italic')
            ax.set_ylim(0, 1.05)

        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Random (0.5)')
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("AUROC", fontsize=10)
        ax.set_title(f"Detection AUROC — {model_name}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle("Layer-wise ACF: Magnitude vs Discriminative Power (STA-5K-Mini)",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        out_plot = os.path.join(OUTPUT_DIR, f"fig_layerwise_auroc.{ext}")
        plt.savefig(out_plot, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/fig_layerwise_auroc.pdf")

    # ── Print key conclusion ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Key finding:")
    for model_name in MODELS:
        for dataset in ["sta"]:
            if dataset in all_results.get(model_name, {}):
                m = all_results[model_name][dataset]
                auroc_first = m["auroc_per_layer"][0]
                auroc_last = m["auroc_per_layer"][-1]
                acf_first = m["mean_acf_all"][0]
                acf_last = m["mean_acf_all"][-1]
                trend_auroc = "↓" if auroc_last < auroc_first else "↑"
                trend_acf = "↑" if acf_last > acf_first else "↓"
                print(f"  {model_name}: ACF {acf_first:.4f}→{acf_last:.4f} {trend_acf}, "
                      f"AUROC {auroc_first:.3f}→{auroc_last:.3f} {trend_auroc}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
