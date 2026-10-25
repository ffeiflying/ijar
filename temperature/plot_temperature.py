"""
Plot temperature analysis — fig6_temperature.pdf (IJAR Phase 1.2)
-----------------------------------------------------------------
2×2 subplots: one model per panel, AUROC + AUPR vs Temperature (STA).
RTA is printed to console only (baseline reference).
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/temperature"
OUTPUT_DIR = RESULTS_DIR

MODELS = [
    ("Qwen2.5-VL-7B",    os.path.join(RESULTS_DIR, "temperature_qwen_L11.json")),
    ("MoF-Models-7B",     os.path.join(RESULTS_DIR, "temperature_mof_L25.json")),
    ("InternVL2_5-8B",   os.path.join(RESULTS_DIR, "temperature_internvl_L20.json")),
    ("DeepSeek-VL2-Tiny", os.path.join(RESULTS_DIR, "temperature_deepseek_L1.json")),
]

# Consistent colors across subplots
AUROC_COLOR = "#0072B2"   # Okabe-Ito blue
AUPR_COLOR  = "#D55E00"   # Okabe-Ito vermilion

XTICKS = [0.5, 1.0, 1.5, 2.0]
PLOT_TEMPS = {0.5, 1.0, 1.5, 2.0}


def load_results(json_path):
    with open(json_path) as f:
        return json.load(f)


def flip_below_half(v):
    if v is None or np.isnan(v):
        return float('nan')
    return 1.0 - v if v < 0.5 else v


def extract_metric(results, dataset, metric):
    """Return (temps, raw_vals, flipped_vals) for a given dataset+metric."""
    temps = [r["temperature"] for r in results]
    raw, flipped = [], []
    for r in results:
        v = r[dataset][metric]
        v = v if v is not None else float('nan')
        raw.append(v)
        flipped.append(flip_below_half(v))
    return temps, raw, flipped


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, (model_name, path) in enumerate(MODELS):
        ax = axes[idx]
        if not os.path.exists(path):
            print(f"[{model_name}] skipping — file not found: {path}")
            ax.set_title(model_name, fontsize=14)
            continue

        results = load_results(path)
        results = [r for r in results if r["temperature"] in PLOT_TEMPS]
        print(f"\n[{model_name}]")

        # ---- STA (main result) ----
        temps, auroc_raw, auroc_plot = extract_metric(results, "STA", "auroc")
        _,     aupr_raw,  aupr_plot  = extract_metric(results, "STA", "aupr")

        valid_auroc = [(t, v) for t, v in zip(temps, auroc_plot) if not np.isnan(v)]
        valid_aupr  = [(t, v) for t, v in zip(temps, aupr_plot)  if not np.isnan(v)]

        if valid_auroc:
            vt, vv = zip(*valid_auroc)
            ax.plot(vt, vv, color=AUROC_COLOR, marker="o", linestyle="-",
                    linewidth=2.5, markersize=10, label="AUROC")
        if valid_aupr:
            vt, vv = zip(*valid_aupr)
            ax.plot(vt, vv, color=AUPR_COLOR, marker="s", linestyle="--",
                    linewidth=2.5, markersize=10, label="AUPR")

        ax.set_title(model_name, fontsize=16)
        ax.set_xlabel("Generation Temperature", fontsize=18)
        ax.set_ylabel("Score", fontsize=18)
        ax.set_xticks(XTICKS)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=14)

        # Print summary
        print(f"  STA AUROC: {[(t, f'{r:.4f}→{p:.4f}') for t, r, p in zip(temps, auroc_raw, auroc_plot)]}")
        print(f"  STA AUPR:  {[(t, f'{r:.4f}→{p:.4f}') for t, r, p in zip(temps, aupr_raw, aupr_plot)]}")

        # ---- RTA (console only) ----
        _, rta_auroc_raw, rta_auroc_plot = extract_metric(results, "RTA", "auroc")
        _, rta_aupr_raw,  rta_aupr_plot  = extract_metric(results, "RTA", "aupr")
        print(f"  RTA AUROC: {[(t, f'{r:.4f}→{p:.4f}') for t, r, p in zip(temps, rta_auroc_raw, rta_auroc_plot)]}")
        print(f"  RTA AUPR:  {[(t, f'{r:.4f}→{p:.4f}') for t, r, p in zip(temps, rta_aupr_raw, rta_aupr_plot)]}")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = os.path.join(OUTPUT_DIR, f"fig6_temperature.{ext}")
        plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/fig6_temperature.pdf")


if __name__ == "__main__":
    main()
