"""
Plot training loss curves for DINO projector training.
Reads .pth checkpoint files and overlays loss curves.
"""

import os, json
import torch
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/interleave"
OUTPUT_DIR = RESULTS_DIR

# Which checkpoints to plot
CHECKPOINTS = [
    ("internvl", "concat"),
    ("internvl", "interleaved"),
    ("qwen", "concat"),
    ("qwen", "interleaved"),
    ("deepseek", "fused"),
]

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#E69F00"]


def main():
    fig, ax = plt.subplots(figsize=(10, 5))

    for (model, mode), color in zip(CHECKPOINTS, COLORS):
        path = os.path.join(RESULTS_DIR, f"{model}_dino_projector_{mode}.pth")
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found")
            continue

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        losses = ckpt.get("losses", [])
        if not losses:
            print(f"[SKIP] No losses in {path}")
            continue

        steps = np.arange(len(losses))
        label = f"{model} ({mode})"

        # Raw + smoothed (moving average, window=20)
        ax.plot(steps, losses, alpha=0.15, color=color, linewidth=0.5)
        if len(losses) >= 20:
            smoothed = np.convolve(losses, np.ones(20) / 20, mode='valid')
            ax.plot(steps[19:], smoothed, color=color, linewidth=1.5, label=label)
        else:
            ax.plot(steps, losses, color=color, linewidth=1.5, label=label)

        print(f"{model}/{mode}: {len(losses)} steps, "
              f"start={losses[0]:.2f}, end={losses[-1]:.2f}, "
              f"min={min(losses):.2f}, final_smooth={smoothed[-1]:.2f}" if len(losses) >= 20 else "")

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Loss (cross-entropy)", fontsize=13)
    ax.set_title("DINO Projector Training Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = os.path.join(OUTPUT_DIR, f"training_loss.{ext}")
        plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/training_loss.pdf")


if __name__ == "__main__":
    main()
