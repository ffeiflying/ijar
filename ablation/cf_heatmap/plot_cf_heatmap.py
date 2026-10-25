#!/usr/bin/env python
"""
把 run_cf_heatmap_{qwen,internvl,mof,deepseek}.py 产出的 NPZ 渲染成三格热力图。
每个模型读 npz/<model>/ 与对应 image_path，输出到 figs/<model>/。
Usage: python ablation/cf_heatmap/plot_cf_heatmap.py [N]   (N=每模型前 N 个 npz, 0=全部)
"""
import sys
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_IJAR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPZ_ROOT = os.path.join(_IJAR, "results/cf_heatmap/npz")
OUT_ROOT = os.path.join(_IJAR, "results/cf_heatmap/figs")

MODELS = {
    "qwen":     {"layer": 11, "name": "Qwen2.5-VL-7B"},
    "internvl": {"layer": 20, "name": "InternVL2.5-8B"},
    "mof":      {"layer": 25, "name": "MoF-Models-7B"},
    "deepseek": {"layer": 1,  "name": "DeepSeek-VL2-Tiny"},
}


def render_model(model_key, cfg, limit):
    npz_dir = os.path.join(NPZ_ROOT, model_key)
    out_dir = os.path.join(OUT_ROOT, model_key)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(npz_dir):
        print(f"[{model_key}] missing {npz_dir} — skip")
        return 0
    files = [f for f in sorted(os.listdir(npz_dir)) if f.endswith(".npz")]
    if limit > 0:
        files = files[:limit]
    for fn in files:
        d = np.load(os.path.join(npz_dir, fn), allow_pickle=True)
        img = np.asarray(Image.open(str(d["image_path"])).convert("RGB"))
        H, W = img.shape[:2]
        kappa_map = d["kappa_map"].astype(float)
        cam_map = d["gradcam_map"].astype(float)
        gt = str(d["ground_truth"]).strip().lower()[:1]
        pred = str(d["model_output"]).strip().lower()[:1]
        ok = (pred == gt)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].axis("off")
        axes[0].set_title(f"{d['image_id']}\nGT={gt} pred={pred} "
                          f"{'OK' if ok else 'XX'}", fontsize=10)

        axes[1].imshow(img)
        c1 = axes[1].imshow(cam_map, cmap="YlOrRd", alpha=0.55,
                            extent=[0, W, H, 0], interpolation="bilinear",
                            vmin=0, vmax=max(float(cam_map.max()), 1e-9))
        axes[1].set_title("CF-gradient saliency", fontsize=11)
        axes[1].axis("off")
        fig.colorbar(c1, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(img)
        c2 = axes[2].imshow(kappa_map, cmap="RdBu_r", alpha=0.55,
                            extent=[0, W, H, 0], interpolation="bilinear",
                            vmin=float(kappa_map.min()),
                            vmax=float(kappa_map.max()))
        axes[2].set_title(f"CF-value (mean={float(d['mean_kappa']):.4f})", fontsize=11)
        axes[2].axis("off")
        fig.colorbar(c2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(f"{cfg['name']} · Layer L{cfg['layer']} · {d['image_id']}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        stem = fn[:-4]
        fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=200,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"  saved {out_dir}/{stem}.png/.pdf")
    return len(files)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    total = 0
    for model_key, cfg in MODELS.items():
        print(f"=== {cfg['name']} (L{cfg['layer']}) ===")
        total += render_model(model_key, cfg, limit)
    print(f"\nDone. {total} figures rendered.")


if __name__ == "__main__":
    main()
