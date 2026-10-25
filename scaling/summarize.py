"""
Aggregate per-layer AUROC / AUPR / accuracy for the scaling experiment.

Reads both the new `scaling/results/*.json` runs and -- optionally -- the
reference runs under `IJAR/results/layerauroc/`, so a reproduction can be
checked side by side.

Metrics use the same convention as `layerauroc/layer_auroc.py`:
    label 0 = correct, 1 = incorrect ; score = ACF at that layer
The answer parser is imported from the reference file so labels stay identical.

Usage:
    python summarize.py                       # scaling results only
    python summarize.py --with-reference      # + IJAR/results/layerauroc
    python summarize.py --plot                # per-layer AUROC figure
    python summarize.py --orient --plot       # flip layers whose AUROC < 0.5

`--orient` handles layers where the conflict score is *anti*-correlated with
being wrong.  Rather than replacing the number with `1 - AUROC`, the score is
negated and both metrics are recomputed, so AUROC becomes exactly 1 - AUROC
and AUPR stays a genuine average precision (AUPR is not symmetric under that
substitution).  The per-layer decision is made separately for each layer.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_IJAR_DIR = os.path.dirname(_HERE)
for p in (_HERE, _IJAR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util  # noqa: E402


def _load_reference_parser():
    ref = os.path.join(_IJAR_DIR, "layerauroc", "layer_auroc.py")
    spec = importlib.util.spec_from_file_location("layer_auroc_ref", ref)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._extract_choice


_extract_choice = _load_reference_parser()


def labels_of(records):
    """0 = correct, 1 = incorrect. Prefer the stored flag; else re-parse."""
    out = []
    for r in records:
        if "correct" in r:
            out.append(0 if r["correct"] else 1)
        else:
            pred = _extract_choice(r["model_output"])
            gt = r["ground_truth"].strip().lower()[:1]
            out.append(0 if pred == gt else 1)
    return out


def per_layer_auroc(records, orient=False):
    """Per-layer (AUROC, AUPR).  With orient=True a layer whose AUROC < 0.5 is
    recomputed on the negated score, i.e. reported as 1 - AUROC."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    y = np.asarray(labels_of(records))
    n_layers = len(records[0]["per_layer_conflict"])
    auroc, aupr, flipped = [], [], 0
    for l in range(n_layers):
        s = np.asarray([r["per_layer_conflict"][l] for r in records], dtype=float)
        if len(set(y.tolist())) < 2 or not np.isfinite(s).all():
            auroc.append(float("nan"))
            aupr.append(float("nan"))
            continue
        if orient and roc_auc_score(y, s) < 0.5:
            s = -s          # anti-correlated layer: flip the score, not the metric
            flipped += 1
        auroc.append(float(roc_auc_score(y, s)))
        aupr.append(float(average_precision_score(y, s)))
    return y, np.asarray(auroc), np.asarray(aupr), flipped


def metrics_for(records, orient=False):
    y, auroc, aupr, flipped = per_layer_auroc(records, orient=orient)
    n = len(y)
    n_ok = int((y == 0).sum())
    return {
        "n_samples": n,
        "n_correct": n_ok,
        "n_incorrect": n - n_ok,
        "accuracy": n_ok / n if n else float("nan"),
        "n_layers": len(records[0]["per_layer_conflict"]),
        "auroc_per_layer": [None if np.isnan(v) else v for v in auroc],
        "aupr_per_layer": [None if np.isnan(v) else v for v in aupr],
        "auroc_L0": None if np.isnan(auroc[0]) else float(auroc[0]),
        "aupr_L0": None if np.isnan(aupr[0]) else float(aupr[0]),
        "auroc_mean_over_layers": float(np.nanmean(auroc)),
        "auroc_max_over_layers": float(np.nanmax(auroc)),
        "best_layer_by_auroc": int(np.nanargmax(auroc)),
        "oriented": bool(orient),
        "n_layers_flipped": int(flipped),
    }


def collect_scaling_results(orient=False):
    rows = {}
    for path in sorted(glob.glob(os.path.join(_HERE, "results", "*.json"))):
        base = os.path.basename(path)
        if base.endswith("_meta.json") or base in ("summary.json", "eval_summary.json",
                                                   "eval_summary_oriented.json"):
            continue
        stem = base[:-5]
        model, _, dataset = stem.rpartition("_")
        if not model or dataset not in ("STA", "RTA"):
            continue
        with open(path, encoding="utf-8") as f:
            recs = json.load(f)
        if not recs:
            continue
        rows.setdefault(model, {})[dataset] = (metrics_for(recs, orient=orient), path)
    return rows


REFERENCE_DIR = os.path.join(_IJAR_DIR, "results", "layerauroc")

# The four models the original IJAR layerauroc sweep covered, both datasets.
REFERENCE = [
    # (display name, filename suffix)
    ("Qwen2.5-VL-7B", "qwen"),
    ("InternVL2_5-8B", "internvl"),
    ("MoF-Models-7B", "mof"),
    ("DeepSeek-VL2-Tiny", "deepseek"),
    ("Cambrian-7B", "cambrian"),
]

def collect_reference(orient=False):
    rows = {}
    for name, suffix in REFERENCE:
        for ds in ("STA", "RTA"):
            p = os.path.join(REFERENCE_DIR, f"{ds}_layerauroc_{suffix}.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    recs = json.load(f)
                rows.setdefault(name + " [IJAR ref]", {})[ds] = (
                    metrics_for(recs, orient=orient), p)
    return rows


def print_table(rows, title):
    print(f"\n{title}")
    print("-" * 100)
    hdr = (f"{'model':32s} {'data':5s} {'N':>4s} {'acc':>7s} {'wrong':>6s} "
           f"{'L0 AUROC':>9s} {'L0 AUPR':>8s} {'best L':>7s} {'max AUROC':>10s} {'flip':>5s}")
    print(hdr)
    for model in sorted(rows):
        for ds in sorted(rows[model]):
            m = rows[model][ds][0]
            a0 = m["auroc_L0"]
            p0 = m["aupr_L0"]
            print(
                f"{model:32s} {ds:5s} {m['n_samples']:>4d} {m['accuracy']:>7.4f} "
                f"{m['n_incorrect']:>6d} "
                f"{(f'{a0:.4f}' if a0 is not None else '   n/a'):>9s} "
                f"{(f'{p0:.4f}' if p0 is not None else '   n/a'):>8s} "
                f"{m['best_layer_by_auroc']:>7d} {m['auroc_max_over_layers']:>10.4f} "
                f"{m['n_layers_flipped']:>5d}"
            )
        print()


def param_count(model):
    p = os.path.join(_HERE, "results", f"{model}_meta.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("params_b")
    except Exception:
        return None


def make_plot(rows, orient, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    colors = plt.cm.viridis(np.linspace(0.05, 0.75, max(len(rows), 1)))
    for ax, ds in zip(axes, ("STA", "RTA")):
        for color, model in zip(colors, sorted(rows)):
            if ds not in rows[model]:
                continue
            m = rows[model][ds][0]
            a = m["auroc_per_layer"]
            pb = param_count(model)
            label = f"{model.split('-Instruct')[0]}" + (f" ({pb:g}B)" if pb else "")
            if m["n_layers_flipped"]:
                label += f"  [flip {m['n_layers_flipped']}/{m['n_layers']}]"
            ax.plot(range(len(a)), a, marker="o", ms=3, lw=1.5, color=color, label=label)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_title(f"{ds} — per-layer AUROC" + ("  (oriented)" if orient else ""),
                     fontweight="bold")
        ax.set_xlabel("text decoder layer")
        ax.set_ylabel("AUROC" + ("  (max(a, 1-a) per layer)" if orient else ""))
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    name = "fig_scaling_auroc_oriented" if orient else "fig_scaling_auroc"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200)
    print(f"saved -> results/{name}.png / .pdf")
    return name


def make_l0_plot(rows, orient, outdir):
    """Layer-0 only, one figure per dataset: models on x, AUROC + acc annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(rows, key=lambda m: (param_count(m) or 0))
    colors = {"STA": "#3b6ea5", "RTA": "#c8553d"}
    saved = []

    for ds in ("STA", "RTA"):
        if not any(ds in rows[m] for m in models):
            continue
        vals, accs, errs = [], [], []
        for m in models:
            if ds in rows[m]:
                mm = rows[m][ds][0]
                vals.append(mm["auroc_L0"])
                accs.append(mm["accuracy"])
                errs.append(mm["n_incorrect"])
            else:
                vals.append(np.nan), accs.append(np.nan), errs.append(np.nan)

        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(5.7, 4.6))
        ax.bar(x, vals, 0.6, color=colors.get(ds, "gray"), zorder=3)
        acc_y = 1.13          # fixed band, so the acc labels line up across models
        for xi, v, a, e in zip(x, vals, accs, errs):
            if not np.isfinite(v):
                continue
            ax.text(xi, v + 0.014, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", zorder=4)
            ax.text(xi, acc_y, f"acc {a:.3f} ({int(e)} err)", ha="center",
                    va="center", fontsize=7.8, color="#333333", zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-Instruct", "") + f"\n({param_count(m):g}B)"
                            if param_count(m) else m for m in models], fontsize=8.8)
        ax.axhline(0.5, color="gray", ls=":", lw=1.2, zorder=2)
        ax.annotate("chance (0.5)", xy=(1.0, 0.5), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color="gray", ha="left", va="center")
        ax.set_ylabel("layer-0 AUROC" + ("  (oriented)" if orient else ""),
                      fontsize=9.5)
        ax.set_title(f"{ds} — layer-0 AUROC vs. model size",
                     fontweight="bold", fontsize=11)
        ax.set_ylim(0.4, 1.25)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        fig.tight_layout()
        name = f"fig_scaling_l0_{ds}" + ("_oriented" if orient else "")
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200)
        plt.close(fig)
        print(f"saved -> results/{name}.png / .pdf")
        saved.append(name)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-reference", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-l0", action="store_true",
                    help="layer-0 only bar chart (models on x, one bar per dataset)")
    ap.add_argument("--orient", action="store_true",
                    help="per layer, if AUROC<0.5 negate the score and recompute "
                         "(reported AUROC becomes 1-AUROC)")
    args = ap.parse_args()

    rows = collect_scaling_results(orient=args.orient)
    if not rows:
        print("no results yet under", os.path.join(_HERE, "results"))
        return 1

    title = "scaling experiment" + (" [oriented]" if args.orient else "")
    print_table(rows, title)

    ref = collect_reference(orient=args.orient) if args.with_reference else {}
    if ref:
        print_table(ref, "reference (IJAR/results/layerauroc)"
                         + (" [oriented]" if args.orient else ""))

    out = {m: {d: rows[m][d][0] for d in rows[m]} for m in rows}
    for m in ref:
        out[m] = {d: ref[m][d][0] for d in ref[m]}
    out_name = "eval_summary_oriented.json" if args.orient else "eval_summary.json"
    out_path = os.path.join(_HERE, "results", out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_path}")

    if args.plot:
        make_plot(rows, args.orient, os.path.join(_HERE, "results"))
    if args.plot_l0:
        make_l0_plot(rows, args.orient, os.path.join(_HERE, "results"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
