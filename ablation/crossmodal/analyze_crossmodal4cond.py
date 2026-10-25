"""
Join unimodal runs (text-only, image-only) with the existing bimodal JSONs and
quantify whether ACF has *discrimination* across the conditions:

  T_sta : STA question text, no image            (new, n=200)
  T_cln : clean question text, no image          (new, n=200)
  I_att : STA attacked image + generic prompt    (new, n=200)
  I_cln : clean source image + generic prompt    (new, n=200)
  C     : clean image + clean question  (congruent)   (existing clean_layerauroc, n=500 -> 200)
  D     : STA image + STA question     (conflicting)  (existing STA_layerauroc,   n=500 -> 200)

Questions:
  Q1 conflict contrast : does ACF(D) exceed ACF(C)  (attacked > clean bimodal)?
  Q2 mono-modal floors : is ACF(D) also above its own unimodal baselines
                         ACF(I_att), ACF(T_sta), while ACF(C) stays near its own
                         unimodal baselines ACF(I_cln), ACF(T_cln)?
  Q3 within-image text : does the typographic overlay raise conflict even with a
                         generic (non-task) prompt: ACF(I_att) vs ACF(I_cln)?
All metrics are reported per layer (mean ACF curves + signed effect sizes), and
the layer chosen by the paper protocol (qwen L0) is highlighted.
"""
import json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results")
CM = os.path.join(ROOT, "crossmodal")
AU = os.path.join(ROOT, "layerauroc")
CL = os.path.join(ROOT, "selectlayer")
B = 28


def load(fname, base=CM):
    d = json.load(open(os.path.join(base, fname)))
    X = np.array([r["per_layer_conflict"] for r in d])   # [n, B]
    idx = [str(r["index"]) for r in d]
    return X, idx


def subset_by_ids(X, ids, keep):
    m = {i: k for k, i in enumerate(ids)}
    sel = [m[i] for i in keep if i in m]
    return X[sel]


def main():
    ids = json.load(open(os.path.join(CM, "crossmodal4cond_ids.json")))
    sta_ids, cln_ids = ids["sta_ids"], ids["clean_ids"]

    XtS, _ = load("text_only_sta_layerwise_qwen.json")
    XtC, _ = load("text_only_cln_layerwise_qwen.json")
    XIa, _ = load("image_only_att_layerwise_qwen.json")
    XIc, _ = load("image_only_cln_layerwise_qwen.json")
    XC, cidx = load("clean_layerauroc_qwen.json", CL)
    XD, didx = load("STA_layerauroc_qwen.json", AU)

    # align bimodal to the same ids used for the new unimodal conditions
    XC = subset_by_ids(XC, cidx, cln_ids)   # clean image + clean question
    XD = subset_by_ids(XD, didx, sta_ids)   # attacked image + STA question

    sets = {"T_sta": XtS, "T_cln": XtC, "I_att": XIa,
            "I_cln": XIc, "C": XC, "D": XD}
    n = len(sta_ids)
    print(f"n per condition: { {k: len(v) for k, v in sets.items()} }")

    # ---------------- per-layer mean curves ----------------
    print("\nPer-layer mean ACF (rows=condition, cols=L0 L1 L3 L5 L8 L14 L27):")
    show = [0, 1, 3, 5, 8, 14, 27]
    hdr = "cond    " + "".join(f"{'L'+str(l):>9}" for l in show)
    print(hdr)
    for k, X in sets.items():
        m = X.mean(0)
        print(f"{k:<7} " + "".join(f"{m[l]:9.4f}" for l in show))

    # ---------------- helpers ----------------
    def cohend(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        sp = np.sqrt((a.var() + b.var()) / 2 + 1e-12)
        return (a.mean() - b.mean()) / sp

    def mw_z(a, b):
        # two-sided normal-approx Mann-Whitney U statistic -> z (no scipy dependency)
        a, b = np.asarray(a, float), np.asarray(b, float)
        na, nb = len(a), len(b)
        allv = np.concatenate([a, b])
        order = allv.argsort(kind="mergesort")
        ranks = np.empty(len(allv))
        ranks[order] = np.arange(1, len(allv) + 1)
        ra = ranks[:na].sum()
        U = ra - na * (na + 1) / 2
        mu = na * nb / 2
        sd = np.sqrt(na * nb * (na + nb + 1) / 12.0)
        return (U - mu) / sd

    def summarize(name, a, b, paired):
        d = cohend(a, b)
        z = mw_z(a, b)
        tag = "paired " if paired else "unpaired"
        return (f"  {name:<22} d={d:+7.2f}  z={z:+7.2f}  "
                f"mean[a]={np.mean(a):.4f} mean[b]={np.mean(b):.4f}")

    L0 = 0
    print("\n---- Effect sizes at the protocol layer L0 ----")
    print("  [a] is the FIRST group in each pair.")
    for nm, a, b, pr in [
        ("D  vs C      (conflict ctrl)", XD[:, L0], XC[:, L0], False),
        ("D  vs I_att  (bimodal over img)", XD[:, L0], XIa[:, L0], True),
        ("D  vs T_sta  (bimodal over txt)", XD[:, L0], XtS[:, L0], True),
        ("C  vs I_cln  (congruent over img)", XC[:, L0], XIc[:, L0], True),
        ("C  vs T_cln  (congruent over txt)", XC[:, L0], XtC[:, L0], True),
        ("I_att vs I_cln (overlay alone)", XIa[:, L0], XIc[:, L0], False),
        ("T_sta vs T_cln (text-only forms)", XtS[:, L0], XtC[:, L0], False),
    ]:
        print(summarize(nm, a, b, pr))

    # ---------------- per-layer signed effect curves ----------------
    print("\nPer-layer signed Cohen's d (D-C and I_att-I_cln):")
    dDC = np.array([cohend(XD[:, l], XC[:, l]) for l in range(B)])
    dIaIc = np.array([cohend(XIa[:, l], XIc[:, l]) for l in range(B)])
    for l in range(B):
        mark = "  <-- L0" if l == 0 else ""
        if l in (0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 20, 27) or abs(dDC[l]) > 1.5:
            print(f"  L{l:<3} d(D-C)={dDC[l]:+6.2f}   d(I_att-I_cln)={dIaIc[l]:+6.2f}{mark}")

    # ---------------- membership AUROC per layer (attack vs clean bimodal) ----
    def auroc(y, s):
        y = np.asarray(y); s = np.asarray(s, float)
        o = np.argsort(s, kind="mergesort"); r = np.empty_like(o, float)
        r[o] = np.arange(len(s)) + 1
        P = int(y.sum()); N = int((1 - y).sum())
        return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)

    y = np.concatenate([np.ones(len(XD)), np.zeros(len(XC))])
    au = [auroc(y, np.concatenate([XD[:, l], XC[:, l]])) for l in range(B)]
    print("\nMembership AUROC (attacked=1 vs clean) per layer:")
    print("  " + " ".join(f"L{l}:{au[l]:.3f}" for l in [0, 1, 2, 3, 4, 6, 8, 14, 27]))

    # ---------------- save summary ----------------
    out = {
        "n_per_cond": {k: len(v) for k, v in sets.items()},
        "layer": B,
        "mean_ACF_per_layer": {k: v.mean(0).tolist() for k, v in sets.items()},
        "cohen_d_D_minus_C_per_layer": dDC.tolist(),
        "cohen_d_Iatt_minus_Icln_per_layer": dIaIc.tolist(),
        "membership_AUROC_D_vs_C_per_layer": au,
        "L0_effects": {
            "D_vs_C": float(cohend(XD[:, 0], XC[:, 0])),
            "D_vs_I_att": float(cohend(XD[:, 0], XIa[:, 0])),
            "D_vs_T_sta": float(cohend(XD[:, 0], XtS[:, 0])),
            "C_vs_I_cln": float(cohend(XC[:, 0], XIc[:, 0])),
            "C_vs_T_cln": float(cohend(XC[:, 0], XtC[:, 0])),
            "I_att_vs_I_cln": float(cohend(XIa[:, 0], XIc[:, 0])),
        },
    }
    with open(os.path.join(CM, "crossmodal4cond_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {os.path.join(CM, 'crossmodal4cond_summary.json')}")


if __name__ == "__main__":
    main()
