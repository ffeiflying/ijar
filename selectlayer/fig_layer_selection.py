"""Figure: the ACF layer-selection process (locked protocol).

Selection (dev = 50 STA + 50 clean, ACF only, answers NEVER used):
    score(l) = |mean_ACF(STA@l) - mean_ACF(clean@l)| / pooled_std(ACF@l)
    l* = argmax_l score(l);  sign = direction of (mean_STA - mean_clean)
Evaluation reported on held-out STA-leftover + clean-leftover, AUROC vs ANSWER correctness.

Per model (column): top = the selection statistic score(l) on the dev 100;
bottom = the raw per-layer ACF (dev 100) that motivates the normalization
(conflict magnitude grows with depth -> a raw |gap| would always pick the last block).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data/liuxiaofei/liuxiaofei/IJAR/results"
BASE = ROOT + "/layerauroc"
OUT = os.path.join(ROOT, "selectlayer", "figs")
os.makedirs(OUT, exist_ok=True)
MODELS = {"qwen": ("Qwen2.5-VL-7B", 28), "internvl": ("InternVL2.5-8B", 32),
          "mof": ("MoF-7B", 40), "deepseek": ("DeepSeek-VL2-Tiny", 12)}

# ---- palette (dataviz reference; categorical 1/2, ink, grid, surface) ----
CLEAN, STA = "#2a78d6", "#eb6834"     # blue, orange (adjacent slots, CVD-gate-safe)
INK, MUTED, AXIS = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SELECT = "#0b0b0b"                    # emphasis disc for chosen layer

def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    if len(set(y.tolist())) < 2: return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty_like(o, float); r[o] = np.arange(len(s)) + 1
    P = int(y.sum()); N = int((1 - y).sum())
    if P == 0 or N == 0: return float("nan")
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)

def wrong(d): return np.array([str(r["model_output"]).strip().lower()[0] != "a" for r in d], int)

# ---- data ----
D = {}
for m, (_, B) in MODELS.items():
    s = json.load(open(f"{BASE}/STA_layerauroc_{m}.json"))
    c = json.load(open(f"{ROOT}/selectlayer/clean_layerauroc_{m}.json"))
    D[m] = dict(B=B,
                Xs=np.array([[r["per_layer_conflict"][l] for r in s] for l in range(B)]), ws=wrong(s),
                Xc=np.array([[r["per_layer_conflict"][l] for r in c] for l in range(B)]), wc=wrong(c))

def escore(a, b):
    mu, ms = a.mean(1), b.mean(1); nu, ns = a.shape[1], b.shape[1]
    po = np.sqrt(((nu - 1) * a.var(1) + (ns - 1) * b.var(1)) / (nu + ns - 2) + 1e-12)
    return np.abs(mu - ms) / po, np.sign(mu - ms)

res = {}
for m, d in D.items():
    B = d["B"]; rng = np.random.RandomState(42)
    i = rng.permutation(500); sd, st = i[:50], i[50:]
    j = rng.permutation(500); cd, ct = j[:50], j[50:]
    es, sg = escore(d["Xs"][:, sd], d["Xc"][:, cd]); l = int(np.argmax(es)); sgn = sg[l]
    X = np.concatenate([d["Xs"][:, st], d["Xc"][:, ct]], 1)
    y = np.concatenate([d["ws"][st], d["wc"][ct]])
    res[m] = dict(B=B, l=l, sgn=sgn, es=es, a=auroc(y, sgn * X[l]),
                  mus=d["Xs"][:, sd].mean(1), muc=d["Xc"][:, cd].mean(1),
                  ses=np.sqrt(d["Xs"][:, sd].var(1) / 50), sec=np.sqrt(d["Xc"][:, cd].var(1) / 50))

# ---- figure: 2 rows (score / raw ACF) x 4 columns (models) ----
names = list(MODELS)
fig = plt.figure(figsize=(14.5, 6.6), dpi=200)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.30,
                      left=0.055, right=0.99, top=0.86, bottom=0.15)
score_axs, acf_axs = [], []
for col, m in enumerate(names):
    score_axs.append(fig.add_subplot(gs[0, col]))
    acf_axs.append(fig.add_subplot(gs[1, col]))
x0 = np.arange(res[m]["B"])

# shared y for score panels (normalized -> comparable across models)
score_ymax = max(res[m]["es"].max() for m in names) * 1.10

for col, m in enumerate(names):
    B = res[m]["B"]; r = res[m]
    x = np.arange(B)
    panel = score_axs[col]
    panel.set_facecolor(SURF)
    panel.plot(x, r["es"], color=INK, lw=1.6, zorder=3)
    panel.set_ylim(0, score_ymax); panel.set_xlim(-0.6, B - 0.4)
    # chosen layer emphasis
    panel.axvline(r["l"], color=MUTED, lw=1.0, ls=":", zorder=1)
    panel.plot(r["l"], r["es"][r["l"]], marker="o", ms=10, mfc=SELECT, mec=SURF,
               mew=2.0, ls="None", zorder=5)
    # in-axes label block (top-left; score is low there early on -> empty space)
    sg = "+" if r["sgn"] > 0 else "−"
    panel.text(0.03, 0.90, f"({'abcd'[col]})", transform=panel.transAxes, fontsize=11, fontweight="bold",
               color=INK, ha="left", va="top")
    panel.text(0.03, 0.78, f"$l^{{*}}$={r['l']}, sign {sg}", transform=panel.transAxes,
               fontsize=9, color=INK, ha="left", va="top")
    panel.text(0.03, 0.70, f"AUROC = {r['a']:.3f}", transform=panel.transAxes,
               fontsize=9, color=MUTED, ha="left", va="top")
    # axes
    for sp in panel.spines.values(): sp.set_visible(False)
    panel.tick_params(axis="x", labelbottom=False, length=0)
    panel.set_yticks([0, 2, 4, 6] if score_ymax >= 6 else [0, 1, 2, 3, 4, 5])
    panel.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    if col == 0:
        panel.set_ylabel("effect size  $|\\Delta\\mu|/\\sigma_{\\text{pooled}}$", fontsize=9.5,
                         color=MUTED, labelpad=4)
    panel.set_title(MODELS[m][0] + f"   (B={B})", fontsize=10.5, color=INK, pad=6)

    # ---- bottom: raw dev ACF means +/- 1 se ----
    ac = acf_axs[col]; ac.set_facecolor(SURF)
    for mu, se, c, z in ((r["mus"], r["ses"], CLEAN, 3), (r["muc"], r["sec"], STA, 4)):
        ac.plot(x, mu, color=c, lw=2.0, zorder=z)
        ac.fill_between(x, mu - se, mu + se, color=c, alpha=0.10, lw=0, zorder=1)
    ac.axvline(r["l"], color=MUTED, lw=1.0, ls=":", zorder=2)
    ac.text(0.03, 0.90, f"({'efgh'[col]})", transform=ac.transAxes, fontsize=11,
            fontweight="bold", color=INK, ha="left", va="top")
    ymax = max((r["mus"] + r["ses"]).max(), (r["muc"] + r["sec"]).max()) * 1.12
    ac.set_ylim(0, ymax); ac.set_xlim(-0.6, B - 0.4)
    step = 5 if B > 16 else (2 if B > 8 else 1)
    ac.set_xticks(range(0, B, step)); ac.set_xticks(range(B), minor=True)
    for sp in ac.spines.values(): sp.set_visible(False)
    ac.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ac.tick_params(axis="x", length=0, labelsize=8.5, colors=MUTED)
    ac.tick_params(axis="y", length=0, labelsize=8.5, colors=MUTED)
    if col == 0:
        ac.set_ylabel("mean ACF conflict", fontsize=9.5, color=MUTED, labelpad=4)
    ac.set_xlabel("block (FFN down-projection input)", fontsize=9, color=MUTED, labelpad=3)

# legend (clean / STA series)
handles = [plt.Line2D([], [], color=CLEAN, lw=2.4, label="clean (safe inputs)"),
           plt.Line2D([], [], color=STA, lw=2.4, label="STA (attacked inputs)"),
           plt.Line2D([], [], marker="o", ms=9, mfc=SELECT, mec=SURF, mew=1.6,
                      ls="None", label="selected layer $l^*$")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, 0.06))
fig.suptitle("Layer selection from per-block ACF conflict on a 100-sample dev set (50 clean + 50 STA); "
             "answer-AUROC then measured on the held-out pool",
             fontsize=11.5, color=INK, y=0.955)
fig.text(0.5, 0.015,
         "Top: the selection statistic (normalized gap;  $\\cdot$ = argmax).  Bottom: the raw ACF that motivates "
         "normalization — magnitude grows with depth, so an un-normalized gap would always pick the last block.",
         ha="center", va="center", fontsize=8.8, color=MUTED)
fig.savefig(os.path.join(OUT, "layer_selection.pdf"))
fig.savefig(os.path.join(OUT, "layer_selection.png"))
print("saved", os.path.join(OUT, "layer_selection.png"))
for m in names:
    r = res[m]
    print(f"{m:<9} l*={r['l']} sign={'+' if r['sgn']>0 else '-'} es_max={r['es'].max():.2f} answer-AUROC={r['a']:.4f}")
