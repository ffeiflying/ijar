"""E1 + E2 (DS_FOLLOWUP_PLAN): InternVL2_5-8B three-config sample-level ACF + probability baselines.

New file; does NOT modify run_interleave_internvl.py or any env file.

Why a rewrite instead of reusing run_interleave_internvl.py:
  The old script read `all_input_down_proj` from the trust_remote_code cache. A probe
  (probe_e1_struct.py / probe_e2_index.py, 2026-09-13) showed that in THIS env the
  DINO code path returns ONE tensor per step, not a 32-layer list, because the
  append in the cached `modeling_internlm2.py` is commented out in the copy actually
  loaded. So `Layer0ACFComputer.layer_idx` was not selecting a layer. Here we capture
  the per-layer MLP down-projection INPUT with explicit forward_pre_hooks, which is
  deterministic and independent of which cache copy is loaded.

Protocol (aligned with selectlayer/acf_layer_select_final.py, the locked protocol):
  - greedy (do_sample=False), torch.manual_seed(42), MAX_NEW_TOKENS=128
  - sample order: STA = csv order; clean = csv.sample(n=500, random_state=42) order
    (so positional dev/eval split of the canonical script applies verbatim)
  - dev = 50 STA + 50 clean, drawn with np.random.RandomState(42); rest = eval pool
  - ACF weights = /data/liuxiaofei/uncertainty/InternVL/weight/*.pth (same source as
    the canonical native results); verified against the live w2.weight at startup
  - acf        = mean kappa over all forward calls (prefill + decode), last position
                 (== the canonical runner's averaging)
  - acf_decision = kappa of the FIRST forward call only (prefill last position = the
                 decision token); added to check output-length coupling (Cambrian lesson)

Outputs (results/e1_e2/):
  e1_internvl_sample_layer.csv  long format, one row per (sample, layer)
  e1_internvl_probs.csv         per-sample probability baselines

Env: internvl. Usage: CUDA_VISIBLE_DEVICES=1 python run_e1_e2_internvl.py
"""
import os
import sys

for _p in ("~/.local/lib/python3.10/site-packages",
           "~/.local/lib/python3.9/site-packages",
           "~/.local/lib/python3.8/site-packages"):
    _p = os.path.expanduser(_p)
    while _p in sys.path:
        sys.path.remove(_p)

_THIS = os.path.dirname(os.path.abspath(__file__))
_IJAR = os.path.dirname(os.path.dirname(_THIS))
if _IJAR not in sys.path:
    sys.path.insert(0, _IJAR)
sys.path.insert(0, os.path.join(_THIS, "internvl"))

import csv  # noqa: E402
import time  # noqa: E402
import re  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from torchvision.transforms.functional import InterpolationMode  # noqa: E402

from envidence import EvidenceModel  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────
MODEL_ID = "OpenGVLab/InternVL2_5-8B"
MODEL_CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/InternVL/weight"
WEIGHT_PATTERN = "internvl_MLP_Layer{layer}_down_proj.pth"
N_LAYERS = 32
SEED = 42
MAX_NEW_TOKENS = 128
MODES = os.environ.get("E1_MODES", "native_only,concat,interleaved").split(",")

# Dataset override (E6 second-dataset axis).  Defaults = the original STA + clean.
#   E1_ATTACK_CSV / E1_ATTACK_IMG  e.g. datasets/RTA/rta_mini.csv + datasets/RTA/images
#   E1_TAG                         output filename suffix, e.g. "rta"
#   E1_MATCH_OPTION_TEXT=1         RTA-style prompts ask for the full option WORD, not a
#                                  letter -> fall back to option-text matching
_TAG = os.environ.get("E1_TAG", "")
SUFFIX = f"_{_TAG}" if _TAG else ""
MATCH_OPT = os.environ.get("E1_MATCH_OPTION_TEXT", "0") == "1"

STA_CSV = os.environ.get("E1_ATTACK_CSV",
                         os.path.join(_IJAR, "datasets", "STA", "sta_mini.csv"))
STA_IMG = os.environ.get("E1_ATTACK_IMG",
                         os.path.join(_IJAR, "datasets", "STA", "images"))
CLEAN_CSV = os.environ.get("E1_CLEAN_CSV",
                           os.path.join(_IJAR, "typo-parameter", "clean", "data.csv"))
CLEAN_IMG = os.environ.get("E1_CLEAN_IMG",
                           os.path.join(_IJAR, "typo-parameter", "clean", "images"))
PROJECTOR_DIR = os.path.join(_IJAR, "results", "interleave")
OUT_DIR = os.path.join(_IJAR, "results", "e1_e2")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ── Image preprocessing (copied verbatim) ───────────────────────────────
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    out = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        out.append(resized_img.crop(box))
    if use_thumbnail and len(out) != 1:
        out.append(image.resize((image_size, image_size)))
    return out


def load_image(path, input_size=448, max_num=1):
    image = Image.open(path).convert('RGB')
    tf = build_transform(input_size)
    return torch.stack([tf(i) for i in dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)])


def resolve(img_dir, image_id):
    for ext in ("", ".png", ".jpg", ".jpeg"):
        p = os.path.join(img_dir, f"{image_id}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(image_id)


def extract_choice(output_text):
    """Copied verbatim from run_interleave_internvl.py."""
    text = output_text.strip()
    if not text:
        return None
    first = text[0].lower()
    if first in ('a', 'b'):
        rest = text[1:3] if len(text) > 1 else ''
        if not rest or rest[0] in (' ', '.', ',', ')', ']', ':', '\n', '', '('):
            return first
    for pat in [r'\(([abAB])\)', r'\[([abAB])\]',
                r'([abAB])\)', r'\(([abAB])(?:\s|$|[\.\,])',
                r'\b([abAB])[\.\,](?:\s|$)',
                r'(?:^|\s)([abAB])(?:\s|$|\.|\,)']:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1].lower()
    m = re.search(r'\b([abAB])\b', text)
    if m:
        return m.group(1).lower()
    return None


def parse_options(q):
    ma = re.search(r"\(a\)\s*(.+?)\s*\(b\)", str(q))
    mb = re.search(r"\(b\)\s*(.+?)\s*($|\(c\))", str(q))
    return (ma.group(1).strip() if ma else "",
            mb.group(1).strip() if mb else "")


def match_option_text(resp, ta, tb):
    """RTA-style: the prompt asks for the full option word ('(a) slipper (b) card').

    Pick whichever option text appears, and only fall back to a bare letter when the
    response is nothing but one.  Ambiguous / unparsable -> None (counted wrong).
    """
    r = str(resp).strip().lower()
    m_a = bool(ta) and re.search(rf"\b{re.escape(ta.lower())}\b", r) is not None
    m_b = bool(tb) and re.search(rf"\b{re.escape(tb.lower())}\b", r) is not None
    if m_a and not m_b:
        return "a"
    if m_b and not m_a:
        return "b"
    s = re.fullmatch(r"\(?([ab])\)?[\.\)]?", r)
    return s.group(1) if s else None


# ── Per-layer ACF via explicit hooks ────────────────────────────────────
class LayerACFCollector:
    """Captures the MLP down-projection INPUT of every decoder layer.

    Each forward call starts at layer 0, so when layer 0's hook fires we open a new
    record and fill records[-1][i] as layers 0..N-1 fire in order.
    """

    def __init__(self, weight_base, weight_pattern, n_layers, device):
        self.n_layers = n_layers
        self.device = device
        self.evidence = []
        for i in range(n_layers):
            w = torch.load(os.path.join(weight_base, weight_pattern.format(layer=i)),
                           map_location="cpu")
            self.evidence.append(EvidenceModel(w.to(device)))
        self.records = []
        self.handles = []

    def _hook(self, i):
        def fn(module, args):
            x = args[0]
            if i == 0:
                self.records.append([None] * self.n_layers)
            x = x[..., -1, :]  # last position -> [B, H]
            self.records[-1][i] = x[0].detach()
        return fn

    def attach(self, model):
        layers = model.language_model.model.layers
        assert len(layers) == self.n_layers, (len(layers), self.n_layers)
        for i, layer in enumerate(layers):
            ff = getattr(layer, "feed_forward", None) or getattr(layer, "mlp")
            down = getattr(ff, "w2", None) or getattr(ff, "down_proj")
            self.handles.append(down.register_forward_pre_hook(self._hook(i)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def compute(self, layer_idx, x):
        ev = self.evidence[layer_idx]
        x = x.to(torch.float32).unsqueeze(0)        # [1, H]
        ev.get_evidence_weights(x.T)                # [H, 1]
        return float(ev.get_evidence_conflict().item())

    def sample_acf(self):
        """-> (acf_mean[32], acf_decision[32], n_forward_calls)"""
        recs = self.records
        n = len(recs)
        mean = [0.0] * self.n_layers
        for r in recs:
            for i in range(self.n_layers):
                mean[i] += self.compute(i, r[i])
        mean = [v / n for v in mean]
        dec = [self.compute(i, recs[0][i]) for i in range(self.n_layers)]
        return mean, dec, n


# ── Probability baselines ───────────────────────────────────────────────
def prob_metrics(p, ids):
    """p: [V] float32 softmax at the decision position. ids: dict of token ids."""
    p = p.float()
    top = torch.topk(p, 2)
    ent = float(-(p * torch.log(p + 1e-12)).sum())
    pa = float(p[ids["a"]])
    pb = float(p[ids["b"]])
    denom = pa + pb
    return {
        "max_prob": float(top.values[0]),
        "margin": float(top.values[0] - top.values[1]),
        "entropy": ent,
        "logprob_argmax": float(torch.log(top.values[0] + 1e-12)),
        "p_a": pa, "p_b": pb,
        "p_b_renorm": (pb / denom) if denom > 0 else float("nan"),
        "top1_id": int(top.indices[0]),
    }


# ── Split (canonical, verbatim randomness) ──────────────────────────────
def canonical_splits(n_sta=500, n_clean=500):
    rng = np.random.RandomState(SEED)
    i = rng.permutation(n_sta)
    sd, st = set(i[:50].tolist()), set(i[50:].tolist())
    j = rng.permutation(n_clean)
    cd, ct = set(j[:50].tolist()), set(j[50:].tolist())
    return sd, st, cd, ct


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty_like(o, float)
    r[o] = np.arange(len(s)) + 1
    P = int(y.sum())
    N = int(len(y) - P)
    if P == 0 or N == 0:
        return float("nan")
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


# ── Main ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda"
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from modeling_internvl_chat_dino import InternVLChatModelDINO
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True,
                                        use_fast=False, cache_dir=MODEL_CACHE)
    ids = {}
    for c in ("a", "b"):
        enc = tok(c, add_special_tokens=False)["input_ids"]
        ids[c] = enc[0]
    print(f"[info] choice token ids: {ids}", flush=True)

    sta = pd.read_csv(STA_CSV)
    clean = pd.read_csv(CLEAN_CSV).sample(n=500, random_state=SEED).reset_index(drop=True)
    items = ([("STA", 1, int(idx), row) for idx, row in sta.iterrows()] +
             [("clean", 0, int(idx), row) for idx, row in clean.iterrows()])
    _lim = os.environ.get("E1_LIMIT")
    if _lim:
        items = items[:int(_lim)]
        print(f"[info] E1_LIMIT -> {len(items)} items (smoke test)")
    print(f"[info] {len(sta)} STA + {len(clean)} clean = {len(items)} items", flush=True)

    sd, st, cd, ct = canonical_splits()

    print("[info] loading 32 layer weights (.pth) once ...", flush=True)
    _t = time.time()
    collector = LayerACFCollector(WEIGHT_BASE, WEIGHT_PATTERN, N_LAYERS, device)
    print(f"[info] weights loaded in {time.time()-_t:.0f}s", flush=True)

    col = None
    prob_col = None
    for mode in MODES:
        print(f"\n{'='*60}\n  mode = {mode}\n{'='*60}", flush=True)
        _t = time.time()
        model = InternVLChatModelDINO.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, use_flash_attn=True,
            trust_remote_code=True, cache_dir=MODEL_CACHE,
            ignore_mismatched_sizes=True).to(device).eval()
        model.set_dino_mode(mode)
        print(f"  model loaded in {time.time()-_t:.0f}s", flush=True)
        if mode != "native_only":
            pth = os.path.join(PROJECTOR_DIR, f"internvl_dino_projector_{mode}.pth")
            state = torch.load(pth, map_location=device)
            model.dino_projector.load_state_dict(state["dino_projector"])
            if "dino_gate" in state:
                with torch.no_grad():
                    model.dino_gate.fill_(float(state["dino_gate"]))
            print(f"  projector loaded: {pth} "
                  f"(gate={float(model.dino_gate.item()):.4f})", flush=True)

        # verify .pth weights == live w2.weight (once, on native_only)
        if mode == "native_only":
            with torch.no_grad():
                for i in (0, 15, 31):
                    live = model.language_model.model.layers[i].feed_forward.w2.weight
                    ref = torch.load(os.path.join(
                        WEIGHT_BASE, WEIGHT_PATTERN.format(layer=i)),
                        map_location=device)
                    d = (live.float() - ref.to(device).float()).abs().max().item()
                    print(f"  [verify] L{i} live-vs-pth max|d| = {d:.3e}", flush=True)

        collector.attach(model)
        n_vis_native = int(model.num_image_token)

        mode_rows = []
        mode_probs = []
        t0 = time.time()
        for k, (ds, attack, pos, row) in enumerate(items):
            iid = str(row["index"])
            img_dir = STA_IMG if attack == 1 else CLEAN_IMG
            pv = load_image(resolve(img_dir, iid)).to(torch.bfloat16).to(device)
            collector.records = []
            with torch.no_grad():
                resp, all_probs, hidden, _down = model.chat(
                    tok, pv, str(row["question"]).strip(),
                    dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False))
            mean_acf, dec_acf, n_fwd = collector.sample_acf()

            gt = str(row["answer"]).strip().lower()[0]
            if MATCH_OPT:
                _ta, _tb = parse_options(row["question"])
                choice = match_option_text(resp, _ta, _tb)
            else:
                choice = extract_choice(resp)
            correct = int(choice == gt)
            split = "dev" if (pos in sd if attack == 1 else pos in cd) else "eval"
            n_vis = n_vis_native * (2 if mode not in ("native_only", "fused") else 1) * int(pv.shape[0])

            pm = prob_metrics(all_probs[0][0], ids)
            mode_probs.append(dict(
                sample_id=iid, split=split, seed=SEED, model="internvl2_5-8b",
                fusion=mode, n_vision_tokens=n_vis, n_forward_calls=n_fwd,
                clean_or_attack=attack, gt_letter=gt, response=resp.replace("\n", " ")[:60],
                choice=choice if choice else "", correct=correct,
                max_prob=round(pm["max_prob"], 6),
                logprob=round(pm["logprob_argmax"], 6),
                margin=round(pm["margin"], 6), entropy=round(pm["entropy"], 6),
                p_a_renorm=round(pm["p_a"] / (pm["p_a"] + pm["p_b"] + 1e-12), 6),
                p_b_renorm=round(pm["p_b_renorm"], 6)))

            for L in range(N_LAYERS):
                mode_rows.append(dict(
                    sample_id=iid, split=split, seed=SEED, model="internvl2_5-8b",
                    fusion=mode, layer=L, n_vision_tokens=n_vis, n_forward_calls=n_fwd,
                    clean_or_attack=attack, correct=correct,
                    acf=round(mean_acf[L], 8), acf_decision=round(dec_acf[L], 8),
                    max_prob=round(pm["max_prob"], 6),
                    logprob=round(pm["logprob_argmax"], 6),
                    margin=round(pm["margin"], 6), entropy=round(pm["entropy"], 6),
                    p_b_renorm=round(pm["p_b_renorm"], 6)))

            if (k + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  [{mode}] {k+1}/{len(items)}  {el:.0f}s "
                      f"({el/(k+1):.2f}s/it)", flush=True)

        collector.detach()
        if col is None:
            col = list(mode_rows[0].keys())
            prob_col = list(mode_probs[0].keys())
        with open(os.path.join(OUT_DIR, f"e1_sample_layer_{mode}{SUFFIX}.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=col)
            w.writeheader()
            w.writerows(mode_rows)
        with open(os.path.join(OUT_DIR, f"e1_probs_{mode}{SUFFIX}.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=prob_col)
            w.writeheader()
            w.writerows(mode_probs)
        print(f"  [{mode}] wrote {len(mode_rows)} rows / {len(mode_probs)} samples",
              flush=True)

        # ── summary: ACC + per-layer AUROC on the eval pool ──
        import pandas as _pd
        pdf = _pd.DataFrame(mode_probs)
        ev = pdf[pdf.split == "eval"]
        acc_all = pdf.correct.mean()
        acc_eval = ev.correct.mean()
        acc_sta = pdf[pdf.clean_or_attack == 1].correct.mean()
        acc_cl = pdf[pdf.clean_or_attack == 0].correct.mean()
        print(f"  [{mode}] ACC all={acc_all:.4f} eval={acc_eval:.4f} "
              f"STA={acc_sta:.4f} clean={acc_cl:.4f}", flush=True)
        sdf = _pd.DataFrame(mode_rows)
        evs = sdf[sdf.split == "eval"]
        y = (1 - evs[evs.layer == 0].correct.values).astype(int)  # wrong=1
        best = (None, -1)
        line = []
        for L in range(N_LAYERS):
            a = auroc(y, evs[evs.layer == L].acf.values)
            if not np.isnan(a):
                if a < 0.5:
                    a = 1 - a
                if a > best[1]:
                    best = (L, a)
            if L % 4 == 0:
                line.append(f"L{L}:{a:.3f}")
        print(f"  [{mode}] ACF-AUROC(eval) {', '.join(line)}  best={best}", flush=True)
        for pcol in ("max_prob", "margin", "entropy", "p_b_renorm"):
            yy = (1 - ev.correct.values).astype(int)
            a = auroc(yy, ev[pcol].values)
            if not np.isnan(a) and a < 0.5:
                a = 1 - a
            print(f"  [{mode}] prob-AUROC(eval) {pcol}={a:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
