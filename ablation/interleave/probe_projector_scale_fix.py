"""Diagnose: after training the DINO projector, the fused model should answer the
MCQ (a/b) roughly like native — but it captions instead.  Is that a SCALE defect
(the projector is too loud) or a CONTENT defect (the projector encodes "caption")?

Method: keep the trained projector direction, sweep only its OUTPUT magnitude:
    dino_token = img_ctx_emb + alpha * dino_residual
and measure strict-choice MCQ accuracy vs alpha.  alpha=0 -> DINO silent (pure
native behaviour);  alpha=1 -> the trained fusion;  alpha_native -> residual
magnitude matched to |native mlp1 output|.

Reference scales printed once: token-embedding table, <IMG_CONTEXT> embedding,
and |native vit_embeds| (what the LLM normally sees at a vision position).

Read-only w.r.t. files (monkeypatches the projector at runtime).  Env: internvl.
"""
import os
import re
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

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from torchvision.transforms.functional import InterpolationMode  # noqa: E402

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
MODE = os.environ.get("PSF_MODE", "interleaved")
N = int(os.environ.get("PSF_N", "20"))
ALPHAS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class ScaledProj(nn.Module):
    def __init__(self, proj):
        super().__init__()
        self.proj = proj
        self.alpha = 1.0

    def forward(self, x):
        return self.proj(x) * self.alpha


def strict_choice(resp):
    m = re.fullmatch(r"\(?([abcdABCD])\)?[\.\,\:\;\)\]]?", str(resp).strip())
    return m.group(1).lower() if m else None


def resolve(imgd, iid):
    for e in ("", ".png", ".jpg", ".jpeg"):
        p = os.path.join(imgd, f"{iid}{e}")
        if os.path.exists(p):
            return p
    return None


def main():
    from modeling_internvl_chat_dino import InternVLChatModelDINO
    from transformers import AutoTokenizer

    dev = "cuda"
    model = InternVLChatModelDINO.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, use_flash_attn=True,
        trust_remote_code=True, cache_dir=CACHE,
        ignore_mismatched_sizes=True).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True,
                                        use_fast=False, cache_dir=CACHE)
    model.set_dino_mode(MODE)
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.dino_projector.load_state_dict(
        torch.load(os.path.join(_IJAR, "results", "interleave",
                                f"internvl_dino_projector_{MODE}.pth"),
                   map_location="cpu")["dino_projector"])
    model._init_dino(dev)
    model.dino_projector = ScaledProj(model.dino_projector)

    emb = model.language_model.get_input_embeddings()
    with torch.no_grad():
        img_ctx = emb(torch.tensor([model.img_context_token_id], device=dev))

    tf = T.Compose([T.Lambda(lambda im: im.convert("RGB")),
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(MEAN, STD)])
    imgd = os.path.join(_IJAR, "datasets", "STA", "images")
    df = pd.read_csv(os.path.join(_IJAR, "datasets", "STA", "sta_mini.csv"))

    print(f"mode={MODE}  N={N}")
    print(f"  token_embed_table |.|mean={emb.weight.abs().mean().item():.5f}")
    print(f"  <IMG_CONTEXT>     |.|mean={img_ctx.abs().mean().item():.5f}")
    print(f"  alphas = {ALPHAS}\n")

    acc = {a: [] for a in ALPHAS}
    nat_acc = []
    ratios, nat_abs_l, res_abs_l = [], [], []

    for k in range(N):
        row = df.iloc[k]
        iid = str(row["index"])
        p = resolve(imgd, iid)
        if p is None:
            continue
        pv = tf(Image.open(p)).unsqueeze(0).to(dev, torch.bfloat16)
        q = str(row["question"]).strip()
        gt = str(row["answer"]).strip().lower()[0]

        model.dino_projector.alpha = 1.0                      # measure at trained scale
        with torch.no_grad():
            nat = model.extract_feature(pv)
            resid = model.extract_dino_feature(pv).float() - img_ctx.float()
        nat_abs, res_abs = nat.abs().mean().item(), resid.abs().mean().item()
        nat_abs_l.append(nat_abs); res_abs_l.append(res_abs)
        ratios.append(res_abs / nat_abs if nat_abs else float("nan"))

        with torch.no_grad():
            r0 = model.chat(tok, pv, q, dict(max_new_tokens=16, do_sample=False))[0]
        nat_acc.append(int(strict_choice(r0) == gt))

        line = [f"{iid:>12}  nat={nat_abs:.3f} res={res_abs:7.3f} "
                f"({res_abs/nat_abs:5.1f}x nat)"]
        for a in ALPHAS:
            model.dino_projector.alpha = a
            with torch.no_grad():
                resp = model.chat(tok, pv, q,
                                  dict(max_new_tokens=16, do_sample=False))[0]
            sc = strict_choice(resp)
            acc[a].append(int(sc == gt))
            line.append(f"a={a:g}:{sc or '-'}")
        print("  ".join(line), flush=True)
        if (k + 1) % 5 == 0:
            print(f"  .. {k+1}/{N}  native_acc={np.mean(nat_acc):.3f} "
                  f"mean_ratio={np.nanmean(ratios):.1f}x", flush=True)

    print("\n=== strict-choice MCQ accuracy vs residual scale ===")
    print(f"  native_only (equivalent, alpha=0 row)     ACC={np.mean(nat_acc):.3f}")
    for a in ALPHAS:
        print(f"  alpha={a:<7g} (resid ~{a*np.nanmean(ratios):6.2f}x native)  "
              f"ACC={np.mean(acc[a]):.3f}")
    print(f"\n  |native vit_embeds| mean={np.mean(nat_abs_l):.4f}   "
          f"|dino_residual| mean={np.mean(res_abs_l):.4f}   "
          f"ratio={np.nanmean(ratios):.1f}x")
    print("DONE")


if __name__ == "__main__":
    main()
