"""Diagnose the degenerate DINO-fusion output in the `internvl` env.

Checks, for concat mode on 2 STA samples:
  - |img_ctx_emb| vs |dino_residual| magnitudes
  - greedy response vs do_sample=True temperature=1.0 response (historical setting)
  - response when dino_residual is zeroed (projector last layer = 0)
Read-only; GPU 0. Env: internvl
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

import torch  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from torchvision.transforms.functional import InterpolationMode  # noqa: E402

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
CACHE = "/data/public/models/InternVL/InternVL2_5-8B"


def tf(ss=448):
    return T.Compose([
        T.Lambda(lambda im: im.convert('RGB') if im.mode != 'RGB' else im),
        T.Resize((ss, ss), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


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
    mode = "concat"
    model.set_dino_mode(mode)
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    pth = os.path.join(_IJAR, "results", "interleave",
                       f"internvl_dino_projector_{mode}.pth")
    st = torch.load(pth, map_location=dev)
    print("ckpt keys:", list(st.keys()))
    sd = st["dino_projector"]
    print("proj keys:", list(sd.keys()))
    for k, v in sd.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype} absmean={v.abs().mean():.5f} "
              f"absmax={v.abs().max():.5f}")
    model.dino_projector.load_state_dict(sd)
    print("loaded (params now dtype:",
          next(model.dino_projector.parameters()).dtype, ")")

    df = pd.read_csv(os.path.join(_IJAR, "datasets", "STA", "sta_mini.csv"))
    imgd = os.path.join(_IJAR, "datasets", "STA", "images")

    for k in range(2):
        row = df.iloc[k]
        iid = str(row["index"])
        p = None
        for e in ("", ".png", ".jpg", ".jpeg"):
            c = os.path.join(imgd, f"{iid}{e}")
            if os.path.exists(c):
                p = c
                break
        pv = tf()(Image.open(p).convert("RGB")).unsqueeze(0).to(torch.bfloat16).to(dev)
        q = str(row["question"]).strip()

        # --- magnitudes of img_ctx_emb vs dino_residual ---
        model._init_dino(dev)
        with torch.no_grad():
            dino_feats = model.extract_dino_feature(pv)   # [1,N,C]
            ic = model.language_model.get_input_embeddings()(
                torch.tensor([model.img_context_token_id], device=dev))
            print(f"\n## {iid}")
            print(f"  img_ctx_emb |.|mean={ic.abs().mean():.5f} "
                  f"max={ic.abs().max():.5f}")
            print(f"  dino_embeds(from extract) |.|mean={dino_feats.abs().mean():.5f} "
                  f"max={dino_feats.abs().max():.5f}  shape={tuple(dino_feats.shape)}")
            resid = dino_feats.float() - ic.float().expand_as(dino_feats.float())
            print(f"  dino_residual |.|mean={resid.abs().mean():.5f} "
                  f"max={resid.abs().max():.5f}")

        for tag, gc in (("greedy", dict(max_new_tokens=24, do_sample=False)),
                        ("sample", dict(max_new_tokens=24, do_sample=True,
                                        temperature=1.0))):
            with torch.no_grad():
                out = model.chat(tok, pv, q, gc)
            print(f"  [{tag}] {out[0][:110]!r}")

        # zero the projector last layer -> pure native context tokens (2x)
        with torch.no_grad():
            model.dino_projector[-1].weight.zero_()
            model.dino_projector[-1].bias.zero_()
            out = model.chat(tok, pv, q, dict(max_new_tokens=24, do_sample=False))
        print(f"  [zeroed-proj] {out[0][:110]!r}")
        # restore
        model.dino_projector.load_state_dict(sd)
    print("\nDONE")


if __name__ == "__main__":
    main()
