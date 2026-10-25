"""Smoke test for the new native-anchored "fused" mode.

Loads InternVL, sets dino_mode="fused", drops in the EXISTING interleaved projector
as a stand-in (no retrain yet), gate=0.01, and runs the STA MCQ on a few samples.

Expectation: fused = native + gate*residual on the SAME 256 tokens (no doubling),
so the model should still answer a/b (unlike the 512-token concat/interleaved).
Read-only.  Env: internvl.
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from torchvision.transforms.functional import InterpolationMode  # noqa: E402

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
SRC = os.environ.get("SMOKE_SRC", "interleaved")   # which trained projector to borrow
GATE = float(os.environ.get("SMOKE_GATE", "0.01"))
N = int(os.environ.get("SMOKE_N", "8"))


def strict_choice(resp):
    m = re.fullmatch(r"\(?([abcdABCD])\)?[\.\,\:\;\)\]]?", str(resp).strip())
    return m.group(1).lower() if m else None


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
    model.set_dino_mode("fused")
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    ck = torch.load(os.path.join(_IJAR, "results", "interleave",
                                 f"internvl_dino_projector_{SRC}.pth"),
                    map_location="cpu")
    model.dino_projector.load_state_dict(ck["dino_projector"])
    gate = float(os.environ.get("SMOKE_GATE", ck.get("dino_gate", GATE)))
    with torch.no_grad():
        model.dino_gate.fill_(gate)
    print(f"fused mode: borrowed {SRC} projector, gate={float(model.dino_gate.item()):.4f}")

    tf = T.Compose([T.Lambda(lambda im: im.convert("RGB")),
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize((0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225))])
    imgd = os.path.join(_IJAR, "datasets", "STA", "images")
    df = pd.read_csv(os.path.join(_IJAR, "datasets", "STA", "sta_mini.csv"))

    ok = []
    for k in range(N):
        row = df.iloc[k]
        iid = str(row["index"])
        p = next((os.path.join(imgd, f"{iid}{e}") for e in
                  ("", ".png", ".jpg", ".jpeg")
                  if os.path.exists(os.path.join(imgd, f"{iid}{e}"))), None)
        pv = tf(Image.open(p)).unsqueeze(0).to(dev, torch.bfloat16)
        q = str(row["question"]).strip()
        gt = str(row["answer"]).strip().lower()[0]
        with torch.no_grad():
            resp = model.chat(tok, pv, q, dict(max_new_tokens=16, do_sample=False))[0]
        sc = strict_choice(resp)
        ok.append(int(sc == gt))
        print(f"  {iid:>12}  gt={gt}  resp={resp.strip()[:50]!r}  choice={sc}")
    print(f"\nstrict ACC over {N} = {np.mean(ok):.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
