"""Control for the encoder ablation: does the NATIVE tower caption the true subject too?

Under the E1 MCQ prompt the native tower emits a bare option letter, so it can never
"name the subject".  The fused configs instead emit a caption.  So the raw comparison
  interleaved caption-recovery 0.704  vs  native MCQ-accuracy 0.482
is NOT like-for-like.  This run gives the native tower the SAME captioning channel on the
SAME 500 STA images, so the two can be compared.

native_only, prompt = "Describe the image.", greedy, 500 STA.  Writes a csv of raw
responses + per-sample subject recovery (option-(a) head word) and distractor (option-(b)).

Read-only; GPU 0.  Env: internvl.
"""
import os
import sys
import re
import csv

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
PROJ = os.path.join(_IJAR, "results", "interleave")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PROMPT = os.environ.get("CAP_PROMPT", "Describe the image.")
MAX_NEW = int(os.environ.get("CAP_MAX_NEW", "48"))
MODE = os.environ.get("CAP_MODE", "native_only")
OUT = os.path.join(_IJAR, "results", "e3",
                   "native_caption_control.csv" if MODE == "native_only"
                   else f"{MODE}_caption_control.csv")


def head_word(s):
    t = re.findall(r"[a-z]+", str(s).lower())
    return t[-1] if t else ""


def parse_options(q):
    ma = re.search(r"\(a\)\s*(.+?)\s*\(b\)", str(q))
    mb = re.search(r"\(b\)\s*(.+?)\s*($|\(c\))", str(q))
    return (ma.group(1).strip() if ma else "",
            mb.group(1).strip() if mb else "")


def has(resp, w):
    return bool(w) and re.search(rf"\b{re.escape(w)}\b", str(resp).lower()) is not None


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
    if MODE != "native_only":
        sd = torch.load(os.path.join(PROJ, f"internvl_dino_projector_{MODE}.pth"),
                        map_location="cpu")["dino_projector"]
        model.dino_projector.load_state_dict(sd)
        model.dino_projector = model.dino_projector.float()
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    print(f"[info] mode={MODE} prompt={PROMPT!r}", flush=True)

    tf = T.Compose([T.Lambda(lambda im: im.convert("RGB")),
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(MEAN, STD)])
    imgd = os.path.join(_IJAR, "datasets", "STA", "images")
    df = pd.read_csv(os.path.join(_IJAR, "datasets", "STA", "sta_mini.csv"))
    n = int(os.environ.get("CAP_LIMIT", str(len(df))))

    rows = []
    for k in range(n):
        row = df.iloc[k]
        iid = str(row["index"])
        ta, tb = parse_options(row["question"])
        ha, hb = head_word(ta), head_word(tb)
        p = next((os.path.join(imgd, f"{iid}{e}") for e in
                  ("", ".png", ".jpg", ".jpeg")
                  if os.path.exists(os.path.join(imgd, f"{iid}{e}"))), None)
        px = tf(Image.open(p)).unsqueeze(0).to(dev, torch.bfloat16)
        with torch.no_grad():
            resp = model.chat(tok, px, PROMPT,
                              dict(max_new_tokens=MAX_NEW, do_sample=False))[0]
        resp = resp.strip().replace("\n", " ")
        t_ok, d_ok = int(has(resp, ha)), int(has(resp, hb))
        rows.append(dict(sample_id=iid, true_subject=ta, true_head=ha,
                         distractor=tb, dist_head=hb, subj_true=t_ok,
                         subj_dist=d_ok, neither=int(not t_ok and not d_ok),
                         response=resp[:160]))
        if (k + 1) % 25 == 0:
            st = sum(r["subj_true"] for r in rows) / len(rows)
            sd = sum(r["subj_dist"] for r in rows) / len(rows)
            print(f"  [{k+1}/{n}] subj_true={st:.4f} subj_dist={sd:.4f}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    N = len(rows)
    st = sum(r["subj_true"] for r in rows) / N
    sd = sum(r["subj_dist"] for r in rows) / N
    print(f"\n=== {MODE}, prompt={PROMPT!r}, N={N} ===")
    print(f"  subject recovery (true)   = {st:.4f}")
    print(f"  distractor   (attack)     = {sd:.4f}")
    print(f"  neither                   = {1 - st - sd:.4f}")
    print(f"[saved] {OUT}")
    print("DONE")


if __name__ == "__main__":
    main()
