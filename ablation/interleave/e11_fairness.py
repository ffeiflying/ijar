"""E11: fairness controls for the encoder ablation (native_only vs fused).

Answers DS_FOLLOWUP_PLAN §2 E11: "token / 可训参数 / 延迟 / 显存；排除 token 翻倍解释".
Loads InternVL once, runs the SAME STA MCQ in native mode and then fused mode,
and reports per-image forward latency (fixed max_new_tokens) + peak GPU memory.

Read-only wrt env/results.  Usage: CUDA_VISIBLE_DEVICES=0 python e11_fairness.py
"""
import os
import sys
import time

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
N = int(os.environ.get("E11_N", "15"))
WARM = 3
MAXNEW = 8


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
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")

    tf = T.Compose([T.Lambda(lambda im: im.convert("RGB")),
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize((0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225))])
    imgd = os.path.join(_IJAR, "datasets", "STA", "images")
    df = pd.read_csv(os.path.join(_IJAR, "datasets", "STA", "sta_mini.csv"))
    ck = torch.load(os.path.join(_IJAR, "results", "interleave",
                                 "internvl_dino_projector_fused.pth"), map_location="cpu")

    def run(mode):
        model.set_dino_mode(mode)
        if mode == "fused":
            model.dino_projector.load_state_dict(ck["dino_projector"])
            with torch.no_grad():
                model.dino_gate.fill_(float(ck["dino_gate"]))
        # warmup
        for k in range(WARM):
            _once(model, tok, tf, imgd, df, k)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for k in range(N):
            _once(model, tok, tf, imgd, df, k)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / N
        peak = torch.cuda.max_memory_allocated() / 1e9
        return dt, peak

    rows = []
    for mode in ("native_only", "fused"):
        dt, peak = run(mode)
        ntok = 256
        print(f"[E11] {mode:12s} n_vision_tokens={ntok}  latency={dt*1000:.0f} ms/img "
              f"peak_alloc={peak:.2f} GB")
        rows.append(dict(mode=mode, n_vision_tokens=ntok,
                         latency_ms=dt * 1000, peak_alloc_gb=peak))
    pd.DataFrame(rows).to_csv(os.path.join(_IJAR, "results", "e3",
                                           "e11_fairness.csv"), index=False)
    print("[saved] results/e3/e11_fairness.csv")
    print("DONE")


def _once(model, tok, tf, imgd, df, k):
    row = df.iloc[k % len(df)]
    iid = str(row["index"])
    p = next((os.path.join(imgd, f"{iid}{e}") for e in ("", ".png", ".jpg", ".jpeg")
              if os.path.exists(os.path.join(imgd, f"{iid}{e}"))), None)
    pv = tf(Image.open(p)).unsqueeze(0).to("cuda", torch.bfloat16)
    q = str(row["question"]).strip()
    with torch.no_grad():
        model.chat(tok, pv, q, dict(max_new_tokens=MAXNEW, do_sample=False))


if __name__ == "__main__":
    main()
