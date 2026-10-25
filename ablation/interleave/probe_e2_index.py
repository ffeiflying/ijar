"""Probe: which index of all_probs / llm_head_hidden_states is the DECISION token?

Runs 3 STA samples, prints for each list entry the argmax token string + prob, and
the response. Read-only; uses GPU 0. Env: internvl
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
from PIL import Image  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from torchvision.transforms.functional import InterpolationMode  # noqa: E402


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_image(path, input_size=448):
    img = Image.open(path).convert('RGB')
    return build_transform(input_size)(img).unsqueeze(0)


def main():
    from modeling_internvl_chat_dino import InternVLChatModelDINO
    from transformers import AutoTokenizer

    model = InternVLChatModelDINO.from_pretrained(
        "OpenGVLab/InternVL2_5-8B", torch_dtype=torch.bfloat16, use_flash_attn=True,
        trust_remote_code=True, cache_dir="/data/public/models/InternVL/InternVL2_5-8B",
        ignore_mismatched_sizes=True).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL2_5-8B", trust_remote_code=True, use_fast=False,
        cache_dir="/data/public/models/InternVL/InternVL2_5-8B")
    model.set_dino_mode("native_only")

    import pandas as pd
    df = pd.read_csv(os.path.join(_IJAR, "datasets", "STA", "sta_mini.csv"))
    imgdir = os.path.join(_IJAR, "datasets", "STA", "images")

    for k in range(3):
        row = df.iloc[k]
        iid = str(row["index"])
        p = None
        for ext in ("", ".png", ".jpg", ".jpeg"):
            c = os.path.join(imgdir, f"{iid}{ext}")
            if os.path.exists(c):
                p = c
                break
        pv = load_image(p).to(torch.bfloat16).to("cuda")
        out = model.chat(tok, pv, str(row["question"]).strip(),
                         dict(max_new_tokens=8, do_sample=False))
        resp, probs, hidden, down = out
        print(f"\n### sample {iid}  response={resp[:40]!r}")
        print(f"  all_probs len={len(probs)}  hidden len={len(hidden)}  down len={len(down)}")
        for i, pr in enumerate(probs):
            t = torch.topk(pr[0].float(), 3)
            toks = [tok.decode([j]) for j in t.indices.tolist()]
            print(f"  probs[{i}] shape={tuple(pr.shape)} top3={list(zip(toks,[round(v,4) for v in t.values.tolist()]))}")
        for i, h in enumerate(hidden):
            with torch.no_grad():
                lg = model.lm_head(h.to(model.lm_head.weight.dtype))[0].float()
            j = int(lg.argmax())
            print(f"  hidden[{i}] shape={tuple(h.shape)} argmax_tok={tok.decode([j])!r} "
                  f"prob={torch.softmax(lg,-1).max().item():.4f}")
        # which down entry shape
        print(f"  down[0]={tuple(down[0].shape)}  down[-1]={tuple(down[-1].shape)}")
    print("\nDONE")


if __name__ == "__main__":
    main()
