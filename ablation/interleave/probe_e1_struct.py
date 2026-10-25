"""Probe: inspect the InternVL-DINO chat() return structure before writing E1/E2.

Read-only: loads the model on one GPU, runs ONE STA sample, prints the shapes of
`all_probs`, `llm_head_hidden_states` and `all_input_down_proj`, plus whether all
32 layers are recoverable for the ACF and whether logits can be rebuilt from the
lm_head hidden states.

Env: internvl  (has flash_attn + the patched transformers that returns the 4-tuple)
Usage: CUDA_VISIBLE_DEVICES=1 python probe_e1_struct.py
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from torchvision.transforms.functional import InterpolationMode  # noqa: E402

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
MODEL_CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
STA_CSV = os.path.join(_IJAR, "datasets", "STA", "sta_mini.csv")
STA_IMG = os.path.join(_IJAR, "datasets", "STA", "images")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def resolve(image_id):
    for ext in ("", ".png", ".jpg", ".jpeg"):
        p = os.path.join(STA_IMG, f"{image_id}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(image_id)


def describe(name, obj, depth=0):
    pad = "  " * depth
    if torch.is_tensor(obj):
        print(f"{pad}{name}: Tensor shape={tuple(obj.shape)} dtype={obj.dtype}")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__} len={len(obj)}")
        if len(obj):
            describe(f"{name}[0]", obj[0], depth + 1)
    elif isinstance(obj, dict):
        print(f"{pad}{name}: dict keys={list(obj.keys())[:10]}")
    else:
        print(f"{pad}{name}: {type(obj).__name__} = {str(obj)[:80]}")


def main():
    from modeling_internvl_chat_dino import InternVLChatModelDINO
    from transformers import AutoTokenizer

    dev = "cuda"
    print("loading model ...", flush=True)
    model = InternVLChatModelDINO.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, use_flash_attn=True,
        trust_remote_code=True, cache_dir=MODEL_CACHE,
        ignore_mismatched_sizes=True)
    model = model.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True,
                                        use_fast=False, cache_dir=MODEL_CACHE)
    model.set_dino_mode("native_only")
    print("native_only set", flush=True)

    df = pd.read_csv(STA_CSV)
    row = df.iloc[0]
    image_id = str(row["index"])
    pv = load_image(resolve(image_id)).to(torch.bfloat16).to(dev)
    print(f"sample {image_id} pixel_values={tuple(pv.shape)}", flush=True)

    torch.manual_seed(42)
    out = model.chat(tok, pv, str(row["question"]).strip(),
                     dict(max_new_tokens=16, do_sample=False))
    response, all_probs, hidden, down = out
    print(f"\nresponse={response[:90]!r}\n")

    print("== all_probs =="); describe("all_probs", all_probs)
    print("== llm_head_hidden_states =="); describe("hidden", hidden)
    print("== all_input_down_proj =="); describe("down", down)

    # ---- can we rebuild logits from the lm_head hidden states? ----
    print("\n== lm_head rebuild ==")
    try:
        lmh = model.lm_head
        h = hidden
        if isinstance(h, (list, tuple)):
            h = h[-1]
        print(f"  hidden used: shape={tuple(h.shape)} dtype={h.dtype}")
        with torch.no_grad():
            logits = lmh(h.to(lmh.weight.dtype))
        print(f"  logits shape={tuple(logits.shape)}")
        if logits.dim() == 3:
            lg = logits[0, -1].float()
        else:
            lg = logits[-1].float()
        p = torch.softmax(lg, dim=-1)
        top = torch.topk(p, 3)
        print(f"  top3 prob={[round(v,4) for v in top.values.tolist()]} "
              f"idx={top.indices.tolist()}")
        print(f"  max_prob={p.max().item():.4f}  "
              f"entropy={float(-(p*torch.log(p+1e-12)).sum()):.4f}  "
              f"margin={float(top.values[0]-top.values[1]):.4f}")
    except Exception as e:
        print(f"  lm_head rebuild FAILED: {type(e).__name__}: {e}")

    # ---- ACF across all 32 layers from `down` ----
    print("\n== per-layer ACF from all_input_down_proj ==")
    from envidence import EvidenceModel
    W_BASE = "/data/liuxiaofei/uncertainty/InternVL/weight"
    try:
        ntok = len(down)
        first = down[0]
        print(f"  #generated tokens (outer len) = {ntok}; "
              f"inner len = {len(first) if hasattr(first,'__len__') else 'n/a'}")
        for probe_l in (0, 15, 31):
            w = torch.load(os.path.join(
                W_BASE, f"internvl_MLP_Layer{probe_l}_down_proj.pth"),
                map_location=dev)
            ev = EvidenceModel(w.to(dev))
            tot = 0.0
            for step in down:
                x = step[probe_l]
                if x.dim() == 3:
                    x = x[:, -1:, :]
                elif x.dim() == 2:
                    x = x[-1:, :]
                x = x.squeeze(0)
                if x.dim() == 2 and x.size(0) > 1:
                    x = x[-1, :].unsqueeze(0)
                ev.get_evidence_weights(x.T)
                tot += ev.get_evidence_conflict().item()
            print(f"  L{probe_l}: mean ACF over {ntok} tokens = {tot/ntok:.6f}")
    except Exception as e:
        import traceback
        print(f"  ACF FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

    # ---- greedy determinism check ----
    print("\n== greedy repeatability ==")
    torch.manual_seed(42)
    o2 = model.chat(tok, pv, str(row["question"]).strip(),
                    dict(max_new_tokens=16, do_sample=False))
    print(f"  run2 response == run1: {o2[0] == response}  ({o2[0][:60]!r})")
    print("\nDONE")


if __name__ == "__main__":
    main()
