"""Is the DINO fusion BROKEN, or just out-of-domain?

train_dino_projector.py trains the projector on ImageNet-val (STA images excluded)
with the template  `<image>\n{class_name}`.  The earlier probe_dino_scale.py test used
an STA image, which is out of that distribution -- so its loss (9.34) is NOT comparable
to the recorded training loss (1.2) and does not prove a train/infer mismatch.

This probe separates the two hypotheses:
  A) IN-DOMAIN : ImageNet val image + its class name, template as in training
                 -> forward loss vs the recorded ~1.2, and chat coherence
  B) STA (OOD) : few STA samples in concat/interleaved
                 -> raw response, first letter, and what extract_choice would grab

Read-only; GPU 0.  Env: internvl
"""
import os
import sys
import re

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

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMAGENET_VAL = "/data/public/datasets/ImageNet-val/val"
PROJ = os.path.join(_IJAR, "results", "interleave")


def extract_choice(resp):
    """mirror run_e1_e2_internvl.py: scrape a standalone a/b/c/d letter."""
    r = resp.strip().lower()
    m = re.search(r"\(([abcd])\)", r)
    if m:
        return m.group(1)
    for ch in r:
        if ch in "abcd":
            return ch
    return ""


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
    model._init_dino(dev)

    def load_px(path):
        im = Image.open(path).convert("RGB").resize((448, 448))
        t = T.Normalize(MEAN, STD)(T.ToTensor()(im)).unsqueeze(0)
        return t.to(dev, torch.bfloat16)

    def set_mode(mode):
        model.set_dino_mode(mode)
        if mode != "native_only":
            sd = torch.load(os.path.join(PROJ, f"internvl_dino_projector_{mode}.pth"),
                            map_location="cpu")["dino_projector"]
            model.dino_projector.load_state_dict(sd)
            model.dino_projector = model.dino_projector.float()
        model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")

    def fwd_loss(px, target):
        img_tok = model.num_image_token * (2 if model.dino_mode != "native_only" else 1)
        text = (f"<image>\n{target}").replace(
            "<image>", '<img>' + '<IMG_CONTEXT>' * img_tok + '</img>', 1)
        ids = tok(text, return_tensors='pt')['input_ids'].to(dev)
        lab = ids.clone()
        for t in ("<IMG_CONTEXT>", "<img>", "</img>"):
            lab[lab == tok.convert_tokens_to_ids(t)] = -100
        with torch.no_grad():
            o = model(pixel_values=px.to(dev, torch.bfloat16), input_ids=ids,
                      attention_mask=torch.ones_like(ids),
                      image_flags=torch.ones(1, 1, dtype=torch.long, device=dev),
                      labels=lab, use_cache=False)
        return float(o.loss)

    # ---------------- A) in-domain: ImageNet val + class name ----------------
    print("=== A) IN-DOMAIN (ImageNet val, template `<image>\\n{class_name}`) ===")
    print("    recorded training loss: concat ~1.2 (3000 steps, 2000 imgs, 3 ep)\n")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ic", "/data/liuxiaofei/uncertainty/Defense-Prefix-main/"
              "datasets/configs/imagenet_config.py")
    ic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ic)
    folders = sorted(os.listdir(IMAGENET_VAL))
    for ci in range(3):
        cname = ic.classes[ci]
        d = os.path.join(IMAGENET_VAL, folders[ci])
        f = sorted(os.listdir(d))[0]
        px = load_px(os.path.join(d, f))
        out = [f"  {cname:<22}"]
        for mode in ("native_only", "concat", "interleaved"):
            set_mode(mode)
            L = fwd_loss(px, cname)
            a = model.chat(tok, px, "What is in this image?", dict(
                max_new_tokens=20, do_sample=False))[0].strip().replace("\n", " ")
            out.append(f"| {mode[:5]} loss={L:6.3f} chat={a[:42]!r}")
        print("".join(out))

    # ---------------- B) STA (out of distribution) ----------------
    print("\n=== B) STA (OUT of training distribution) ===")
    df = pd.read_csv(os.path.join(_IJAR, "datasets", "STA", "sta_mini.csv"))
    imgd = os.path.join(_IJAR, "datasets", "STA", "images")
    for k in range(3):
        row = df.iloc[k]
        iid = str(row["index"])
        q = str(row["question"]).strip()
        p = None
        for e in ("", ".png", ".jpg", ".jpeg"):
            c = os.path.join(imgd, f"{iid}{e}")
            if os.path.exists(c):
                p = c
                break
        px = load_px(p)
        for mode in ("native_only", "concat", "interleaved"):
            set_mode(mode)
            a = model.chat(tok, px, q, dict(
                max_new_tokens=48, do_sample=False))[0]
            ch = extract_choice(a)
            print(f"  {iid[:22]:<22} {mode[:5]:<5} choice={ch!r} "
                  f"resp={a.strip().replace(chr(10),' ')[:64]!r}")
    print("\nDONE")


if __name__ == "__main__":
    main()
