"""Fair re-test of the DINO fusion, using the ACTUAL training recipe.

train_dino_projector.py trains the projector on ImageNet captions drawn from
TEMPLATES = ["This is a photo of a {class}.", "The image shows a {class}.",
             "A {class} is in the picture.", "a photo of a {class}."]
with STA images excluded.  So:

  * the fair IN-DOMAIN loss uses a caption template, not the bare class name;
  * the fair STA behaviour is "does the fused model name the TRUE subject",
    since captioning (not MCQ answering) is what it was trained to do.

For STA the option-(a) text IS the true subject (GT is always 'a'), so we can
score subject recovery directly, and compare against the MCQ readout the runner
currently uses.

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
CAP_TEMPLATE = "A {c} is in the picture."
MODES = ("native_only", "concat", "interleaved")


def extract_choice(resp):
    r = resp.strip().lower()
    m = re.search(r"\(([abcd])\)", r)
    if m:
        return m.group(1)
    for ch in r:
        if ch in "abcd":
            return ch
    return ""


def option_a(q):
    m = re.search(r"\(a\)\s*(.+?)\s*\(b\)", q)
    return m.group(1).strip() if m else ""


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
        n = model.num_image_token * (2 if model.dino_mode != "native_only" else 1)
        text = (f"<image>\n{target}").replace(
            "<image>", '<img>' + '<IMG_CONTEXT>' * n + '</img>', 1)
        ids = tok(text, return_tensors='pt')['input_ids'].to(dev)
        lab = ids.clone()
        for t in ("<IMG_CONTEXT>", "<img>", "</img>"):
            lab[lab == tok.convert_tokens_to_ids(t)] = -100
        with torch.no_grad():
            o = model(pixel_values=px, input_ids=ids,
                      attention_mask=torch.ones_like(ids),
                      image_flags=torch.ones(1, 1, dtype=torch.long, device=dev),
                      labels=lab, use_cache=False)
        return float(o.loss)

    def ask(px, q, n=24):
        return model.chat(tok, px, q, dict(max_new_tokens=n,
                                           do_sample=False))[0].strip()

    # ------- A) in-domain, CAPTION template (the training target) -------
    print("=== A) IN-DOMAIN caption loss, target = 'A {class} is in the picture.' ===")
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
        px = load_px(os.path.join(d, sorted(os.listdir(d))[0]))
        row = [f"  {cname:<20}"]
        for mode in MODES:
            set_mode(mode)
            L = fwd_loss(px, CAP_TEMPLATE.format(c=cname))
            a = ask(px, "Describe the image.", 16).replace("\n", " ")
            row.append(f"| {mode[:5]} L={L:5.2f} {a[:34]!r}")
        print("".join(row))

    # ------- B) STA: caption vs MCQ, and subject recovery -------
    print("\n=== B) STA: true subject = option-(a) text ===")
    df = pd.read_csv(os.path.join(_IJAR, "datasets", "STA", "sta_mini.csv"))
    imgd = os.path.join(_IJAR, "datasets", "STA", "images")
    n_cap_ok = {m: 0 for m in MODES}
    n_mcq_ok = {m: 0 for m in MODES}
    N = 12
    for k in range(N):
        row = df.iloc[k]
        iid = str(row["index"])
        q = str(row["question"]).strip()
        subj = option_a(q).lower()
        p = next((os.path.join(imgd, f"{iid}{e}") for e in ("", ".png", ".jpg", ".jpeg")
                  if os.path.exists(os.path.join(imgd, f"{iid}{e}"))), None)
        px = load_px(p)
        print(f"  {iid[:20]:<20} true_subject={subj!r}")
        for mode in MODES:
            set_mode(mode)
            cap = ask(px, "Describe the image.", 20).replace("\n", " ")
            mcq_raw = ask(px, q, 24)
            c = extract_choice(mcq_raw)
            cap_ok = subj.split()[-1] in cap.lower() if subj else False
            mcq_ok = (c == "a")
            n_cap_ok[mode] += int(cap_ok)
            n_mcq_ok[mode] += int(mcq_ok)
            print(f"      {mode[:5]:<5} cap_ok={int(cap_ok)} {cap[:44]!r}")
            print(f"            mcq choice={c!r} mcq_ok={int(mcq_ok)} "
                  f"{mcq_raw.strip()[:40]!r}")
    print(f"\n=== over {N} STA samples ===")
    for mode in MODES:
        print(f"  {mode:<12} caption_recovery={n_cap_ok[mode]}/{N}   "
              f"mcq_acc={n_mcq_ok[mode]}/{N}")
    print("\nDONE")


if __name__ == "__main__":
    main()
