"""Differential test of the degenerate DINO fusion (concat mode).

1) training-style model.forward(...) loss  (does the trained path still work?)
2) residual scale sweep a in extract_dino_feature: img_ctx + a*residual
   -> forward loss and greedy chat response for each a
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

import types  # noqa: E402
import torch  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
import torchvision.transforms as T  # noqa: E402

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def main():
    from modeling_internvl_chat_dino import InternVLChatModelDINO
    from transformers import AutoTokenizer

    dev = "cuda"
    model = InternVLChatModelDINO.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, use_flash_attn=True,
        trust_remote_code=True, cache_dir=CACHE,
        ignore_mismatched_sizes=True)
    model = model.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True,
                                        use_fast=False, cache_dir=CACHE)
    mode = "concat"
    model.set_dino_mode(mode)
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    sd = torch.load(os.path.join(_IJAR, "results", "interleave",
                                 f"internvl_dino_projector_{mode}.pth"),
                    map_location=dev)["dino_projector"]
    model.dino_projector.load_state_dict(sd)
    model._init_dino(dev)

    # wrap extract_dino_feature with a residual scale
    orig = model.extract_dino_feature

    def make_scaled(a):
        def f(pixel_values):
            px_dtype = pixel_values.dtype
            px = pixel_values.float()
            if px.shape[-1] != 336 or px.shape[-2] != 336:
                px = torch.nn.functional.interpolate(
                    px, size=(336, 336), mode='bilinear', align_corners=False)
            B = px.shape[0]
            out = model.dino_encoder.forward_features(px)
            feats = out['x_norm_patchtokens']
            nd = feats.shape[1]
            gd = int(nd ** 0.5)
            tg = int(model.num_image_token ** 0.5)
            if gd != tg:
                feats = feats.reshape(B, gd, gd, -1).permute(0, 3, 1, 2)
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, (tg, tg))
                feats = feats.permute(0, 2, 3, 1).reshape(B, tg * tg, -1)
            resid = model.dino_projector(feats)
            ic = model.language_model.get_input_embeddings()(
                torch.tensor([model.img_context_token_id], device=resid.device))
            emb = ic.unsqueeze(0).expand(resid.shape[0], resid.shape[1], -1) \
                + a * resid
            return emb.to(px_dtype)
        return f

    df = pd.read_csv(os.path.join(_IJAR, "datasets", "STA", "sta_mini.csv"))
    imgd = os.path.join(_IJAR, "datasets", "STA", "images")

    def load_px(iid, size=448):
        p = None
        for e in ("", ".png", ".jpg", ".jpeg"):
            c = os.path.join(imgd, f"{iid}{e}")
            if os.path.exists(c):
                p = c
                break
        im = Image.open(p).convert("RGB").resize((size, size))
        t = T.ToTensor()(im)
        t = T.Normalize(MEAN, STD)(t)
        return torch.stack([t])

    for k in range(2):
        row = df.iloc[k]
        iid = str(row["index"])
        q = str(row["question"]).strip()
        ans = str(row["answer"]).strip()
        print(f"\n## {iid}")

        for a in (1.0, 0.1, 0.02, 0.0):
            model.extract_dino_feature = types.MethodType(
                lambda self, pv, _a=a: make_scaled(_a)(pv), model)
            # --- training-style forward loss ---
            px = load_px(iid).to(dev, dtype=torch.bfloat16)
            img_token_count = model.num_image_token * 2
            text = (f"<image>\n{q} {ans}").replace(
                "<image>", '<img>' + '<IMG_CONTEXT>' * img_token_count + '</img>', 1)
            tk = tok(text, return_tensors='pt')
            ids = tk['input_ids'].to(dev)
            lab = ids.clone()
            lab[lab == model.img_context_token_id] = -100
            lab[lab == tok.convert_tokens_to_ids('<img>')] = -100
            lab[lab == tok.convert_tokens_to_ids('</img>')] = -100
            with torch.no_grad():
                o = model(pixel_values=px, input_ids=ids,
                          attention_mask=torch.ones_like(ids),
                          image_flags=torch.ones(1, 1, dtype=torch.long, device=dev),
                          labels=lab, use_cache=False)
            loss = float(o.loss)
            with torch.no_grad():
                out = model.chat(tok, px, q, dict(max_new_tokens=24, do_sample=False))
            print(f"  a={a:<5} fwd_loss={loss:7.4f}  chat={out[0][:70]!r}")
    print("\nDONE")


if __name__ == "__main__":
    main()
