"""Layer-by-layer magnitude decomposition of the DINO projector. GPU 0, env internvl."""
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

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def mag(t):
    t = t.float()
    return f"mean={t.mean():+.4f} std={t.std():.4f} absmean={t.abs().mean():.4f} absmax={t.abs().max():.3f}"


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
    model.set_dino_mode("concat")
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    sd = torch.load(os.path.join(_IJAR, "results", "interleave",
                                 "internvl_dino_projector_concat.pth"),
                    map_location="cpu")["dino_projector"]
    model.dino_projector.load_state_dict(sd)
    model._init_dino(dev)

    df = pd.read_csv(os.path.join(_IJAR, "datasets", "STA", "sta_mini.csv"))
    imgd = os.path.join(_IJAR, "datasets", "STA", "images")
    iid = str(df.iloc[0]["index"])
    p = None
    for e in ("", ".png", ".jpg", ".jpeg"):
        c = os.path.join(imgd, f"{iid}{e}")
        if os.path.exists(c):
            p = c
            break
    im = Image.open(p).convert("RGB").resize((448, 448))
    px = T.Normalize(MEAN, STD)(T.ToTensor()(im)).unsqueeze(0).to(dev, torch.bfloat16)

    with torch.no_grad():
        f32 = px.float()
        f32 = torch.nn.functional.interpolate(f32, size=(336, 336), mode='bilinear',
                                             align_corners=False)
        out = model.dino_encoder.forward_features(f32)
        feats = out['x_norm_patchtokens']
        print("dino x_norm_patchtokens:", mag(feats), "shape", tuple(feats.shape))
        B, nd, _ = feats.shape
        gd, tg = int(nd ** 0.5), int(model.num_image_token ** 0.5)
        pf = feats.reshape(B, gd, gd, -1).permute(0, 3, 1, 2)
        pf = torch.nn.functional.adaptive_avg_pool2d(pf, (tg, tg))
        pf = pf.permute(0, 2, 3, 1).reshape(B, tg * tg, -1)
        print("pooled:", mag(pf))
        ln = model.dino_projector[0]
        lin1 = model.dino_projector[1]
        gelu = model.dino_projector[2]
        lin3 = model.dino_projector[3]
        a = ln(pf.float())
        print("after LN[0]:", mag(a))
        b = lin1(a)
        print("after Lin[1]:", mag(b))
        c = gelu(b)
        print("after GELU[2]:", mag(c))
        d = lin3(c)
        print("after Lin[3]:", mag(d))
        print("\nLN[0].weight:", mag(ln.weight), "LN[0].bias:", mag(ln.bias))
        print("Lin[1].weight:", mag(lin1.weight), "Lin[1].bias:", mag(lin1.bias))
        print("Lin[3].weight:", mag(lin3.weight), "Lin[3].bias:", mag(lin3.bias))
        # row norm of Lin3 (per output unit)  -> expected output scale
        w3 = lin3.weight.float()
        print(f"\nLin[3] ||row|| mean={w3.norm(dim=1).mean():.4f} "
              f"max={w3.norm(dim=1).max():.4f}")
        w1 = lin1.weight.float()
        print(f"Lin[1] ||row|| mean={w1.norm(dim=1).mean():.4f}")
        # check the full-model native embedding scale for reference
        emb = model.language_model.get_input_embeddings().weight.float()
        print(f"\nLM token embedding table: absmean={emb.abs().mean():.4f} "
              f"std={emb.std():.4f}")
        ic = model.language_model.get_input_embeddings()(
            torch.tensor([model.img_context_token_id], device=dev)).float()
        print("img_context embedding:", mag(ic))
    print("\nDONE")


if __name__ == "__main__":
    main()
