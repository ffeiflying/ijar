"""
Unimodal ACF conditions for the text<->visual conflict validity experiment.
============================================================================
Question the experiment answers:
  does ACF truly capture *cross-modal* conflict, or does it just fire for any
  image/text input?  We add the two MONO-modal baselines that are missing from
  the existing per-layer bimodal JSONs (STA = attacked image+question,
  clean = congruent image+question, both already saved at n=500 all layers):

     text_only_sta   : STA question text, NO image
     text_only_cln   : clean  question text, NO image
     image_only_att  : STA (attacked, typographic overlay) image + generic prompt
     image_only_cln  : clean (no-overlay source photo)  image + generic prompt

  ACF is computed EXACTLY as in layerauroc/run_qwen.py (per-layer conflict kappa
  at the down-projection input of the last generated-token position, mean over
  generated tokens), so numbers are comparable with the saved STA/clean
  layerauroc JSONs.  One EvidenceModel per layer is preloaded (not reloaded per
  token as in run_qwen.py) -- same math, much faster.

Output (results/crossmodal/):
  crossmodal4cond_ids.json
  text_only_sta_layerwise_qwen.json / text_only_cln_layerwise_qwen.json
  image_only_att_layerwise_qwen.json / image_only_cln_layerwise_qwen.json
"""

import sys
import os

# Remove user-local site-packages to avoid CUDA mismatch (mirror run_qwen.py)
_USER_SITES = [
    os.path.expanduser(p) for p in (
        "~/.local/lib/python3.10/site-packages",
        "~/.local/lib/python3.9/site-packages",
        "~/.local/lib/python3.8/site-packages",
    )
]
for p in _USER_SITES:
    while p in sys.path:
        sys.path.remove(p)

_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from envidence import EvidenceModel

# ============================================================================
# Config
# ============================================================================
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_CACHE = "/data/public/models/Qwen"
NUM_LAYERS = 28
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/weight"
WEIGHT_PATTERN = "QWen_MLP_down_proj_{layer}_weight.pth"

STA_CSV = os.path.join(_IJAR_DIR, "datasets", "STA", "sta_mini.csv")
STA_IMG = os.path.join(_IJAR_DIR, "datasets", "STA", "images")
CLN_CSV = os.path.join(_IJAR_DIR, "typo-parameter", "clean", "data.csv")
CLN_IMG = os.path.join(_IJAR_DIR, "typo-parameter", "clean", "images")

OUT_DIR = os.path.join(_IJAR_DIR, "results", "crossmodal")
GENERIC_PROMPT = "Please describe this image briefly."
N_SUB = 200
SEED = 42
MAX_NEW_TOKENS = 128
DEVICE = "cuda:0"


# ============================================================================
# ACF (identical math to run_qwen.compute_layerwise_acf, but weights preloaded)
# ============================================================================
def load_evidence_models():
    evs = []
    for l in range(NUM_LAYERS):
        path = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=l))
        beta = torch.load(path, map_location=DEVICE)
        evs.append(EvidenceModel(beta))
    return evs


def per_layer_conflict(all_input_down_proj, evs):
    """Mean conflict kappa per layer over generated tokens."""
    total = [0.0] * NUM_LAYERS
    token_count = len(all_input_down_proj)
    for input_layer_list in all_input_down_proj:
        # keep only last position per token (2D/3D handling identical to run_qwen)
        processed = []
        for x in input_layer_list:
            if x.dim() == 3:
                x = x[:, -1:, :]
            elif x.dim() == 2:
                x = x[-1:, :]
            processed.append(x)
        for l, x in enumerate(processed):
            if x.dim() == 3:
                x = x.squeeze(0)
                if x.size(0) > 1:
                    x = x[-1, :].unsqueeze(0)
            evs[l].get_evidence_weights(x.T)
            total[l] += evs[l].get_evidence_conflict().item()
    return [c / token_count for c in total], token_count


# ============================================================================
# Inference helpers (single sample)
# ============================================================================
def _generate(model, processor, messages):
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        generated_ids, _probs, _hidden, all_input_down_proj = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, output_hidden_states=True)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
    out = processor.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    return out, all_input_down_proj


def run_text_only(model, processor, question, evs):
    msgs = [{"role": "user",
             "content": [{"type": "text", "text": question}]}]
    out, all_down = _generate(model, processor, msgs)
    cf, nt = per_layer_conflict(all_down, evs)
    return cf, nt, out


def run_image_only(model, processor, image_path, evs):
    msgs = [{"role": "user",
             "content": [{"type": "image", "image": image_path},
                         {"type": "text", "text": GENERIC_PROMPT}]}]
    out, all_down = _generate(model, processor, msgs)
    cf, nt = per_layer_conflict(all_down, evs)
    return cf, nt, out


def load_subset(csv_path, n, seed):
    df = pd.read_csv(csv_path)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))[:n]
    return df.iloc[idx].reset_index(drop=True)


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading model ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE,
        cache_dir=MODEL_CACHE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)

    print(f"Preloading {NUM_LAYERS} EvidenceModels ...", flush=True)
    evs = load_evidence_models()

    sta = load_subset(STA_CSV, N_SUB, SEED)     # has index/question
    cln = load_subset(CLN_CSV, N_SUB, SEED)
    sta_ids = sta["index"].astype(str).tolist()
    cln_ids = cln["index"].astype(str).tolist()

    # save id lists for offline join with the existing bimodal JSONs
    with open(os.path.join(OUT_DIR, "crossmodal4cond_ids.json"), "w") as f:
        json.dump({"seed": SEED, "n": N_SUB,
                   "sta_ids": sta_ids, "clean_ids": cln_ids}, f, indent=2)

    def run_cond(name, series_iter, fn):
        recs = []
        for i, row in enumerate(tqdm(series_iter, desc=name)):
            cf, nt, out = fn(row)
            recs.append({"index": str(row["index"]), "condition": name,
                         "per_layer_conflict": cf, "n_tokens": nt,
                         "model_output": out})
            if i < 2:
                print(f"   [{i}] ntok={nt} out={out[:60]!r}", flush=True)
        path = os.path.join(OUT_DIR, f"{name}_layerwise_qwen.json")
        with open(path, "w") as f:
            json.dump(recs, f, indent=2, ensure_ascii=False)
        print(f"  saved {path}  ({len(recs)})", flush=True)

    def _series(df):
        return (s for _, s in df.iterrows())

    print("\n[1/4] text_only_sta (STA question, no image)", flush=True)
    run_cond("text_only_sta", _series(sta),
             lambda r: run_text_only(model, processor, str(r["question"]).strip(), evs))

    print("\n[2/4] text_only_cln (clean question, no image)", flush=True)
    run_cond("text_only_cln", _series(cln),
             lambda r: run_text_only(model, processor, str(r["question"]).strip(), evs))

    print("\n[3/4] image_only_att (STA/attacked image + generic prompt)", flush=True)
    run_cond("image_only_att", _series(sta),
             lambda r: run_image_only(model, processor,
                                      os.path.join(STA_IMG, str(r["index"])), evs))

    print("\n[4/4] image_only_cln (clean image + generic prompt)", flush=True)
    run_cond("image_only_cln", _series(cln),
             lambda r: run_image_only(model, processor,
                                      os.path.join(CLN_IMG, str(r["index"])), evs))

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
