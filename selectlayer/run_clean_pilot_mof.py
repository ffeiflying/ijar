"""
Clean-sample layerwise ACF collection — MoF-Models-7B (pilot)
=============================================================
Collect per-sample per-layer ACF on the typo-parameter clean set
(generation settings identical to STA_layerauroc runs), enabling the
4-line plot: clean-correct / clean-incorrect / sta-correct / sta-incorrect.

Usage: python selectlayer/run_clean_pilot_mof.py [N]
  N = number of clean samples (default 50)
"""

import sys, os

_USER_SITES = [os.path.expanduser(p) for p in (
    "~/.local/lib/python3.10/site-packages",
    "~/.local/lib/python3.9/site-packages",
    "~/.local/lib/python3.8/site-packages")]
for p in _USER_SITES:
    while p in sys.path: sys.path.remove(p)

LLAVA_PATH = "/data/liuxiaofei/uncertainty/liuxiaofei/MMVP/LLaVA"
if LLAVA_PATH not in sys.path: sys.path.insert(0, LLAVA_PATH)

_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IJAR_DIR not in sys.path: sys.path.insert(0, _IJAR_DIR)

import json
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from myllava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                               DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from myllava.conversation import conv_templates, SeparatorStyle
from myllava.model.builder import load_pretrained_model
from myllava.utils import disable_torch_init
from myllava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from envidence import EvidenceModel

# ============================================================================
# Config (identical to select_layer_mof.py)
# ============================================================================

MODEL_PATH = "/data/public/models/MoF_Models/MoF_Models/MoF_Models"
CONV_MODE = "llava_v1"
NUM_LAYERS = 40
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/liuxiaofei/MMVP/scripts/weight"
WEIGHT_PATTERN = "LLama_MLP_layer_{layer}_down_proj_weight.pth"
CLEAN_CSV = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/data.csv"
CLEAN_IMAGE_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/images"
OUTPUT = "/data/liuxiaofei/liuxiaofei/IJAR/results/selectlayer/clean_layerauroc_mof.json"
N_DEFAULT = 50
RANDOM_SEED = 42
MAX_NEW_TOKENS = 128

GEN_KWARGS = dict(do_sample=True, temperature=0.2, top_p=None, num_beams=1,
                  max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
                  output_hidden_states=True, output_scores=True)


def resolve_image_path(image_dir, image_id):
    for ext in [".png", ".jpg", ".jpeg"]:
        c = os.path.join(image_dir, image_id)
        if not image_id.lower().endswith(ext):
            c = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(c): return c
    import glob as g
    m = g.glob(os.path.join(image_dir, f"{image_id}*"))
    return m[0] if m else os.path.join(image_dir, image_id)


def build_prompt(model, question):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def compute_layerwise_acf(all_input_down_proj):
    total_conflict = [0.0] * NUM_LAYERS
    token_count = len(all_input_down_proj)
    for input_layer_list in all_input_down_proj:
        processed = []
        for x in input_layer_list:
            if x.dim() == 3: x = x[:, -1:, :]
            elif x.dim() == 2: x = x[-1:, :]
            processed.append(x)
        input_layer_list = processed
        for i, input_proj in enumerate(input_layer_list):
            weight_path = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=i))
            state_dict = torch.load(weight_path, map_location="cuda")
            evm = EvidenceModel(state_dict)
            if input_proj.dim() == 3:
                input_proj = input_proj.squeeze(0)
                if input_proj.size(0) > 1:
                    input_proj = input_proj[-1, :].unsqueeze(0)
            evm.get_evidence_weights(input_proj.T)
            total_conflict[i] += evm.get_evidence_conflict().item()
    return [t / token_count for t in total_conflict]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    print("=" * 70)
    print(f"Clean-sample layerwise ACF — MoF-Models-7B (N={n})")
    print("=" * 70)

    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, "llava", device_map="cuda:0", device="cuda")
    print("  Model loaded.")

    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], None)
    conv.append_message(conv.roles[1], None)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    df_clean = pd.read_csv(CLEAN_CSV).sample(
        n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    results = []
    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Clean"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        gt = str(row["answer"]).strip().lower()[:1]
        image_path = resolve_image_path(CLEAN_IMAGE_DIR, image_id)
        image = Image.open(image_path)

        prompt = build_prompt(model, question)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]

        with torch.inference_mode():
            (output_ids, all_probs, hidden_states_list,
             all_input_down_proj) = model.generate(
                input_ids, images=image_tensor.unsqueeze(0).half().cuda(), **GEN_KWARGS)

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        cf = compute_layerwise_acf(all_input_down_proj)
        pred = outputs.strip().lower()[:1]
        results.append({
            "index": image_id, "question": question,
            "ground_truth": gt, "model_output": outputs,
            "per_layer_conflict": cf,
        })
        print(f"  [{image_id}] out={outputs[:40]:<40} "
              f"pred={pred} gt={gt} {'OK' if pred == gt else 'XX'} "
              f"L0={cf[0]:.4f} L39={cf[39]:.4f}")

    n_c = sum(1 for r in results if r["ground_truth"] == r["model_output"].strip().lower()[:1])
    print(f"\n  Clean accuracy: {n_c}/{len(results)} = {n_c/len(results):.3f} "
          f"({len(results)-n_c} incorrect)")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
