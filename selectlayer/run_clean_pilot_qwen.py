"""
Clean-sample layerwise ACF collection — Qwen2.5-VL-7B (pilot)
================================================================
Collect per-sample per-layer ACF on the typo-parameter clean set
(generation settings identical to STA_layerauroc runs), enabling the
4-line plot: clean-correct / clean-incorrect / sta-correct / sta-incorrect.

Usage: python selectlayer/run_clean_pilot_qwen.py [N]
  N = number of clean samples (default 50)
"""

import sys, os

_USER_SITES = [os.path.expanduser(p) for p in (
    "~/.local/lib/python3.10/site-packages",
    "~/.local/lib/python3.9/site-packages",
    "~/.local/lib/python3.8/site-packages")]
for p in _USER_SITES:
    while p in sys.path: sys.path.remove(p)

_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IJAR_DIR not in sys.path: sys.path.insert(0, _IJAR_DIR)

import json
import numpy as np
import torch
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
CLEAN_CSV = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/data.csv"
CLEAN_IMAGE_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/images"
OUTPUT = "/data/liuxiaofei/liuxiaofei/IJAR/results/selectlayer/clean_layerauroc_qwen.json"
N_DEFAULT = 50
RANDOM_SEED = 42
MAX_NEW_TOKENS = 128


def build_message(image_path, question):
    return [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question}]}]


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


def resolve_image_path(image_dir, image_id):
    for ext in [".png", ".jpg", ".jpeg"]:
        c = os.path.join(image_dir, image_id)
        if not image_id.lower().endswith(ext):
            c = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(c): return c
    import glob as g
    m = g.glob(os.path.join(image_dir, f"{image_id}*"))
    return m[0] if m else os.path.join(image_dir, image_id)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    print("=" * 70)
    print(f"Clean-sample layerwise ACF — Qwen2.5-VL-7B (N={n})")
    print("=" * 70)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=MODEL_CACHE).eval().cuda()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
    print("  Model loaded.")

    df_clean = pd.read_csv(CLEAN_CSV).sample(
        n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    results = []
    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Clean"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        gt = str(row["answer"]).strip().lower()[:1]
        image_path = resolve_image_path(CLEAN_IMAGE_DIR, image_id)
        messages = build_message(image_path, question)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to("cuda")
        generated_ids, _, _, all_input_down_proj = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            output_hidden_states=True)
        generated_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in
                             zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        cf = compute_layerwise_acf(all_input_down_proj)
        pred = output_text.strip().lower()[:1]
        results.append({
            "index": image_id, "question": question,
            "ground_truth": gt, "model_output": output_text,
            "per_layer_conflict": cf,
        })
        print(f"  [{image_id}] out={output_text[:40]:<40} "
              f"pred={pred} gt={gt} {'OK' if pred == gt else 'XX'} "
              f"L0={cf[0]:.4f} L27={cf[27]:.4f}")

    n_c = sum(1 for r in results if r["ground_truth"] == r["model_output"].strip().lower()[:1])
    print(f"\n  Clean accuracy: {n_c}/{len(results)} = {n_c/len(results):.3f} "
          f"({len(results)-n_c} incorrect)")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
