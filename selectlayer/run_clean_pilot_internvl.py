"""
Clean-sample layerwise ACF collection — InternVL2.5-8B
======================================================
Collect per-sample per-layer ACF on the typo-parameter clean set,
mirroring select_layer_internvl.py (same model.loading + patched
model.chat that returns all_input_down_proj), but on clean images.

Usage: python selectlayer/run_clean_pilot_internvl.py [N]
  N = number of clean samples (default 500)
"""

import sys
import os

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
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from envidence import EvidenceModel

# ============================================================================
# Config (identical to select_layer_internvl.py)
# ============================================================================

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
MODEL_CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
NUM_LAYERS = 32
GPU_ID = 0
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/InternVL/weight"
WEIGHT_PATTERN = "internvl_MLP_Layer{layer}_down_proj.pth"
CLEAN_CSV = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/data.csv"
CLEAN_IMAGE_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/images"
OUTPUT = "/data/liuxiaofei/liuxiaofei/IJAR/results/selectlayer/clean_layerauroc_internvl.json"
N_DEFAULT = 500
RANDOM_SEED = 42
MAX_NEW_TOKENS = 128

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
MAX_NUM = 1


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
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448,
                       use_thumbnail=False):
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
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=INPUT_SIZE, max_num=MAX_NUM):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def resolve_image_path(image_dir, image_id):
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = os.path.join(image_dir, image_id)
        if not image_id.lower().endswith(ext):
            candidate = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    import glob as _glob
    matches = _glob.glob(os.path.join(image_dir, f"{image_id}*"))
    return matches[0] if matches else os.path.join(image_dir, image_id)


def compute_layerwise_acf(all_input_down_proj, device):
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
            state_dict = torch.load(weight_path, map_location=device)
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
    print(f"Clean-sample layerwise ACF — InternVL2.5-8B (N={n})")
    print("=" * 70)

    device = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True,
        cache_dir=MODEL_CACHE).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False, cache_dir=MODEL_CACHE)
    print("  Model loaded.")

    df_clean = pd.read_csv(CLEAN_CSV).sample(
        n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    gen_cfg = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    results = []
    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Clean"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        gt = str(row["answer"]).strip().lower()[:1]
        image_path = resolve_image_path(CLEAN_IMAGE_DIR, image_id)
        pixel_values = load_image(image_path, max_num=MAX_NUM).to(
            torch.bfloat16).to(device)

        with torch.inference_mode():
            (response, all_probs, llm_head_hidden_states,
             all_input_down_proj) = model.chat(
                 tokenizer, pixel_values, question, gen_cfg)

        cf = compute_layerwise_acf(all_input_down_proj, device)
        pred = response.strip().lower()[:1]
        results.append({
            "index": image_id, "question": question,
            "ground_truth": gt, "model_output": response,
            "per_layer_conflict": cf,
        })
        print(f"  [{image_id}] out={response.strip()[:40]:<40} "
              f"pred={pred} gt={gt} {'OK' if pred == gt else 'XX'} "
              f"L0={cf[0]:.4f} L31={cf[31]:.4f}")

    n_c = sum(1 for r in results if r["ground_truth"] == r["model_output"].strip().lower()[:1])
    print(f"\n  Clean accuracy: {n_c}/{len(results)} = {n_c/len(results):.3f} "
          f"({len(results)-n_c} incorrect)")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
