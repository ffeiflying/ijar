"""
End-to-End Layer Selection — InternVL2_5-8B
============================================
Select the best detection layer using ONLY attacked samples:
  1. Run inference on attacked samples, compute per-layer ACF.
  2. Split by model correctness (correct vs incorrect predictions).
  3. Find layer where |mean_correct - mean_incorrect| ACF is largest.
  4. Evaluate AUROC on held-out samples, with sign flip if needed.

Usage:
  python selectlayer/select_layer_internvl.py
"""

import sys
import os

# Remove user-local site-packages to avoid CUDA mismatch
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

# Parent IJAR directory for envidence.py
_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

import json
import re
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from envidence import EvidenceModel


# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
MODEL_CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
NUM_LAYERS = 32
GPU_ID = 0

# Weight files (MLP down-projection, one per layer)
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/InternVL/weight"
WEIGHT_PATTERN = "internvl_MLP_Layer{layer}_down_proj.pth"

# Attacked dataset (STA)
ATTACKED_CSV = "/data/liuxiaofei/liuxiaofei/IJAR/datasets/STA/sta_mini.csv"
ATTACKED_IMAGE_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/datasets/STA/images"

OUTPUT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/selectlayer"
N_TOTAL = 500       # total attacked samples
N_CALIB = 50        # calibration samples per group (correct / incorrect)
RANDOM_SEED = 42
MAX_NEW_TOKENS = 128

# Image preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
MAX_NUM = 1


# ============================================================================
# Image preprocessing
# ============================================================================

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
        if i * j <= max_num and i * j >= min_num
    )
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


# ============================================================================
# Core ACF computation
# ============================================================================

def compute_layerwise_acf(all_input_down_proj, weight_base, weight_pattern,
                          num_layers, device):
    total_conflict = [0.0] * num_layers
    total_ignorance = [0.0] * num_layers
    per_token_conflict = []
    token_count = len(all_input_down_proj)

    for input_layer_list in all_input_down_proj:
        processed = []
        for x in input_layer_list:
            if x.dim() == 3:
                x = x[:, -1:, :]
            elif x.dim() == 2:
                x = x[-1:, :]
            processed.append(x)
        input_layer_list = processed

        token_conflicts = []
        for i, input_proj in enumerate(input_layer_list):
            weight_path = os.path.join(
                weight_base, weight_pattern.format(layer=i))
            state_dict = torch.load(weight_path, map_location=device)
            evidence_model = EvidenceModel(state_dict)

            if input_proj.dim() == 3:
                input_proj = input_proj.squeeze(0)
                if input_proj.size(0) > 1:
                    input_proj = input_proj[-1, :].unsqueeze(0)

            evidence_model.get_evidence_weights(input_proj.T)
            conflict_value = evidence_model.get_evidence_conflict().item()
            ig_value = evidence_model.get_evidence_ignorance().item()

            total_conflict[i] += conflict_value
            total_ignorance[i] += ig_value
            token_conflicts.append(conflict_value)

        per_token_conflict.append(token_conflicts)

    average_conflict = [total / token_count for total in total_conflict]
    average_ig = [total / token_count for total in total_ignorance]

    return {
        "per_layer_conflict": average_conflict,
        "per_layer_ignorance": average_ig,
        "per_token_conflict": per_token_conflict,
        "n_tokens": token_count,
    }


# ============================================================================
# Inference loop
# ============================================================================

def run_inference(df, image_dir, model, tokenizer, device, desc="Inference"):
    """Run model inference, return list of ACF result dicts."""
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        ground_truth = str(row.get("answer", "")).strip()

        image_path = resolve_image_path(image_dir, image_id)
        pixel_values = load_image(image_path, max_num=MAX_NUM).to(
            torch.bfloat16).to(device)
        generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        (
            response, all_probs, llm_head_hidden_states,
            all_input_down_proj,
        ) = model.chat(tokenizer, pixel_values, question, generation_config)

        acf_result = compute_layerwise_acf(
            all_input_down_proj, WEIGHT_BASE, WEIGHT_PATTERN,
            NUM_LAYERS, device)

        cf = acf_result["per_layer_conflict"]
        choice = _extract_choice(response)
        correct = choice == ground_truth.strip().lower()[:1]
        mark = "✓" if correct else "✗"

        # Print EVERY sample for verification
        print(f"  [{mark}] Q={question[:45]:<45} "
              f"GT={ground_truth:<6} out={response.strip()[:50]:<50} "
              f"L0={cf[0]:.4f} L8={cf[8]:.4f} L16={cf[16]:.4f} "
              f"L24={cf[24]:.4f} L31={cf[31]:.4f}")

        results.append({
            "index": image_id,
            "question": question,
            "ground_truth": ground_truth,
            "model_output": response,
            "is_correct": correct,
            "per_layer_conflict": acf_result["per_layer_conflict"],
            "per_layer_ignorance": acf_result["per_layer_ignorance"],
            "per_token_conflict": acf_result["per_token_conflict"],
            "n_tokens": acf_result["n_tokens"],
        })

    return results


# ============================================================================
# Helpers
# ============================================================================

def _extract_choice(output_text):
    """Extract answer choice 'a' or 'b' from model output.
    Handles: 'a', '(a)', 'a)', '[a]', '(a) slipper', etc."""
    text = output_text.strip()
    if not text:
        return None
    first = text[0].lower()
    if first in ('a', 'b'):
        rest = text[1:3] if len(text) > 1 else ''
        if not rest or rest[0] in (' ', '.', ',', ')', ']', ':', '\n', '', '('):
            return first
    for pat in [r'\(([abAB])\)', r'\[([abAB])\]',
                r'([abAB])\)', r'\(([abAB])(?:\s|$|[\.\,])',
                r'\b([abAB])[\.\,](?:\s|$)',
                r'(?:^|\s)([abAB])(?:\s|$|\.|\,)']:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1].lower()
    m = re.search(r'\b([abAB])\b', text)
    if m:
        return m.group(1).lower()
    return None


def compute_auroc(labels, scores):
    if len(set(labels)) < 2:
        return float('nan')
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float('nan')


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("End-to-End Layer Selection — InternVL2_5-8B")
    print(f"Strategy: max |ACF_correct - ACF_incorrect| on attacked samples")
    print(f"Total: {N_TOTAL} | Calib: {N_CALIB} per group")
    print("=" * 70)

    device = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(RANDOM_SEED)

    # ── 1. Load model ────────────────────────────────────────────────────
    print("\n[1/3] Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE,
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
        cache_dir=MODEL_CACHE,
    )
    print(f"  Model loaded on {device}")

    for l in [0, NUM_LAYERS - 1]:
        wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=l))
        assert os.path.exists(wp), f"Missing: {wp}"
    print(f"  All {NUM_LAYERS} weight files present.")

    # ── 2. Run inference ─────────────────────────────────────────────────
    print(f"\n[2/3] Running inference on {N_TOTAL} attacked samples...")
    df = pd.read_csv(ATTACKED_CSV)
    df = df.sample(n=N_TOTAL, random_state=RANDOM_SEED).reset_index(drop=True)
    all_results = run_inference(df, ATTACKED_IMAGE_DIR, model, tokenizer,
                                device, desc="Attacked")

    # Separate by correctness
    correct_samples = [r for r in all_results if r["is_correct"]]
    incorrect_samples = [r for r in all_results if not r["is_correct"]]
    print(f"\n  Correct: {len(correct_samples)} | Incorrect: {len(incorrect_samples)} "
          f"(ACC={len(correct_samples)/len(all_results):.3f})")

    # ── 3. Select layer ──────────────────────────────────────────────────
    print(f"\n[3/3] Selecting layer...")

    # Shuffle and pick calibration samples from each group
    correct_idx = rng.permutation(len(correct_samples))
    incorrect_idx = rng.permutation(len(incorrect_samples))

    calib_correct = [correct_samples[i] for i in correct_idx[:N_CALIB]]
    calib_incorrect = [incorrect_samples[i] for i in incorrect_idx[:N_CALIB]]

    # Remaining for evaluation
    eval_correct = [correct_samples[i] for i in correct_idx[N_CALIB:]]
    eval_incorrect = [incorrect_samples[i] for i in incorrect_idx[N_CALIB:]]
    eval_all = eval_correct + eval_incorrect

    mean_correct = np.mean(
        [d["per_layer_conflict"] for d in calib_correct], axis=0)
    mean_incorrect = np.mean(
        [d["per_layer_conflict"] for d in calib_incorrect], axis=0)

    diff = mean_incorrect - mean_correct  # positive → incorrect has higher ACF
    best_layer = int(np.argmax(np.abs(diff)))
    sign = 1 if diff[best_layer] > 0 else -1

    print(f"\n  {'Layer':<8} {'Correct ACF':<14} {'Incorrect ACF':<14} "
          f"{'Diff (I-C)':<14}")
    print(f"  {'-'*50}")
    for l in sorted(set([0, 3, 8, 12, 16, 20, 24, 28, 31, best_layer])):
        marker = " <<< BEST" if l == best_layer else ""
        print(f"  L{l:<7} {mean_correct[l]:<14.8f} {mean_incorrect[l]:<14.8f} "
              f"{diff[l]:<14.8f}{marker}")

    direction = "high ACF → incorrect" if sign > 0 else "low ACF → incorrect"
    print(f"\n  => Selected: L{best_layer} "
          f"(diff = {diff[best_layer]:.8f}, direction: {direction})")

    # ── 4. Evaluate ──────────────────────────────────────────────────────

    eval_labels = np.array(
        [0 if d["is_correct"] else 1 for d in eval_all])
    n_c = np.sum(eval_labels == 0)
    n_i = np.sum(eval_labels == 1)
    print(f"\n  Eval set: {n_c} correct, {n_i} incorrect")

    # Selected layer (with sign flip)
    selected_scores = np.array(
        [d["per_layer_conflict"][best_layer] for d in eval_all])
    selected_auroc = compute_auroc(eval_labels, selected_scores * sign)

    # Oracle
    oracle_aurocs = []
    for l in range(NUM_LAYERS):
        scores = np.array([d["per_layer_conflict"][l] for d in eval_all])
        oracle_aurocs.append(compute_auroc(eval_labels, scores))
    oracle_best = int(np.nanargmax(oracle_aurocs))
    oracle_auroc = oracle_aurocs[oracle_best]

    # Baselines
    l0_auroc = compute_auroc(eval_labels,
        np.array([d["per_layer_conflict"][0] for d in eval_all]))
    llast_auroc = compute_auroc(eval_labels,
        np.array([d["per_layer_conflict"][NUM_LAYERS - 1] for d in eval_all]))

    # ── Results ──────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  {'Method':<30} {'Layer':<10} {'AUROC':<10} {'%Oracle':<10}")
    print(f"  {'-'*60}")
    pct = selected_auroc / oracle_auroc * 100 if oracle_auroc > 0 else 0
    print(f"  {'This work':<30} L{best_layer:<9} {selected_auroc:<10.4f} {pct:<10.1f}")
    print(f"  {'Oracle (upper bound)':<30} L{oracle_best:<9} {oracle_auroc:<10.4f} {'100.0':<10}")
    pct0 = l0_auroc / oracle_auroc * 100 if oracle_auroc > 0 else 0
    print(f"  {'Layer 0':<30} {'L0':<10} {l0_auroc:<10.4f} {pct0:<10.1f}")
    pct_last = llast_auroc / oracle_auroc * 100 if oracle_auroc > 0 else 0
    print(f"  {'Last Layer (L31)':<30} {'L31':<10} {llast_auroc:<10.4f} {pct_last:<10.1f}")

    # Per-layer AUROC
    print(f"\n  Per-layer AUROC on eval set (★ = selected, ◇ = oracle):")
    for l in range(NUM_LAYERS):
        sel = " ★" if l == best_layer else ""
        ora = " ◇" if l == oracle_best else ""
        bar = "+" if oracle_aurocs[l] > 0.9 else ("*" if oracle_aurocs[l] > 0.7 else " ")
        print(f"    L{l:2d}: {oracle_aurocs[l]:.4f} {bar}{sel}{ora}")

    # ── Save ─────────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "select_layer_result_internvl.json")
    with open(out_path, "w") as f:
        json.dump({
            "strategy": "max_correct_incorrect_diff",
            "best_layer": best_layer,
            "diff": diff.tolist(),
            "sign": sign,
            "mean_correct": mean_correct.tolist(),
            "mean_incorrect": mean_incorrect.tolist(),
            "selected_auroc": selected_auroc,
            "oracle_auroc": oracle_auroc,
            "oracle_best_layer": oracle_best,
            "oracle_aurocs": oracle_aurocs,
            "l0_auroc": l0_auroc,
            "llast_auroc": llast_auroc,
            "n_calib_correct": len(calib_correct),
            "n_calib_incorrect": len(calib_incorrect),
            "n_eval": len(eval_all),
            "n_total_correct": len(correct_samples),
            "n_total_incorrect": len(incorrect_samples),
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved -> {out_path}")

    print(f"\n{'='*70}")
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
