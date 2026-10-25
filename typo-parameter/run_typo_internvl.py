"""
Typographic Parameter Analysis — InternVL2_5-8B (IJAR Section 2.1)
------------------------------------------------------------------
Runs inference on all 19 typographic-parameter datasets,
computes AUROC, AUPR, Accuracy, and mean ACF per dataset.

Positive = incorrect (choice != 'a'), Negative = correct (choice == 'a').
"""

import sys
import os

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

_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

import json
import re
import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, average_precision_score

from envidence import EvidenceModel


# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
MODEL_CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
GPU_ID = 0

LAYER_IDX = 20  # selected by max |ACF_correct - ACF_incorrect|
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/InternVL/weight"
WEIGHT_PATTERN = "internvl_MLP_Layer{layer}_down_proj.pth"

TYPO_BASE = os.path.join(_IJAR_DIR, "typo-parameter")
OUTPUT_DIR = os.path.join(_IJAR_DIR, "results", "typo_parameter")

MAX_NEW_TOKENS = 128
GEN_TEMPERATURE = 1.0  # fixed at default

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
MAX_NUM = 1

# 19 datasets in order, grouped by variable
DATASET_NAMES = [
    "font_size_16", "font_size_20", "font_size_24", "font_size_28", "font_size_32",
    "color_contrast_low", "color_contrast_medium", "color_contrast_high",
    "opacity_30", "opacity_50", "opacity_70", "opacity_90",
    "position_center", "position_corner", "position_random",
    "font_family_serif", "font_family_sans_serif", "font_family_monospace",
    "font_family_decorative",
]


# ============================================================================
# Image preprocessing (from original InternVL code)
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
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
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
            ((i // (target_width // image_size)) + 1) * image_size)
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file, input_size=INPUT_SIZE, max_num=MAX_NUM):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# ============================================================================
# Helpers
# ============================================================================

def _extract_choice(output_text):
    """Extract answer choice 'a' or 'b' from model output.
    Handles: "a", "(a)", "a)", "[a]", "(a", "a.", " A ", "(b) goldfish", etc.
    Returns 'a', 'b', or None if no choice found."""
    text = output_text.strip()
    if not text:
        return None
    # 1) Check first non-whitespace character: models that comply with
    #    "only output the letter" often start with a, b, (, [
    first = text[0].lower()
    if first in ('a', 'b'):
        rest = text[1:3] if len(text) > 1 else ''
        if not rest or rest[0] in (' ', '.', ',', ')', ']', ':', '\n', '', '('):
            return first
    # 2) Patterns: (a) (b) a) b) [a] [b] (a (b a. b.
    for pat in [r'\(([abAB])\)', r'\[([abAB])\]',
                r'([abAB])\)', r'\(([abAB])(?:\s|$|[\.\,])',
                r'\b([abAB])[\.\,](?:\s|$)',
                r'(?:^|\s)([abAB])(?:\s|$|\.|\,)']:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1].lower()
    # 3) Last-resort: standalone a/b word boundary
    m = re.search(r'\b([abAB])\b', text)
    if m:
        return m.group(1).lower()
    return None


# ============================================================================
# Layer-0 ACF computer
# ============================================================================

class Layer0ACFComputer:
    def __init__(self, weight_base, weight_pattern, layer_idx, device):
        weight_path = os.path.join(
            weight_base, weight_pattern.format(layer=layer_idx))
        state_dict = torch.load(weight_path, map_location=device)
        self.evidence_model = EvidenceModel(state_dict)
        self.layer_idx = layer_idx

    def compute(self, all_input_down_proj):
        total_conflict = 0.0
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
            input_proj = input_layer_list[self.layer_idx]
            if input_proj.dim() == 3:
                input_proj = input_proj.squeeze(0)
                if input_proj.size(0) > 1:
                    input_proj = input_proj[-1, :].unsqueeze(0)
            self.evidence_model.get_evidence_weights(input_proj.T)
            total_conflict += self.evidence_model.get_evidence_conflict().item()
        return total_conflict / token_count


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(samples):
    """AUROC/AUPR: positive=incorrect, negative=correct. Also accuracy."""
    n_total = len(samples)
    y_true = [0 if s["choice"] == 'a' else 1 for s in samples]
    y_score = [-s["acf"] for s in samples]  # L20: low ACF → incorrect
    n_correct = sum(1 for v in y_true if v == 0)
    n_incorrect = sum(1 for v in y_true if v == 1)
    mean_acf = float(np.mean(y_score))

    acc = n_correct / n_total if n_total > 0 else float('nan')

    if len(set(y_true)) < 2:
        auroc, aupr = float('nan'), float('nan')
    else:
        auroc = float(roc_auc_score(y_true, y_score))
        aupr = float(average_precision_score(y_true, y_score))

    return {
        "auroc": auroc if not np.isnan(auroc) else None,
        "aupr": aupr if not np.isnan(aupr) else None,
        "accuracy": acc,
        "mean_acf": mean_acf,
        "total": n_total,
        "correct": n_correct,
        "incorrect": n_incorrect,
    }


# ============================================================================
# Inference runner
# ============================================================================

def run_dataset(model, tokenizer, device, dataset_name, acf_computer):
    csv_path = os.path.join(TYPO_BASE, dataset_name, "data.csv")

    df = pd.read_csv(csv_path)
    samples = []
    generation_config = dict(max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=True, temperature=GEN_TEMPERATURE)

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {dataset_name}"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        image_path = os.path.join(TYPO_BASE, dataset_name, "images", image_id)

        pixel_values = load_image(image_path, max_num=MAX_NUM).to(
            torch.bfloat16).to(device)

        response, _all_probs, _llm_head_hidden_states, all_input_down_proj = \
            model.chat(tokenizer, pixel_values, question, generation_config)

        acf = acf_computer.compute(all_input_down_proj)
        choice = _extract_choice(response)
        samples.append({"acf": acf, "choice": choice, "index": image_id})

    return samples


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Typographic Parameter Analysis — InternVL2_5-8B")
    print(f"Datasets: {len(DATASET_NAMES)}")
    print(f"Layer:     {LAYER_IDX}")
    print(f"Temp:      {GEN_TEMPERATURE}")
    print("=" * 60)

    device = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("\n[1/3] Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True,
        cache_dir=MODEL_CACHE).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False,
        cache_dir=MODEL_CACHE)
    print(f"  Model loaded on {device}")

    wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=LAYER_IDX))
    assert os.path.exists(wp), f"Missing weight: {wp}"
    print(f"  Weight OK: {wp}")

    acf = Layer0ACFComputer(WEIGHT_BASE, WEIGHT_PATTERN, LAYER_IDX, device)

    # Run inference on each dataset
    print(f"\n[2/3] Running inference on {len(DATASET_NAMES)} datasets...")
    all_results = OrderedDict()

    for ds_name in DATASET_NAMES:
        print(f"\n{'─'*50}\n  {ds_name}\n{'─'*50}")

        csv_path = os.path.join(TYPO_BASE, ds_name, "data.csv")
        if not os.path.exists(csv_path):
            print(f"  SKIP: {csv_path} not found")
            continue

        samples = run_dataset(model, tokenizer, device, ds_name, acf)
        # Typo-only stats (for ASR)
        metrics = compute_metrics(samples)
        asr = metrics['incorrect'] / metrics['total'] if metrics['total'] > 0 else float('nan')

        print(f"  AUROC={metrics['auroc']}, AUPR={metrics['aupr']}, "
              f"ACC={metrics['accuracy']:.4f}, "
              f"ASR={asr:.4f} ({metrics['incorrect']}/{metrics['total']}), "
              f"mean_ACF={metrics['mean_acf']:.6f}")

        all_results[ds_name] = {
            **metrics,
            "asr": asr,
            "samples": samples,
        }

    # Save
    print(f"\n[3/3] Saving results...")
    out_path = os.path.join(OUTPUT_DIR, "typo_internvl_L20.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  {'Dataset':<28s} {'AUROC':>8s} {'AUPR':>8s} {'ACC':>8s} "
          f"{'mean_ACF':>10s} {'corr':>6s} {'incorr':>6s}")
    print(f"  {'-'*76}")
    for ds_name, r in all_results.items():
        auroc_str = f"{r['auroc']:.4f}" if r['auroc'] is not None else "N/A"
        aupr_str = f"{r['aupr']:.4f}" if r['aupr'] is not None else "N/A"
        print(f"  {ds_name:<28s} {auroc_str:>8s} {aupr_str:>8s} "
              f"{r['accuracy']:>8.4f} {r['mean_acf']:>10.6f} "
              f"{r['correct']:>6d} {r['incorrect']:>6d}")
    print(f"  {'='*80}")


if __name__ == "__main__":
    main()
