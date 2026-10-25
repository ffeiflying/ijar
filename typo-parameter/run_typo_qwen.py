"""
Typographic Parameter Analysis — Qwen2.5-VL-7B (IJAR Section 2.1)
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
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.metrics import roc_auc_score, average_precision_score

from envidence import EvidenceModel


# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_CACHE = "/data/public/models/Qwen"

LAYER_IDX = 11  # selected by max |rel_change| (clean vs STA)
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/weight"
WEIGHT_PATTERN = "QWen_MLP_down_proj_{layer}_weight.pth"

TYPO_BASE = os.path.join(_IJAR_DIR, "typo-parameter")
OUTPUT_DIR = os.path.join(_IJAR_DIR, "results", "typo_parameter")

MAX_NEW_TOKENS = 128
GEN_TEMPERATURE = 1.0  # fixed at default

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
    y_score = [-s["acf"] for s in samples]  # L11: low ACF → incorrect
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

def run_dataset(model, processor, dataset_name, acf_computer):
    csv_path = os.path.join(TYPO_BASE, dataset_name, "data.csv")

    df = pd.read_csv(csv_path)
    samples = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {dataset_name}"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        image_path = os.path.join(TYPO_BASE, dataset_name, "images", image_id)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ]},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda:1")

        with torch.inference_mode():
            (
                generated_ids, _all_probs, _llm_head_hidden_states,
                all_input_down_proj,
            ) = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                output_hidden_states=True,
                do_sample=True, temperature=GEN_TEMPERATURE)

        acf = acf_computer.compute(all_input_down_proj)

        generated_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

        choice = _extract_choice(output_text)
        samples.append({"acf": acf, "choice": choice, "index": image_id})

    return samples


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Typographic Parameter Analysis — Qwen2.5-VL-7B")
    print(f"Datasets: {len(DATASET_NAMES)}")
    print(f"Layer:     {LAYER_IDX}")
    print(f"Temp:      {GEN_TEMPERATURE}")
    print("=" * 60)

    device = "cuda:1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("\n[1/3] Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:1",
        cache_dir=MODEL_CACHE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
    print(f"  Model loaded on {model.device}")

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

        samples = run_dataset(model, processor, ds_name, acf)
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
    out_path = os.path.join(OUTPUT_DIR, "typo_qwen_L11.json")
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
