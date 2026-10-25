"""
Typographic Parameter Analysis — MoF-Models-7B (IJAR Section 2.1)
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

LLAVA_PATH = "/data/liuxiaofei/uncertainty/liuxiaofei/MMVP/LLaVA"
if LLAVA_PATH not in sys.path:
    sys.path.insert(0, LLAVA_PATH)

_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

import json
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, average_precision_score

from myllava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)
from myllava.conversation import conv_templates, SeparatorStyle
from myllava.model.builder import load_pretrained_model
from myllava.utils import disable_torch_init
from myllava.mm_utils import tokenizer_image_token

from envidence import EvidenceModel


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "/data/public/models/MoF_Models/MoF_Models/MoF_Models"
CONV_MODE = "llava_v1"

LAYER_IDX = 25  # selected by max |rel_change| (clean vs STA)
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/liuxiaofei/MMVP/scripts/weight"
WEIGHT_PATTERN = "LLama_MLP_layer_{layer}_down_proj_weight.pth"

TYPO_BASE = os.path.join(_IJAR_DIR, "typo-parameter")
OUTPUT_DIR = os.path.join(_IJAR_DIR, "results", "typo_parameter")

MAX_NEW_TOKENS = 128
GEN_TEMPERATURE = 1.0  # fixed at default

GEN_KWARGS = dict(
    do_sample=True, temperature=GEN_TEMPERATURE, top_p=None, num_beams=1,
    max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
    output_hidden_states=True, output_scores=True,
)

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

def build_prompt(model, question):
    if model.config.mm_use_im_start_end:
        qs = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN
              + DEFAULT_IM_END_TOKEN + '\n' + question)
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


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
    y_score = [s["acf"] for s in samples]
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

def run_dataset(model, tokenizer, image_processor, dataset_name, acf_computer):
    csv_path = os.path.join(TYPO_BASE, dataset_name, "data.csv")
    image_dir = os.path.join(TYPO_BASE, dataset_name, "images")

    df = pd.read_csv(csv_path)
    samples = []

    conv = conv_templates[CONV_MODE].copy()
    stop_str = (conv.sep if conv.sep_style != SeparatorStyle.TWO
                else conv.sep2)

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {dataset_name}"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        image_path = os.path.join(image_dir, image_id)
        image = Image.open(image_path)
        prompt = build_prompt(model, question)

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]

        with torch.inference_mode():
            (
                output_ids, _all_probs, _hidden_states_list,
                all_input_down_proj,
            ) = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                stopping_criteria=None,
                **GEN_KWARGS,
            )

        acf = acf_computer.compute(all_input_down_proj)

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:],
            skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        choice = _extract_choice(outputs)
        print(f"    [{image_id}] raw={outputs[:100]} -> choice={choice}")
        samples.append({"acf": acf, "choice": choice, "index": image_id})

    return samples


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Typographic Parameter Analysis — MoF-Models-7B")
    print(f"Datasets: {len(DATASET_NAMES)}")
    print(f"Layer:     {LAYER_IDX}")
    print(f"Temp:      {GEN_TEMPERATURE}")
    print("=" * 60)

    device = "cuda"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("\n[1/3] Loading model...")
    disable_torch_init()
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        MODEL_PATH, None, "llava", device_map="cuda:0", device="cuda")
    print("  Model loaded on cuda:0")

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

        samples = run_dataset(model, tokenizer, image_processor,
                              ds_name, acf)
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
    out_path = os.path.join(OUTPUT_DIR, "typo_mof_L25.json")
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
