"""
Typographic Parameter Analysis — DeepSeek-VL2-Tiny (IJAR Section 2.1)
---------------------------------------------------------------------
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
from transformers import AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, average_precision_score

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

from envidence import EvidenceModel


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = ("/data/public/models/deepseek/"
              "models--deepseek-ai--deepseek-vl2-tiny/"
              "snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa")

LAYER_IDX = 1  # selected by max |rel_change| (clean vs STA)
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/DeepSeek-VL2/weight"
WEIGHT_PATTERN = "DeepSeek_MLP_down_proj_{layer}_weight.pth"

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
        # Verify it's not part of a word like "and", "but", "based"
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

def run_dataset(vl_gpt, vl_chat_processor, tokenizer, dataset_name,
                acf_computer):
    csv_path = os.path.join(TYPO_BASE, dataset_name, "data.csv")

    df = pd.read_csv(csv_path)
    samples = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {dataset_name}"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        image_path = os.path.join(TYPO_BASE, dataset_name, "images", image_id)

        conversation = [
            {"role": "<|User|>",
             "content": f"<image>\n<|ref|>{question}<|/ref|>.",
             "images": [image_path]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images,
            force_batchify=True, system_prompt="").to(vl_gpt.device)

        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        with torch.inference_mode():
            outputs = vl_gpt.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=GEN_TEMPERATURE,
                use_cache=True,
                output_hidden_states=True,
            )

        all_input_down_proj = outputs[-1]
        output_ids = outputs[0]

        # Decode: extract text after the last "<|Assistant|>" marker
        full_text = tokenizer.decode(
            output_ids[0].cpu().tolist(), skip_special_tokens=False)
        marker = '<|Assistant|>'
        if marker in full_text:
            answer = full_text.split(marker)[-1]
        else:
            answer = full_text
        if answer.endswith('<｜end▁of▁sentence｜>'):
            answer = answer[:-len('<｜end▁of▁sentence｜>')]
        answer = answer.strip()

        acf = acf_computer.compute(all_input_down_proj)
        choice = _extract_choice(answer)
        samples.append({"acf": acf, "choice": choice, "index": image_id})

    return samples


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Typographic Parameter Analysis — DeepSeek-VL2-Tiny")
    print(f"Datasets: {len(DATASET_NAMES)}")
    print(f"Layer:     {LAYER_IDX}")
    print(f"Temp:      {GEN_TEMPERATURE}")
    print("=" * 60)

    device = "cuda:1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("\n[1/3] Loading model...")
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).to("cuda:1").eval()
    print(f"  Model loaded on cuda:1")

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

        samples = run_dataset(vl_gpt, vl_chat_processor, tokenizer,
                              ds_name, acf)
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
    out_path = os.path.join(OUTPUT_DIR, "typo_deepseek_L1.json")
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
