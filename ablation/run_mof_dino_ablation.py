"""
MoF DINO Ablation — CLIP-only / DINO-only visual features (IJAR 3.2)
=====================================================================
Runtime monkey-patch only — does NOT modify any source files.

Modes:
  - both:      Full CLIP + DINO interleaved (baseline, untrained format)
  - clip_only: Zero out DINO encoder output (CLIP features only)
  - dino_only: Zero out CLIP encoder output (DINO features only)

Note: Zeroing preserves the 1152-token interleaved structure the model
was trained on, avoiding sequence-length distribution shift.

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

LAYER_IDX = 0
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/liuxiaofei/MMVP/scripts/weight"
WEIGHT_PATTERN = "LLama_MLP_layer_{layer}_down_proj_weight.pth"

STA_CSV = os.path.join(_IJAR_DIR, "datasets", "STA", "sta_mini.csv")
STA_IMAGE_DIR = os.path.join(_IJAR_DIR, "datasets", "STA", "images")
OUTPUT_DIR = os.path.join(_IJAR_DIR, "results", "ablation")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "mof_dino_ablation.json")

MAX_NEW_TOKENS = 128
GEN_TEMPERATURE = 1.0

GEN_KWARGS = dict(
    do_sample=True, temperature=GEN_TEMPERATURE, top_p=None, num_beams=1,
    max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
    output_hidden_states=True, output_scores=True,
)

MODES = ["both", "clip_only", "dino_only"]

# ── Saved original encoder methods (for restore) ───────────────────
_saved_originals = {}


def apply_visual_mode(model, mode):
    """
    Runtime monkey-patch on model instance — does NOT modify source files.
    - clip_only: patch encode_images_withdino to return zeros
    - dino_only: patch encode_images_withclip to return zeros
    - both:       restore originals
    """
    # Save originals once
    if not _saved_originals:
        _saved_originals['dino'] = model.model.encode_images_withdino
        _saved_originals['clip'] = model.model.encode_images_withclip

    # Always restore originals first
    model.model.encode_images_withdino = _saved_originals['dino']
    model.model.encode_images_withclip = _saved_originals['clip']

    if mode == "both":
        print("  [mode] full CLIP+DINO (no patching)")
        return

    if mode == "clip_only":
        orig_dino = _saved_originals['dino']

        def _patched_dino(images):
            result = orig_dino(images)
            return torch.zeros_like(result)

        model.model.encode_images_withdino = _patched_dino
        print("  [mode] CLIP-only (DINO encoder → zeros)")

    elif mode == "dino_only":
        orig_clip = _saved_originals['clip']

        def _patched_clip(images):
            result = orig_clip(images)
            return torch.zeros_like(result)

        model.model.encode_images_withclip = _patched_clip
        print("  [mode] DINO-only (CLIP encoder → zeros)")


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
    """Extract answer choice 'a' or 'b' from model output."""
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


def flip_below_half(v):
    if v is None or np.isnan(v):
        return float('nan')
    return 1.0 - v if v < 0.5 else v


# ============================================================================
# Inference runner
# ============================================================================

def run_inference(model, tokenizer, image_processor, acf_computer):
    df = pd.read_csv(STA_CSV)
    conv = conv_templates[CONV_MODE].copy()
    stop_str = (conv.sep if conv.sep_style != SeparatorStyle.TWO
                else conv.sep2)

    samples = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  STA-5K-Mini"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        image_path = resolve_image_path(STA_IMAGE_DIR, image_id)
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
        samples.append({"acf": acf, "choice": choice, "index": image_id})

    return samples


def resolve_image_path(image_dir, image_id):
    """Try common extensions, fallback to glob."""
    for ext in [".png", ".jpg", ".jpeg"]:
        if image_id.lower().endswith(ext):
            candidate = os.path.join(image_dir, image_id)
        else:
            candidate = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    import glob as _glob
    pattern = os.path.join(image_dir, f"{os.path.splitext(image_id)[0]}.*")
    matches = _glob.glob(pattern)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Image not found: {image_id} in {image_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("MoF DINO Ablation — CLIP-only / DINO-only")
    print(f"Dataset: STA-5K-Mini ({STA_CSV})")
    print(f"Modes:   {MODES}")
    print(f"Layer:   {LAYER_IDX}")
    print(f"Temp:    {GEN_TEMPERATURE}")
    print("=" * 60)

    device = "cuda"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model once
    print("\n[1/3] Loading MoF-Models-7B...")
    disable_torch_init()
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        MODEL_PATH, None, "llava", device_map="cuda:0", device="cuda")
    print("  Model loaded on cuda:0")

    wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=LAYER_IDX))
    assert os.path.exists(wp), f"Missing weight: {wp}"
    print(f"  Weight OK: {wp}")

    acf = Layer0ACFComputer(WEIGHT_BASE, WEIGHT_PATTERN, LAYER_IDX, device)

    # Run each mode
    all_results = OrderedDict()

    for mode in MODES:
        print(f"\n{'='*50}\n  Mode: {mode}\n{'='*50}")

        # Apply runtime monkey-patch for this mode
        apply_visual_mode(model, mode)

        # Run inference
        samples = run_inference(model, tokenizer, image_processor, acf)

        # Metrics
        metrics = compute_metrics(samples)
        asr = (metrics['incorrect'] / metrics['total']
               if metrics['total'] > 0 else float('nan'))

        auroc_f = flip_below_half(metrics['auroc'])
        aupr_f = flip_below_half(metrics['aupr'])

        print(f"  AUROC={metrics['auroc']:.4f} (flipped={auroc_f:.4f}), "
              f"AUPR={metrics['aupr']:.4f} (flipped={aupr_f:.4f}), "
              f"ACC={metrics['accuracy']:.4f}, "
              f"ASR={asr:.4f} ({metrics['incorrect']}/{metrics['total']}), "
              f"mean_ACF={metrics['mean_acf']:.6f}")

        all_results[mode] = {
            **metrics,
            "asr": asr,
            "auroc_flipped": auroc_f,
            "aupr_flipped": aupr_f,
            "samples": samples,
        }

    # Restore original encoder methods
    apply_visual_mode(model, "both")

    # Save
    print(f"\n[3/3] Saving results...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {OUTPUT_JSON}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  {'Mode':<12s} {'AUROC':>8s} {'AUPR':>8s} "
          f"{'flip AUROC':>10s} {'flip AUPR':>10s} "
          f"{'ACC':>8s} {'ASR':>8s} {'mean_ACF':>10s}")
    print(f"  {'-'*68}")
    for mode, r in all_results.items():
        auroc_s = f"{r['auroc']:.4f}" if r['auroc'] is not None else "N/A"
        aupr_s = f"{r['aupr']:.4f}" if r['aupr'] is not None else "N/A"
        auroc_f = f"{r['auroc_flipped']:.4f}"
        aupr_f = f"{r['aupr_flipped']:.4f}"
        print(f"  {mode:<12s} {auroc_s:>8s} {aupr_s:>8s} "
              f"{auroc_f:>10s} {aupr_f:>10s} "
              f"{r['accuracy']:>8.4f} {r['asr']:>8.4f} "
              f"{r['mean_acf']:>10.6f}")
    print(f"  {'='*70}")


if __name__ == "__main__":
    main()
