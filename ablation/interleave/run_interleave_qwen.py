"""
Qwen2.5-VL-7B DINO Interleaving Inference (IJAR 3.1)
======================================================
Runs inference on STA-5K-Mini with native_only / concat / interleaved modes.

Usage:
    python run_interleave_qwen.py
"""

import sys, os, json, re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, average_precision_score

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

_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

# Add interleave directory for imports
_INTERLEAVE_DIR = os.path.dirname(os.path.abspath(__file__))
if _INTERLEAVE_DIR not in sys.path:
    sys.path.insert(0, _INTERLEAVE_DIR)

from envidence import EvidenceModel

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_CACHE = "/data/public/models/Qwen"

LAYER_IDX = 0
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/weight"
WEIGHT_PATTERN = "QWen_MLP_down_proj_{layer}_weight.pth"

STA_CSV = os.path.join(_IJAR_DIR, "datasets", "STA", "sta_mini.csv")
STA_IMAGE_DIR = os.path.join(_IJAR_DIR, "datasets", "STA", "images")
OUTPUT_DIR = os.path.join(_IJAR_DIR, "results", "interleave")
PROJECTOR_DIR = OUTPUT_DIR

MAX_NEW_TOKENS = 128
GEN_TEMPERATURE = 1.0

MODES = ["native_only", "concat", "interleaved"]


# ── Helpers ─────────────────────────────────────────────────────────────
def _extract_choice(output_text):
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
        "total": n_total, "correct": n_correct, "incorrect": n_incorrect,
    }


def flip_below_half(v):
    if v is None or np.isnan(v): return float('nan')
    return 1.0 - v if v < 0.5 else v


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Qwen2.5-VL-7B DINO Interleaving Inference")
    print(f"Modes: {MODES}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda:1"

    # Load ACF computer (reusable)
    acf = Layer0ACFComputer(WEIGHT_BASE, WEIGHT_PATTERN, LAYER_IDX, device)

    all_results = OrderedDict()

    for mode in MODES:
        print(f"\n{'='*50}\n  Mode: {mode}\n{'='*50}")

        # Load model fresh for each mode
        print("  Loading model...")
        from interleave.qwen.qwen_dino_wrapper import (
            load_qwen_dino, double_image_tokens)

        model = load_qwen_dino(dino_mode=mode, device_map=device)
        model.eval()

        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME, cache_dir=MODEL_CACHE)

        # Load projector weights if available
        if mode != "native_only":
            projector_path = os.path.join(
                PROJECTOR_DIR, f"qwen_dino_projector_{mode}.pth")
            if os.path.exists(projector_path):
                state = torch.load(projector_path, map_location=device)
                model.dino_projector.load_state_dict(state['dino_projector'])
                print(f"  Loaded projector: {projector_path}")
            else:
                print(f"  [WARN] Projector not found: {projector_path}")
                print(f"  Using random projector weights!")

        # Run inference
        df = pd.read_csv(STA_CSV)
        samples = []

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"  STA (Qwen, {mode})"):
            image_id = str(row["index"])
            question = str(row["question"]).strip()

            # Resolve image path
            image_path = os.path.join(STA_IMAGE_DIR, image_id)
            if not os.path.exists(image_path):
                for ext in [".png", ".jpg", ".jpeg"]:
                    cand = os.path.join(STA_IMAGE_DIR, f"{image_id}{ext}")
                    if os.path.exists(cand):
                        image_path = cand
                        break

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
            ).to(device)

            # Double image tokens for DINO mode
            if mode != "native_only":
                inputs = double_image_tokens(inputs)
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor)
                          else v for k, v in inputs.items()}

            with torch.inference_mode():
                # Use model.forward() for Wrapped model (not base.generate)
                # to ensure DINO features are injected
                generated_ids = model.base_model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    output_hidden_states=True,
                    do_sample=True, temperature=GEN_TEMPERATURE)

            # Extract ACF - we need all_input_down_proj
            # For Qwen, the custom generate returns 4 values
            # But the DINO wrapper uses base_model.generate, which may differ
            # Let's handle both cases
            if isinstance(generated_ids, tuple):
                all_input_down_proj = generated_ids[3] if len(generated_ids) > 3 else generated_ids[0]
                generated_ids = generated_ids[0]
            else:
                all_input_down_proj = generated_ids

            # Try to compute ACF
            try:
                acf_value = acf.compute(all_input_down_proj)
            except Exception:
                acf_value = float('nan')

            # Decode
            generated_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]

            choice = _extract_choice(output_text)
            samples.append({"acf": acf_value, "choice": choice,
                            "index": image_id})

        metrics = compute_metrics(samples)
        asr = (metrics['incorrect'] / metrics['total']
               if metrics['total'] > 0 else float('nan'))

        auroc_f = flip_below_half(metrics['auroc'])
        aupr_f = flip_below_half(metrics['aupr'])

        print(f"  AUROC={metrics['auroc']:.4f} (flip={auroc_f:.4f}), "
              f"AUPR={metrics['aupr']:.4f} (flip={aupr_f:.4f}), "
              f"ACC={metrics['accuracy']:.4f}, ASR={asr:.4f}")

        all_results[mode] = {**metrics, "asr": asr,
                              "auroc_flipped": auroc_f, "aupr_flipped": aupr_f}

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Save
    out_path = os.path.join(OUTPUT_DIR, "interleave_qwen.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n{'='*55}")
    print(f"  {'Mode':<14s} {'AUROC':>8s} {'AUPR':>8s} {'ACC':>8s}")
    for mode in MODES:
        r = all_results[mode]
        print(f"  {mode:<14s} {flip_below_half(r['auroc']):>8.4f} "
              f"{flip_below_half(r['aupr']):>8.4f} {r['accuracy']:>8.4f}")
    print(f"  {'='*55}")


if __name__ == "__main__":
    main()
