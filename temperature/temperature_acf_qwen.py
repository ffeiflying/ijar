"""
Temperature Analysis — Qwen2.5-VL-7B (IJAR Phase 1.2)
------------------------------------------------------
Sweeps attention softmax temperature during inference,
uses layer-0 ACF as classifier score for STA vs RTA.

Datasets: STA-5K-Mini, RTA-100
Output:   AUROC/AUPR at each temperature, saved as JSON + plot.
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
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.metrics import roc_auc_score, average_precision_score

from envidence import EvidenceModel


# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_CACHE = "/data/public/models/Qwen"

# Only layer 0 is used for the classification score
LAYER_IDX = 11  # selected by max |rel_change| (clean vs STA)

WEIGHT_BASE = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/weight"
WEIGHT_PATTERN = "QWen_MLP_down_proj_{layer}_weight.pth"

DATASET_BASE = "/data/liuxiaofei/liuxiaofei/IJAR/datasets"
DATASETS = {
    "STA": {
        "image_dir": f"{DATASET_BASE}/STA/images",
        "csv_path": f"{DATASET_BASE}/STA/sta_mini.csv",
        "csv_index_col": "index",
    },
    "RTA": {
        "image_dir": f"{DATASET_BASE}/RTA/images",
        "csv_path": f"{DATASET_BASE}/RTA/rta_mini.csv",
        "csv_index_col": "index",
    },
}

OUTPUT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/temperature"
N_SAMPLES = None  # use all available
RANDOM_SEED = 42
MAX_NEW_TOKENS = 128
TEMPERATURES = [0.5, 1.0, 1.5, 2.0, 5.0]


# ============================================================================
# Dataset helpers
# ============================================================================

def load_dataset(name, n, seed):
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["csv_path"])
    if n is not None:
        df = df.sample(n=n, random_state=seed)
    df = df.reset_index(drop=True)
    print(f"  [{name}] Loaded {len(df)} samples")
    return df


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
# Core: layer-0 ACF computation
# ============================================================================

class Layer0ACFComputer:
    """Computes ACF for a single target layer (layer 0 by default).

    Pre-loads the weight matrix once to avoid repeated disk I/O.
    """

    def __init__(self, weight_base, weight_pattern, layer_idx, device):
        weight_path = os.path.join(
            weight_base, weight_pattern.format(layer=layer_idx))
        state_dict = torch.load(weight_path, map_location=device)
        self.evidence_model = EvidenceModel(state_dict)
        self.layer_idx = layer_idx
        self.device = device

    def compute(self, all_input_down_proj):
        """Average ACF over all tokens for the target layer."""
        total_conflict = 0.0
        token_count = len(all_input_down_proj)

        for input_layer_list in all_input_down_proj:
            # Keep only the last position per token
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


def _extract_choice(output_text):
    """Extract answer choice 'a' or 'b' from model output.

    Handles formats: "a", "(a)", "a)", "a.", "[a]", "(a) light bulb", etc.
    Returns 'a', 'b', or None.
    """
    text = output_text.strip()
    m = re.match(r'[\[\(]?\s*([abAB])\s*[\]\)\.\,]?', text)
    if m:
        return m.group(1).lower()
    return None


# ============================================================================
# Inference runner
# ============================================================================

def run_dataset(model, processor, dataset_name, acf_computer, temp, n_samples, seed):
    """Run inference, collect per-sample (ACF, answer_choice).

    Returns
    -------
    samples : list of dict
        [{"acf": float, "choice": "a"|"b"|None, "index": str}, ...]
    """
    cfg = DATASETS[dataset_name]
    df = load_dataset(dataset_name, n_samples, seed)
    samples = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {dataset_name} T={temp}"):
        image_id = str(row[cfg["csv_index_col"]])
        question = str(row["question"]).strip()

        image_path = resolve_image_path(cfg["image_dir"], image_id)
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
        ).to("cuda")

        (
            generated_ids, _all_probs, _llm_head_hidden_states,
            all_input_down_proj,
        ) = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            output_hidden_states=True, do_sample=True, temperature=temp)

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
        print(f"    [{image_id}] ACF={acf:.6f} "
              f"choice={choice} out={output_text[:60]}")

    return samples


def compute_dataset_auroc(samples):
    """Compute AUROC/AUPR within one dataset.

    Positive = incorrect (choice != 'a', including 'b' and None).
    Negative = correct (choice == 'a').
    """
    n_total = len(samples)
    y_true = [0 if s["choice"] == 'a' else 1 for s in samples]
    y_score = [-s["acf"] for s in samples]  # L11: low ACF → incorrect
    n_correct = sum(1 for v in y_true if v == 0)
    n_incorrect = sum(1 for v in y_true if v == 1)
    if len(set(y_true)) < 2:
        return float('nan'), float('nan'), n_total, n_correct, n_incorrect
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    return auroc, aupr, n_total, n_correct, n_incorrect


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Temperature Analysis — Qwen2.5-VL-7B")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Layer index:  {LAYER_IDX}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("\n[1/2] Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0",
        cache_dir=MODEL_CACHE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
    print(f"  Model loaded. Device: {model.device}")

    # Set up layer-0 ACF computer
    acf = Layer0ACFComputer(WEIGHT_BASE, WEIGHT_PATTERN, LAYER_IDX, device)

    # Run sweep
    print("\n[2/2] Sweeping temperatures...")
    all_results = []

    for temp in TEMPERATURES:
        print(f"\n{'─'*50}\n  Temperature = {temp}\n{'─'*50}")

        sta_samples = run_dataset(
            model, processor, "STA", acf, temp, N_SAMPLES, RANDOM_SEED)
        rta_samples = run_dataset(
            model, processor, "RTA", acf, temp, N_SAMPLES, RANDOM_SEED)

        # Per-dataset AUROC/AUPR: can ACF predict attack success (b vs a)?
        sta_auroc, sta_aupr, sta_n, sta_correct, sta_incorrect = \
            compute_dataset_auroc(sta_samples)
        rta_auroc, rta_aupr, rta_n, rta_correct, rta_incorrect = \
            compute_dataset_auroc(rta_samples)

        sta_mean = np.mean([s["acf"] for s in sta_samples])
        rta_mean = np.mean([s["acf"] for s in rta_samples])

        print(f"\n  T={temp}:")
        print(f"    STA: AUROC={sta_auroc:.4f}, AUPR={sta_aupr:.4f}, "
              f"mean_ACF={sta_mean:.6f}, "
              f"correct={sta_correct}/{sta_n}, "
              f"incorrect={sta_incorrect}/{sta_n}")
        print(f"    RTA: AUROC={rta_auroc:.4f}, AUPR={rta_aupr:.4f}, "
              f"mean_ACF={rta_mean:.6f}, "
              f"correct={rta_correct}/{rta_n}, "
              f"incorrect={rta_incorrect}/{rta_n}")

        all_results.append({
            "temperature": temp,
            "STA": {
                "auroc": float(sta_auroc) if not np.isnan(sta_auroc) else None,
                "aupr": float(sta_aupr) if not np.isnan(sta_aupr) else None,
                "mean_acf": float(sta_mean),
                "total": sta_n, "correct": sta_correct,
                "incorrect": sta_incorrect,
                "samples": sta_samples,
            },
            "RTA": {
                "auroc": float(rta_auroc) if not np.isnan(rta_auroc) else None,
                "aupr": float(rta_aupr) if not np.isnan(rta_aupr) else None,
                "mean_acf": float(rta_mean),
                "total": rta_n, "correct": rta_correct,
                "incorrect": rta_incorrect,
                "samples": rta_samples,
            },
        })

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "temperature_qwen_L11.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # Quick table
    print(f"\n{'='*70}")
    print(f"  {'T':<8} {'STA_AUROC':<12} {'STA_AUPR':<12} "
          f"{'RTA_AUROC':<12} {'RTA_AUPR':<12}")
    print(f"  {'-'*56}")
    for r in all_results:
        sta = r["STA"]; rta = r["RTA"]
        print(f"  {r['temperature']:<8} "
              f"{sta['auroc'] or 'N/A':<12} {sta['aupr'] or 'N/A':<12} "
              f"{rta['auroc'] or 'N/A':<12} {rta['aupr'] or 'N/A':<12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
