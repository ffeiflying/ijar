"""
Temperature Analysis — MoF-Models-7B (IJAR Phase 1.2)
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

# LLaVA codebase
LLAVA_PATH = "/data/liuxiaofei/uncertainty/liuxiaofei/MMVP/LLaVA"
if LLAVA_PATH not in sys.path:
    sys.path.insert(0, LLAVA_PATH)

# Parent IJAR directory for envidence.py
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

GEN_KWARGS = dict(
    do_sample=True, temperature=0.2, top_p=None, num_beams=1,
    max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
    output_hidden_states=True, output_scores=True,
)


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


# ============================================================================
# Core: layer-0 ACF computation
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

def run_dataset(model, tokenizer, image_processor, dataset_name,
                acf_computer, temp, n_samples, seed):
    """Run inference, collect per-sample (ACF, answer_choice)."""
    cfg = DATASETS[dataset_name]
    df = load_dataset(dataset_name, n_samples, seed)
    samples = []

    conv = conv_templates[CONV_MODE].copy()
    stop_str = (conv.sep if conv.sep_style != SeparatorStyle.TWO
                else conv.sep2)

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {dataset_name} T={temp}"):
        image_id = str(row[cfg["csv_index_col"]])
        question = str(row["question"]).strip()

        image_path = resolve_image_path(cfg["image_dir"], image_id)
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
                **{**GEN_KWARGS, 'temperature': temp},
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
        print(f"    [{image_id}] ACF={acf:.6f} "
              f"choice={choice} out={outputs[:60]}")

    return samples


def compute_dataset_auroc(samples):
    """Positive = incorrect (choice != 'a'), Negative = correct (choice == 'a')."""
    n_total = len(samples)
    y_true = [0 if s["choice"] == 'a' else 1 for s in samples]
    y_score = [s["acf"] for s in samples]
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
    print("Temperature Analysis — MoF-Models-7B")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Layer index:  {LAYER_IDX}")
    print("=" * 60)

    device = "cuda"

    # Load model
    print("\n[1/2] Loading model...")
    disable_torch_init()
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        MODEL_PATH, None, "llava", device_map="cuda:0", device="cuda")
    print("  Model loaded on cuda:0")

    # Verify weight file
    wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=LAYER_IDX))
    assert os.path.exists(wp), f"Missing: {wp}"
    print(f"  Weight file OK: {wp}")

    # Set up layer-0 ACF computer
    acf = Layer0ACFComputer(WEIGHT_BASE, WEIGHT_PATTERN, LAYER_IDX, device)

    # Run sweep
    print("\n[2/2] Sweeping temperatures...")
    all_results = []

    for temp in TEMPERATURES:
        print(f"\n{'─'*50}\n  Temperature = {temp}\n{'─'*50}")

        sta_samples = run_dataset(
            model, tokenizer, image_processor, "STA",
            acf, temp, N_SAMPLES, RANDOM_SEED)
        rta_samples = run_dataset(
            model, tokenizer, image_processor, "RTA",
            acf, temp, N_SAMPLES, RANDOM_SEED)

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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "temperature_mof_L25.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

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
