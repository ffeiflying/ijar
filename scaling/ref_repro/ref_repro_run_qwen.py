"""
Layer-wise ACF Analysis for IJAR Paper (Phase 1.1)
---------------------------------------------------
Computes per-layer Attention Conflict (ACF) using DST-based
evidence decomposition on Qwen2.5-VL-7B-Instruct.

Datasets: STA-5K-Mini (100 samples), RTA-100 (100 samples)
Output:   Per-sample, per-layer conflict values saved as JSON
          for downstream AUROC/AUPR computation and plotting.
"""

import sys
import os

# Remove user-local site-packages to avoid CUDA mismatch.
# The conda env already has the correct PyTorch.
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
_IJAR_DIR = "/data/liuxiaofei/liuxiaofei/IJAR"
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

import json
import csv
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Local import — EvidenceModel must be in the same directory
from envidence import EvidenceModel


# ============================================================================
# Configuration
# ============================================================================

# Model
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_CACHE = "/data/public/models/Qwen"
NUM_LAYERS = 28  # Qwen2.5-VL-7B has 28 transformer blocks

# Weight files for MLP down-projection matrices (one per layer)
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/weight"
WEIGHT_PATTERN = "QWen_MLP_down_proj_{layer}_weight.pth"
# Mini datasets (500 images each, we use first 100)
DATASET_BASE = "/data/liuxiaofei/liuxiaofei/IJAR/datasets"
DATASETS = {
    "STA": {
        "image_dir": f"{DATASET_BASE}/STA/images",
        "csv_path": f"{DATASET_BASE}/STA/sta_mini.csv",
        "csv_index_col": "index",  # column matching filename (with extension)
    },
    "RTA": {
        "image_dir": f"{DATASET_BASE}/RTA/images",
        "csv_path": f"{DATASET_BASE}/RTA/rta_mini.csv",
        "csv_index_col": "index",
    },
}

# Output
OUTPUT_BASE = "/data/liuxiaofei/liuxiaofei/IJAR/scaling/ref_repro"
N_SAMPLES = None  # use all samples
RANDOM_SEED = 42


# ============================================================================
# Helpers
# ============================================================================

def load_dataset(name: str, n: int, seed: int) -> pd.DataFrame:
    """Load dataset CSV and sample n rows (use all if n is None)."""
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["csv_path"])
    if n is not None:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"  [{name}] Loaded {len(df)} samples")
    return df


def build_message(image_path: str, question: str) -> list[dict]:
    """Build Qwen2.5-VL chat message for a single image + text prompt."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]


def compute_layerwise_acf(
    all_input_down_proj: list,
    weight_base: str,
    weight_pattern: str,
    num_layers: int,
) -> dict:
    """
    Compute per-layer ACF (conflict κ) from model hidden states.

    Parameters
    ----------
    all_input_down_proj : list
        Output from patched model.generate(); each element is a list of
        per-layer tensors for one generated token.
    weight_base : str
        Directory containing precomputed MLP down-proj weight matrices.
    weight_pattern : str
        Filename pattern with `{layer}` placeholder.
    num_layers : int
        Number of transformer blocks.

    Returns
    -------
    dict with keys:
        - per_layer_conflict : list[float] (length num_layers) — mean
          conflict κ averaged over all tokens at each layer.
        - per_layer_ignorance : list[float] — mean ignorance u.
        - per_token_conflict : list[list[float]] — raw conflict per token
          (inner list length = num_layers).
    """
    total_conflict = [0.0] * num_layers
    total_ignorance = [0.0] * num_layers
    per_token_conflict = []
    token_count = len(all_input_down_proj)

    for input_layer_list in all_input_down_proj:
        # Keep only the last position per token, handling 2D/3D tensors
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
            state_dict = torch.load(weight_path, map_location="cuda")
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


def process_dataset(
    name: str,
    model,
    processor,
    n_samples: int,
    seed: int,
) -> list[dict]:
    """
    Run layer-wise ACF extraction on a dataset.

    Returns a list of per-sample result dicts, each containing:
        - index, question, ground_truth
        - model_output
        - per_layer_conflict, per_layer_ignorance
        - n_tokens
    """
    cfg = DATASETS[name]
    df = load_dataset(name, n=n_samples, seed=seed)
    results = []

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Processing {name}"
    ):
        image_id = str(row[cfg["csv_index_col"]])
        question = str(row["question"]).strip()
        ground_truth = str(row.get("answer", "")).strip()

        # Resolve image path (files may have .png, .jpg, or .jpeg)
        image_dir = cfg["image_dir"]
        image_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(image_dir, image_id)
            if not image_id.lower().endswith(ext):
                candidate = os.path.join(image_dir, f"{image_id}{ext}")
            if os.path.exists(candidate):
                image_path = candidate
                break
        if image_path is None:
            # Try glob
            import glob as _glob
            matches = _glob.glob(os.path.join(image_dir, f"{image_id}*"))
            image_path = matches[0] if matches else os.path.join(image_dir, image_id)

        # Build input
        messages = build_message(image_path, question)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # import pdb; pdb.set_trace()
        # Forward pass with hidden states
        (
            generated_ids,
            all_probs,
            llm_head_hidden_states,
            all_input_down_proj,
        ) = model.generate(
            **inputs, max_new_tokens=128, output_hidden_states=True
        )

        # Decode output
        generated_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Compute per-layer ACF
        acf_result = compute_layerwise_acf(
            all_input_down_proj,
            WEIGHT_BASE,
            WEIGHT_PATTERN,
            NUM_LAYERS,
        )

        # Quick verification: print key layer ACF values per sample
        layers_to_show = [0, NUM_LAYERS // 4, NUM_LAYERS // 2,
                          3 * NUM_LAYERS // 4, NUM_LAYERS - 1]
        cf = acf_result["per_layer_conflict"]
        cf_str = " | ".join(
            f"L{l}:{cf[l]:.6f}" for l in layers_to_show
        )
        mean_cf = sum(cf) / len(cf)
        print(
            f"  [{image_id}] tokens={acf_result['n_tokens']} "
            f"ACF mean={mean_cf:.6f} | {cf_str} "
            f"out={output_text[:60]}"
        )

        results.append(
            {
                "index": image_id,
                "question": question,
                "ground_truth": ground_truth,
                "model_output": output_text,
                "per_layer_conflict": acf_result["per_layer_conflict"],
                "per_layer_ignorance": acf_result["per_layer_ignorance"],
                "per_token_conflict": acf_result["per_token_conflict"],
                "n_tokens": acf_result["n_tokens"],
            }
        )

    return results


def save_results(name: str, results: list[dict]) -> str:
    """Save per-sample results as JSON and return output path."""
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    out_path = os.path.join(OUTPUT_BASE, f"{name}_layerauroc_qwen.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return out_path


def print_summary(name: str, results: list[dict]):
    """Print a quick summary table of per-layer mean ACF."""
    n = len(results)
    layers = range(NUM_LAYERS)
    mean_acf = [
        sum(r["per_layer_conflict"][l] for r in results) / n
        for l in layers
    ]
    print(f"\n  [{name}] Mean ACF per layer (N={n}):")
    print(f"  {'Layer':<8} {'Mean ACF':<14}")
    print(f"  {'-'*22}")
    for l in [0, NUM_LAYERS // 4, NUM_LAYERS // 2,
              3 * NUM_LAYERS // 4, NUM_LAYERS - 1]:
        print(f"  {l:<8} {mean_acf[l]:<14.8f}")
    print(f"  {'Mean':<8} {sum(mean_acf)/len(mean_acf):<14.8f}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Layer-wise ACF Analysis — Qwen2.5-VL-7B-Instruct")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Samples per dataset: {N_SAMPLES}")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        cache_dir=MODEL_CACHE,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
    print(f"  Model device: {model.device}")

    # Verify weight files exist
    print("\n[2/3] Verifying weight files...")
    for l in [0, NUM_LAYERS - 1]:
        wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=l))
        assert os.path.exists(wp), f"Missing: {wp}"
    print(f"  All {NUM_LAYERS} weight files present.")

    # Process datasets
    print("\n[3/3] Processing datasets...")
    for name in ["STA", "RTA"]:
        print(f"\n{'─'*40}\n  Dataset: {name}\n{'─'*40}")
        results = process_dataset(name, model, processor, N_SAMPLES, RANDOM_SEED)
        out_path = save_results(name, results)
        print(f"  Saved → {out_path}")
        print_summary(name, results)

    print(f"\n{'='*60}")
    print("Done. Results saved to:", OUTPUT_BASE)
    print("=" * 60)


if __name__ == "__main__":
    main()
