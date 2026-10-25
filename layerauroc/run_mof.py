"""
Layer-wise ACF Analysis — MoF-Models-7B (IJAR Phase 1.1)
---------------------------------------------------------
Computes per-layer Attention Conflict (ACF) on the LLaVA-based
MoF model. Follows the same pattern as Qwen/InternVL scripts.

Datasets: STA-5K-Mini (100 samples), RTA-100 (100 samples)
Output:   Per-sample, per-layer conflict values saved as JSON.
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

# LLaVA codebase required for MoF model loading
LLAVA_PATH = "/data/liuxiaofei/uncertainty/liuxiaofei/MMVP/LLaVA"
if LLAVA_PATH not in sys.path:
    sys.path.insert(0, LLAVA_PATH)

# Parent IJAR directory for envidence.py
_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from myllava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)
from myllava.conversation import conv_templates, SeparatorStyle
from myllava.model.builder import load_pretrained_model
from myllava.utils import disable_torch_init
from myllava.mm_utils import (
    tokenizer_image_token, KeywordsStoppingCriteria,
)

from envidence import EvidenceModel

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "/data/public/models/MoF_Models/MoF_Models/MoF_Models"
CONV_MODE = "llava_v1"
NUM_LAYERS = 40

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

OUTPUT_BASE = "/data/liuxiaofei/liuxiaofei/IJAR/results/layerauroc"
N_SAMPLES = None  # use all samples
RANDOM_SEED = 42
MAX_NEW_TOKENS = 128

GEN_KWARGS = dict(
    do_sample=True,
    temperature=0.2,
    top_p=None,
    num_beams=1,
    max_new_tokens=MAX_NEW_TOKENS,
    use_cache=True,
    output_hidden_states=True,
    output_scores=True,
)


# ============================================================================
# Helpers
# ============================================================================

def load_dataset(name, n, seed):
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["csv_path"])
    if n is not None:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
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
    """Build LLaVA-style prompt with <image> token."""
    if model.config.mm_use_im_start_end:
        qs = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN
              + DEFAULT_IM_END_TOKEN + '\n' + question)
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def compute_layerwise_acf(all_input_down_proj, weight_base, weight_pattern,
                          num_layers):
    """Compute per-layer ACF. Exact pattern from original MoF code."""
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


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Layer-wise ACF Analysis — MoF-Models-7B")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Samples per dataset: {N_SAMPLES}")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, "llava", device_map="cuda:0", device="cuda")
    print(f"  Model loaded on cuda:1")

    # Verify weights
    print("\n[2/3] Verifying weight files...")
    for l in [0, NUM_LAYERS - 1]:
        wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=l))
        assert os.path.exists(wp), f"Missing: {wp}"
    print(f"  All {NUM_LAYERS} weight files present.")

    # Process datasets
    print("\n[3/3] Processing datasets...")
    for name in ["STA", "RTA"]:
        print(f"\n{'─'*40}\n  Dataset: {name}\n{'─'*40}")
        cfg = DATASETS[name]
        df = load_dataset(name, N_SAMPLES, RANDOM_SEED)
        results = []

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"Processing {name}"):
            image_id = str(row[cfg["csv_index_col"]])
            question = str(row["question"]).strip()
            ground_truth = str(row.get("answer", "")).strip()

            image_path = resolve_image_path(cfg["image_dir"], image_id)
            image = Image.open(image_path)
            prompt = build_prompt(model, question)

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX,
                return_tensors='pt').unsqueeze(0).cuda()
            image_tensor = image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]

            conv = conv_templates[CONV_MODE].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            stop_str = (conv.sep if conv.sep_style != SeparatorStyle.TWO
                        else conv.sep2)
            keywords = [stop_str]
            # import pdb; pdb.set_trace()
            with torch.inference_mode():
                (output_ids, all_probs, hidden_states_list,
                 all_input_down_proj) = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    **GEN_KWARGS,
                )

            # Decode output
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:],
                skip_special_tokens=True)[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)].strip()

            # Compute per-layer ACF
            acf_result = compute_layerwise_acf(
                all_input_down_proj, WEIGHT_BASE, WEIGHT_PATTERN,
                NUM_LAYERS)

            # Quick verification
            layers_show = [0, NUM_LAYERS // 4, NUM_LAYERS // 2,
                           3 * NUM_LAYERS // 4, NUM_LAYERS - 1]
            cf = acf_result["per_layer_conflict"]
            cf_str = " | ".join(f"L{l}:{cf[l]:.6f}" for l in layers_show)
            mean_cf = sum(cf) / len(cf)
            print(f"  [{image_id}] tokens={acf_result['n_tokens']} "
                  f"ACF mean={mean_cf:.6f} | {cf_str} "
                  f"out={outputs[:60]}")

            results.append({
                "index": image_id,
                "question": question,
                "ground_truth": ground_truth,
                "model_output": outputs,
                "per_layer_conflict": acf_result["per_layer_conflict"],
                "per_layer_ignorance": acf_result["per_layer_ignorance"],
                "per_token_conflict": acf_result["per_token_conflict"],
                "n_tokens": acf_result["n_tokens"],
            })

        os.makedirs(OUTPUT_BASE, exist_ok=True)
        out_path = os.path.join(OUTPUT_BASE,
                                f"{name}_layerauroc_mof.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved -> {out_path}")

    print(f"\n{'='*60}")
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
