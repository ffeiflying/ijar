"""
Layer-wise ACF Analysis — InternVL2_5-8B (IJAR Phase 1.1)
----------------------------------------------------------
Computes per-layer Attention Conflict (ACF) using DST-based
evidence decomposition on InternVL2.5-8B.

Datasets: STA-5K-Mini, RTA-100, clean (100 samples for layer selection)
Output:   Per-sample, per-layer conflict values saved as JSON.
          Clean results go to results/selectlayer/; STA/RTA to results/layerauroc/.
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
_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

import json
import torch
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Use the fixed device-agnostic EvidenceModel from IJAR directory
from envidence import EvidenceModel


# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
MODEL_CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
NUM_LAYERS = 32
GPU_ID = 0  # use cuda:0

# Weight files (MLP down-projection, one per layer)
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/InternVL/weight"
WEIGHT_PATTERN = "internvl_MLP_Layer{layer}_down_proj.pth"

# Mini datasets
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
    "clean": {
        "image_dir": "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/images",
        "csv_path": "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/data.csv",
        "csv_index_col": "index",
    },
}

OUTPUT_BASE = "/data/liuxiaofei/liuxiaofei/IJAR/results/layerauroc"
SELECTLAYER_OUTPUT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/selectlayer"
N_SAMPLES = None  # use all samples (for STA/RTA)
N_CLEAN = 100      # clean samples for layer selection
RANDOM_SEED = 42
MAX_NEW_TOKENS = 128

# Image preprocessing (from original InternVL code)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
MAX_NUM = 1  # single tile for speed; original uses up to 12


# ============================================================================
# Image preprocessing (from original InternVL test script)
# ============================================================================

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
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
        if i * j <= max_num and i * j >= min_num
    )
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
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=INPUT_SIZE, max_num=MAX_NUM):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# ============================================================================
# Core ACF computation
# ============================================================================

def compute_layerwise_acf(all_input_down_proj, weight_base, weight_pattern,
                          num_layers, device):
    """Compute per-layer ACF from all_input_down_proj.

    Follows the exact pattern from the verified 32B code.
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
            state_dict = torch.load(weight_path, map_location=device)
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
# Dataset processing
# ============================================================================

def load_dataset(name, n, seed):
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["csv_path"])
    if n is not None:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"  [{name}] Loaded {len(df)} samples")
    return df


def resolve_image_path(image_dir, image_id):
    """Resolve image path trying common extensions."""
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = os.path.join(image_dir, image_id)
        if not image_id.lower().endswith(ext):
            candidate = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    import glob as _glob
    matches = _glob.glob(os.path.join(image_dir, f"{image_id}*"))
    return matches[0] if matches else os.path.join(image_dir, image_id)


def process_dataset(name, model, tokenizer, device, n_samples, seed):
    cfg = DATASETS[name]
    df = load_dataset(name, n=n_samples, seed=seed)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {name}"):
        image_id = str(row[cfg["csv_index_col"]])
        question = str(row["question"]).strip()
        ground_truth = str(row.get("answer", "")).strip()

        image_path = resolve_image_path(cfg["image_dir"], image_id)
        pixel_values = load_image(image_path, max_num=MAX_NUM).to(
            torch.bfloat16).to(device)
        generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        # import pdb; pdb.set_trace()
        # model.chat returns: response, all_probs, llm_head_hidden_states,
        # all_input_down_proj
        (
            response, all_probs, llm_head_hidden_states,
            all_input_down_proj,
        ) = model.chat(tokenizer, pixel_values, question, generation_config)

        acf_result = compute_layerwise_acf(
            all_input_down_proj, WEIGHT_BASE, WEIGHT_PATTERN,
            NUM_LAYERS, device)

        # Quick verification print
        layers_to_show = [0, NUM_LAYERS // 4, NUM_LAYERS // 2,
                          3 * NUM_LAYERS // 4, NUM_LAYERS - 1]
        cf = acf_result["per_layer_conflict"]
        cf_str = " | ".join(f"L{l}:{cf[l]:.6f}" for l in layers_to_show)
        mean_cf = sum(cf) / len(cf)
        print(
            f"  [{image_id}] tokens={acf_result['n_tokens']} "
            f"ACF mean={mean_cf:.6f} | {cf_str} "
            f"out={response[:60]}"
        )

        results.append({
            "index": image_id,
            "question": question,
            "ground_truth": ground_truth,
            "model_output": response,
            "per_layer_conflict": acf_result["per_layer_conflict"],
            "per_layer_ignorance": acf_result["per_layer_ignorance"],
            "per_token_conflict": acf_result["per_token_conflict"],
            "n_tokens": acf_result["n_tokens"],
        })

    return results


def save_results(name, results):
    out_dir = SELECTLAYER_OUTPUT_DIR if name == "clean" else OUTPUT_BASE
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_layerauroc_internvl.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return out_path


def print_summary(name, results):
    n = len(results)
    mean_acf = [
        sum(r["per_layer_conflict"][l] for r in results) / n
        for l in range(NUM_LAYERS)
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
    print("Layer-wise ACF Analysis — InternVL2_5-8B")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Samples per dataset: {N_SAMPLES}")
    print("=" * 60)

    device = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load model
    print("\n[1/3] Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE,
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
        cache_dir=MODEL_CACHE,
    )
    print(f"  Model loaded on {device}")

    # Verify weight files
    print("\n[2/3] Verifying weight files...")
    for l in [0, NUM_LAYERS - 1]:
        wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=l))
        assert os.path.exists(wp), f"Missing: {wp}"
    print(f"  All {NUM_LAYERS} weight files present.")

    # Process datasets
    print("\n[3/3] Processing datasets...")
    for name in ["STA", "RTA", "clean"]:
        n = N_CLEAN if name == "clean" else N_SAMPLES
        print(f"\n{'─'*40}\n  Dataset: {name} (n={n})\n{'─'*40}")
        results = process_dataset(name, model, tokenizer, device,
                                  n, RANDOM_SEED)
        out_path = save_results(name, results)
        print(f"  Saved -> {out_path}")
        print_summary(name, results)

    print(f"\n{'='*60}")
    print("Done. Results saved to:", OUTPUT_BASE)
    print("=" * 60)


if __name__ == "__main__":
    main()
