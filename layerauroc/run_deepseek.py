"""
Layer-wise ACF Analysis — DeepSeek-VL2 (IJAR Phase 1.1)
-------------------------------------------------------
Computes per-layer Attention Conflict (ACF) using DST-based
evidence decomposition on DeepSeek-VL2.

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

# Parent IJAR directory for envidence.py
_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

from envidence import EvidenceModel


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = ("/data/public/models/deepseek/"
              "models--deepseek-ai--deepseek-vl2-tiny/"
              "snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa")
NUM_LAYERS = 12

# Weight files (MLP down-projection, one per layer)
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/DeepSeek-VL2/weight"
WEIGHT_PATTERN = "DeepSeek_MLP_down_proj_{layer}_weight.pth"

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
}

OUTPUT_BASE = "/data/liuxiaofei/liuxiaofei/IJAR/results/layerauroc"
N_SAMPLES = None  # use all samples
RANDOM_SEED = 42
MAX_NEW_TOKENS = 128


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


def compute_layerwise_acf(all_input_down_proj, weight_base, weight_pattern,
                          num_layers, device):
    """Compute per-layer ACF. Exact pattern from original DeepSeek code."""
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
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Layer-wise ACF Analysis — DeepSeek-VL2")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Samples per dataset: {N_SAMPLES}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load model
    print("\n[1/3] Loading model...")
    vl_chat_processor: DeepseekVLV2Processor = (
        DeepseekVLV2Processor.from_pretrained(MODEL_PATH))
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).to("cuda:0").eval()
    model = (tokenizer, vl_gpt, vl_chat_processor, None)
    print(f"  Model loaded on {device}")

    # Verify weight files
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

            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n<|ref|>{question}<|/ref|>.",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt="",
            ).to(vl_gpt.device)

            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # import pdb; pdb.set_trace()
            with torch.inference_mode():
                outputs = vl_gpt.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    output_hidden_states=True,
                )

            all_input_down_proj = outputs[-1]
            output_ids = outputs[0]

            # Decode
            answer = tokenizer.decode(
                output_ids[0].cpu().tolist(), skip_special_tokens=False)
            if answer.endswith('<｜end▁of▁sentence｜>'):
                answer = answer[:-len('<｜end▁of▁sentence｜>')]

            # Compute per-layer ACF
            acf_result = compute_layerwise_acf(
                all_input_down_proj, WEIGHT_BASE, WEIGHT_PATTERN,
                NUM_LAYERS, device)

            # Quick verification
            layers_show = [0, NUM_LAYERS // 4, NUM_LAYERS // 2,
                           3 * NUM_LAYERS // 4, NUM_LAYERS - 1]
            cf = acf_result["per_layer_conflict"]
            cf_str = " | ".join(f"L{l}:{cf[l]:.6f}" for l in layers_show)
            mean_cf = sum(cf) / len(cf)
            print(f"  [{image_id}] tokens={acf_result['n_tokens']} "
                  f"ACF mean={mean_cf:.6f} | {cf_str} "
                  f"out={answer[:60]}")

            results.append({
                "index": image_id,
                "question": question,
                "ground_truth": ground_truth,
                "model_output": answer,
                "per_layer_conflict": acf_result["per_layer_conflict"],
                "per_layer_ignorance": acf_result["per_layer_ignorance"],
                "per_token_conflict": acf_result["per_token_conflict"],
                "n_tokens": acf_result["n_tokens"],
            })

        os.makedirs(OUTPUT_BASE, exist_ok=True)
        out_path = os.path.join(OUTPUT_BASE,
                                f"{name}_layerauroc_deepseek.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved -> {out_path}")

        # Summary
        n = len(results)
        mean_acf = [
            sum(r["per_layer_conflict"][l] for r in results) / n
            for l in range(NUM_LAYERS)
        ]
        print(f"\n  [{name}] Mean ACF per layer (N={n}):")
        print(f"  {'Layer':<8} {'Mean ACF':<14}")
        print(f"  {'-'*22}")
        for l in layers_show:
            print(f"  {l:<8} {mean_acf[l]:<14.8f}")
        print(f"  {'Mean':<8} {sum(mean_acf)/len(mean_acf):<14.8f}")

    print(f"\n{'='*60}")
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
