"""
Cross-Modal ACF Ablation — Qwen2.5-VL-7B
========================================
Does ACF truly capture cross-modal interaction?

4 conditions × 50 samples each:
  1. Safety QA     (trivia_qa,  text-only)         → lowest ACF (no visual modality)
  2. Typo Text Only (STA question, no image)        → low ACF (text only)
  3. Typo Image Only(STA image, generic prompt)     → medium ACF (visual only)
  4. Typo Image+Text(STA image + STA question)      → highest ACF (genuine cross-modal)

Hypothesis: ACF(Image+Text) >> max(ACF(Image Only), ACF(Text Only))
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

_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import Dataset

from envidence import EvidenceModel

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_CACHE = "/data/public/models/Qwen"

LAYER_IDX = 11         # Qwen best layer (selectlayer result: L11)
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/weight"
WEIGHT_PATTERN = "QWen_MLP_down_proj_{layer}_weight.pth"

STA_CSV = "/data/liuxiaofei/liuxiaofei/IJAR/datasets/STA/sta_mini.csv"
STA_IMAGE_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/datasets/STA/images"
TRIVIA_ARROW = ("/data/liuxiaofei/huggingface/datasets/trivia_qa/"
                "rc.nocontext/0.0.0/0f7faf33a3908546c6fd5b73a660e0f8ff173c2f/"
                "trivia_qa-train.arrow")

OUTPUT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/crossmodal"
N_SAMPLES = 50
RANDOM_SEED = 42
MAX_NEW_TOKENS = 128

GENERIC_IMAGE_PROMPT = "Please describe this image briefly."


# ============================================================================
# ACF Computer
# ============================================================================

class LayerACFComputer:
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
# Inference helpers
# ============================================================================

def run_text_only_inference(model, processor, prompt, acf_computer, device):
    """Text-only inference (no image)."""
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated_ids, _all_probs, _hidden, all_input_down_proj = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            output_hidden_states=True,
            do_sample=False)

    acf = acf_computer.compute(all_input_down_proj)
    generated_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    n_tokens = len(all_input_down_proj)
    return acf, output_text, n_tokens


def run_image_text_inference(model, processor, image_path, prompt, acf_computer,
                              device):
    """Image + text inference."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated_ids, _all_probs, _hidden, all_input_down_proj = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            output_hidden_states=True,
            do_sample=False)

    acf = acf_computer.compute(all_input_down_proj)
    generated_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    n_tokens = len(all_input_down_proj)
    return acf, output_text, n_tokens


# ============================================================================
# Data loading
# ============================================================================

def load_trivia_questions(n, seed):
    ds = Dataset.from_file(TRIVIA_ARROW)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    return [ds[int(i)]["question"] for i in idx]


def load_sta_data(n, seed):
    df = pd.read_csv(STA_CSV)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].reset_index(drop=True)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Cross-Modal ACF Ablation — Qwen2.5-VL-7B")
    print(f"  Layer: L{LAYER_IDX}  |  Samples: {N_SAMPLES} per condition")
    print("=" * 70)

    device = "cuda:0"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("\n[1/3] Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0",
        cache_dir=MODEL_CACHE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
    print(f"  Model on {model.device}")

    wp = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=LAYER_IDX))
    assert os.path.exists(wp), f"Missing weight: {wp}"
    acf_computer = LayerACFComputer(
        WEIGHT_BASE, WEIGHT_PATTERN, LAYER_IDX, device)

    # Load data
    print(f"\n[2/3] Loading data ({N_SAMPLES} per condition)...")
    trivia_questions = load_trivia_questions(N_SAMPLES, RANDOM_SEED)
    sta_df = load_sta_data(N_SAMPLES, RANDOM_SEED)
    print(f"  TriviaQA questions: {len(trivia_questions)}")
    print(f"  STA samples:        {len(sta_df)}")

    # Run 4 conditions
    print(f"\n[3/3] Running 4 conditions...")
    results = {}

    # Condition 1: Safety QA (trivia_qa, text-only)
    print(f"\n{'─'*50}\n  1/4  Safety QA (TriviaQA, text-only)\n{'─'*50}")
    acf_list, token_list = [], []
    for i, q in enumerate(tqdm(trivia_questions, desc="  TriviaQA")):
        acf, out, ntok = run_text_only_inference(
            model, processor, q, acf_computer, device)
        acf_list.append(acf)
        token_list.append(ntok)
        if i < 3:
            print(f"    [{i}] ACF={acf:.6f} ntok={ntok} out={out[:60]}")
    results["safety_qa_trivia"] = {
        "acf_mean": float(np.mean(acf_list)),
        "acf_std": float(np.std(acf_list)),
        "acf_values": acf_list,
        "tokens_mean": float(np.mean(token_list)),
        "tokens_std": float(np.std(token_list)),
    }

    # Condition 2: Typo Text Only (STA question, no image)
    print(f"\n{'─'*50}\n  2/4  Typo Text Only (STA question, no image)\n{'─'*50}")
    acf_list, token_list = [], []
    for i, (_, row) in enumerate(tqdm(sta_df.iterrows(), total=len(sta_df),
                                       desc="  STA-text")):
        question = str(row["question"]).strip()
        acf, out, ntok = run_text_only_inference(
            model, processor, question, acf_computer, device)
        acf_list.append(acf)
        token_list.append(ntok)
        if i < 3:
            print(f"    [{i}] ACF={acf:.6f} ntok={ntok} out={out[:60]}")
    results["typo_text_only"] = {
        "acf_mean": float(np.mean(acf_list)),
        "acf_std": float(np.std(acf_list)),
        "acf_values": acf_list,
        "tokens_mean": float(np.mean(token_list)),
        "tokens_std": float(np.std(token_list)),
    }

    # Condition 3: Typo Image Only (STA image + generic prompt)
    print(f"\n{'─'*50}\n  3/4  Typo Image Only (STA image, generic prompt)\n{'─'*50}")
    acf_list, token_list = [], []
    for i, (_, row) in enumerate(tqdm(sta_df.iterrows(), total=len(sta_df),
                                       desc="  STA-image")):
        image_id = str(row["index"])
        image_path = os.path.join(STA_IMAGE_DIR, image_id)
        acf, out, ntok = run_image_text_inference(
            model, processor, image_path, GENERIC_IMAGE_PROMPT,
            acf_computer, device)
        acf_list.append(acf)
        token_list.append(ntok)
        if i < 3:
            print(f"    [{i}] ACF={acf:.6f} ntok={ntok} out={out[:60]}")
    results["typo_image_only"] = {
        "acf_mean": float(np.mean(acf_list)),
        "acf_std": float(np.std(acf_list)),
        "acf_values": acf_list,
        "tokens_mean": float(np.mean(token_list)),
        "tokens_std": float(np.std(token_list)),
    }

    # Condition 4: Typo Image+Text (full STA)
    print(f"\n{'─'*50}\n  4/4  Typo Image+Text (STA image + question)\n{'─'*50}")
    acf_list, token_list = [], []
    for i, (_, row) in enumerate(tqdm(sta_df.iterrows(), total=len(sta_df),
                                       desc="  STA-full")):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        image_path = os.path.join(STA_IMAGE_DIR, image_id)
        acf, out, ntok = run_image_text_inference(
            model, processor, image_path, question, acf_computer, device)
        acf_list.append(acf)
        token_list.append(ntok)
        if i < 3:
            print(f"    [{i}] ACF={acf:.6f} ntok={ntok} out={out[:60]}")
    results["typo_image_text"] = {
        "acf_mean": float(np.mean(acf_list)),
        "acf_std": float(np.std(acf_list)),
        "acf_values": acf_list,
        "tokens_mean": float(np.mean(token_list)),
        "tokens_std": float(np.std(token_list)),
    }

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS: Mean ACF by Condition")
    print(f"{'='*70}")
    print(f"  {'Condition':<25s} {'Mean ACF':<14s} {'Std ACF':<14s} "
          f"{'Mean Tokens':<14s}")
    print(f"  {'-'*66}")

    ordered_keys = ["safety_qa_trivia", "typo_text_only",
                    "typo_image_only", "typo_image_text"]
    for k in ordered_keys:
        r = results[k]
        print(f"  {k:<25s} {r['acf_mean']:<14.8f} {r['acf_std']:<14.8f} "
              f"{r['tokens_mean']:<14.1f}")

    # Key comparison: image+text vs max(image only, text only)
    it = results["typo_image_text"]["acf_mean"]
    io = results["typo_image_only"]["acf_mean"]
    to = results["typo_text_only"]["acf_mean"]
    sq = results["safety_qa_trivia"]["acf_mean"]
    ratio = it / max(io, to) if max(io, to) > 0 else float('inf')
    print(f"\n  Cross-modal ratio: ACF(image+text) / max(ACF(image), ACF(text)) "
          f"= {ratio:.2f}x")
    print(f"  Safety QA baseline: {sq:.8f}")
    print(f"{'='*70}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "crossmodal_qwen_L11.json")
    with open(out_path, "w") as f:
        # Don't save raw acf_values (list of floats is fine)
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
