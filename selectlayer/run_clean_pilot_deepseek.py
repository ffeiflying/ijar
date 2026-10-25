"""
Clean-sample layerwise ACF collection — DeepSeek-VL2-Tiny (pilot)
=================================================================
Collect per-sample per-layer ACF on the typo-parameter clean set
(generation settings identical to STA_layerauroc runs), enabling the
4-line plot: clean-correct / clean-incorrect / sta-correct / sta-incorrect.

Usage: python selectlayer/run_clean_pilot_deepseek.py [N]
  N = number of clean samples (default 50)
"""

import sys, os

_USER_SITES = [os.path.expanduser(p) for p in (
    "~/.local/lib/python3.10/site-packages",
    "~/.local/lib/python3.9/site-packages",
    "~/.local/lib/python3.8/site-packages")]
for p in _USER_SITES:
    while p in sys.path: sys.path.remove(p)

_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IJAR_DIR not in sys.path: sys.path.insert(0, _IJAR_DIR)

import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from envidence import EvidenceModel

# ============================================================================
# Config (identical to select_layer_deepseek.py)
# ============================================================================

MODEL_PATH = ("/data/public/models/deepseek/"
              "models--deepseek-ai--deepseek-vl2-tiny/"
              "snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa")
NUM_LAYERS = 12
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/DeepSeek-VL2/weight"
WEIGHT_PATTERN = "DeepSeek_MLP_down_proj_{layer}_weight.pth"
CLEAN_CSV = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/data.csv"
CLEAN_IMAGE_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/images"
OUTPUT = "/data/liuxiaofei/liuxiaofei/IJAR/results/selectlayer/clean_layerauroc_deepseek.json"
N_DEFAULT = 50
RANDOM_SEED = 42
MAX_NEW_TOKENS = 128


def resolve_image_path(image_dir, image_id):
    for ext in [".png", ".jpg", ".jpeg"]:
        c = os.path.join(image_dir, image_id)
        if not image_id.lower().endswith(ext):
            c = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(c): return c
    import glob as g
    m = g.glob(os.path.join(image_dir, f"{image_id}*"))
    return m[0] if m else os.path.join(image_dir, image_id)


def compute_layerwise_acf(all_input_down_proj, device):
    total_conflict = [0.0] * NUM_LAYERS
    token_count = len(all_input_down_proj)
    for input_layer_list in all_input_down_proj:
        processed = []
        for x in input_layer_list:
            if x.dim() == 3: x = x[:, -1:, :]
            elif x.dim() == 2: x = x[-1:, :]
            processed.append(x)
        input_layer_list = processed
        for i, input_proj in enumerate(input_layer_list):
            weight_path = os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=i))
            state_dict = torch.load(weight_path, map_location=device)
            evm = EvidenceModel(state_dict)
            if input_proj.dim() == 3:
                input_proj = input_proj.squeeze(0)
                if input_proj.size(0) > 1:
                    input_proj = input_proj[-1, :].unsqueeze(0)
            evm.get_evidence_weights(input_proj.T)
            total_conflict[i] += evm.get_evidence_conflict().item()
    return [t / token_count for t in total_conflict]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    print("=" * 70)
    print(f"Clean-sample layerwise ACF — DeepSeek-VL2-Tiny (N={n})")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).to("cuda:0").eval()
    print("  Model loaded.")

    df_clean = pd.read_csv(CLEAN_CSV).sample(
        n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    results = []
    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Clean"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        gt = str(row["answer"]).strip().lower()[:1]
        image_path = resolve_image_path(CLEAN_IMAGE_DIR, image_id)

        conversation = [
            {"role": "<|User|>",
             "content": f"<image>\n<|ref|>{question}<|/ref|>.",
             "images": [image_path]},
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
        answer = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=False)
        if answer.endswith('<｜end▁of▁sentence｜>'):
            answer = answer[:-len('<｜end▁of▁sentence｜>')]
        output_text = answer

        cf = compute_layerwise_acf(all_input_down_proj, device)
        pred = output_text.strip().lower()[:1]
        results.append({
            "index": image_id, "question": question,
            "ground_truth": gt, "model_output": output_text,
            "per_layer_conflict": cf,
        })
        print(f"  [{image_id}] out={output_text[:40]:<40} "
              f"pred={pred} gt={gt} {'OK' if pred == gt else 'XX'} "
              f"L0={cf[0]:.4f} L11={cf[11]:.4f}")

    n_c = sum(1 for r in results if r["ground_truth"] == r["model_output"].strip().lower()[:1])
    print(f"\n  Clean accuracy: {n_c}/{len(results)} = {n_c/len(results):.3f} "
          f"({len(results)-n_c} incorrect)")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
