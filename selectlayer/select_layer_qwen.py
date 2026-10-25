"""
Layer Selection — Qwen2.5-VL-7B (Clean vs STA, Relative ACF Change)
====================================================================
Select best detection layer: max |(mean_STA - mean_clean) / mean_clean|
Usage: python selectlayer/select_layer_qwen.py
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

import json, re
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from envidence import EvidenceModel

# Config
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_CACHE = "/data/public/models/Qwen"
NUM_LAYERS = 28
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/weight"
WEIGHT_PATTERN = "QWen_MLP_down_proj_{layer}_weight.pth"
CLEAN_CSV = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/data.csv"
CLEAN_IMAGE_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/images"
STA_JSON = "/data/liuxiaofei/liuxiaofei/IJAR/results/layerauroc/STA_layerauroc_qwen.json"
OUTPUT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/selectlayer"
N_CLEAN = 50
N_CALIB_STA = 50
RANDOM_SEED = 42

def build_message(image_path, question):
    return [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question}]}]

def compute_layerwise_acf(all_input_down_proj, weight_base, weight_pattern, num_layers):
    total_conflict = [0.0] * num_layers
    token_count = len(all_input_down_proj)
    for input_layer_list in all_input_down_proj:
        processed = []
        for x in input_layer_list:
            if x.dim() == 3: x = x[:, -1:, :]
            elif x.dim() == 2: x = x[-1:, :]
            processed.append(x)
        input_layer_list = processed
        for i, input_proj in enumerate(input_layer_list):
            weight_path = os.path.join(weight_base, weight_pattern.format(layer=i))
            state_dict = torch.load(weight_path, map_location="cuda")
            evm = EvidenceModel(state_dict)
            if input_proj.dim() == 3:
                input_proj = input_proj.squeeze(0)
                if input_proj.size(0) > 1: input_proj = input_proj[-1, :].unsqueeze(0)
            evm.get_evidence_weights(input_proj.T)
            total_conflict[i] += evm.get_evidence_conflict().item()
    return {"per_layer_conflict": [t/token_count for t in total_conflict], "n_tokens": token_count}

def resolve_image_path(image_dir, image_id):
    for ext in [".png", ".jpg", ".jpeg"]:
        c = os.path.join(image_dir, image_id)
        if not image_id.lower().endswith(ext): c = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(c): return c
    import glob as g
    m = g.glob(os.path.join(image_dir, f"{image_id}*"))
    return m[0] if m else os.path.join(image_dir, image_id)

def auroc(labels, scores):
    if len(set(labels)) < 2: return float('nan')
    return float(roc_auc_score(labels, scores))

def main():
    print("=" * 70)
    print("Layer Selection — Qwen2.5-VL-7B (Clean vs STA)")
    print("=" * 70)

    # 1. Load model
    print("\n[1/3] Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=MODEL_CACHE).eval().cuda()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
    print("  Model loaded.")

    for l in [0, NUM_LAYERS-1]:
        assert os.path.exists(os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=l)))
    print(f"  All {NUM_LAYERS} weight files present.")

    # 2. Run inference on clean samples
    print(f"\n[2/3] Running clean inference ({N_CLEAN} samples)...")
    rng = np.random.RandomState(RANDOM_SEED)
    df_clean = pd.read_csv(CLEAN_CSV).sample(n=N_CLEAN, random_state=RANDOM_SEED).reset_index(drop=True)
    clean_results = []

    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Clean"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        image_path = resolve_image_path(CLEAN_IMAGE_DIR, image_id)
        messages = build_message(image_path, question)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to("cuda")
        generated_ids, _, _, all_input_down_proj = model.generate(
            **inputs, max_new_tokens=128, output_hidden_states=True)
        generated_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in
                             zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)[0]
        acf = compute_layerwise_acf(all_input_down_proj, WEIGHT_BASE, WEIGHT_PATTERN, NUM_LAYERS)
        cf = acf["per_layer_conflict"]
        print(f"  [{image_id}] out={output_text[:50]:<50} L0={cf[0]:.4f} L7={cf[7]:.4f} L14={cf[14]:.4f} L21={cf[21]:.4f} L27={cf[27]:.4f}")
        clean_results.append({"per_layer_conflict": cf, "model_output": output_text})

    # 3. Load STA data
    print(f"\n[3/3] Loading STA data + selecting layer...")
    with open(STA_JSON) as f:
        sta_data = json.load(f)

    # Use N_CALIB_STA for calibration
    sta_idx = rng.permutation(len(sta_data))
    calib_sta = [sta_data[i] for i in sta_idx[:N_CALIB_STA]]

    mean_clean = np.mean([d["per_layer_conflict"] for d in clean_results], axis=0)
    mean_sta = np.mean([d["per_layer_conflict"] for d in calib_sta], axis=0)

    eps = 1e-8
    rel_change = (mean_sta - mean_clean) / (mean_clean + eps)
    best_layer = int(np.argmax(np.abs(rel_change)))
    sign = 1 if rel_change[best_layer] > 0 else -1

    print(f"\n  {'Layer':<8} {'Clean ACF':<14} {'STA ACF':<14} {'Rel Change %':<14}")
    for l in sorted(set([0, 4, 8, 12, 16, 20, 24, 27, best_layer])):
        m = " <<< BEST" if l == best_layer else ""
        print(f"  L{l:<7} {mean_clean[l]:<14.8f} {mean_sta[l]:<14.8f} {rel_change[l]*100:<14.4f}{m}")

    direction = "high ACF → attacked" if sign > 0 else "low ACF → attacked"
    print(f"\n  => Selected: L{best_layer} (rel_change={rel_change[best_layer]*100:.4f}%, {direction})")

    # Evaluate on ALL STA samples
    eval_labels = []
    for d in sta_data:
        pred = d["model_output"].strip().lower()[:1]
        gt = d["ground_truth"].strip().lower()[:1]
        eval_labels.append(0 if pred == gt else 1)
    eval_labels = np.array(eval_labels)

    n_c = int(np.sum(eval_labels == 0))
    n_i = int(np.sum(eval_labels == 1))
    print(f"\n  STA eval: {n_c} correct, {n_i} incorrect (ACC={n_c/len(eval_labels):.3f})")

    selected_scores = np.array([d["per_layer_conflict"][best_layer] for d in sta_data])
    selected_auroc = auroc(eval_labels, selected_scores * sign)

    oracle_aurocs = [auroc(eval_labels, np.array([d["per_layer_conflict"][l] for d in sta_data]))
                     for l in range(NUM_LAYERS)]
    oracle_best = int(np.nanargmax(oracle_aurocs))
    oracle_auroc = oracle_aurocs[oracle_best]

    l0_auroc = auroc(eval_labels, np.array([d["per_layer_conflict"][0] for d in sta_data]))
    llast_auroc = auroc(eval_labels, np.array([d["per_layer_conflict"][NUM_LAYERS-1] for d in sta_data]))

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  {'Method':<30} {'Layer':<10} {'AUROC':<10} {'%Oracle':<10}")
    print(f"  {'-'*60}")
    pct = selected_auroc/oracle_auroc*100 if oracle_auroc > 0 else 0
    print(f"  {'Clean vs STA (this work)':<30} L{best_layer:<9} {selected_auroc:<10.4f} {pct:<10.1f}")
    print(f"  {'Oracle (upper bound)':<30} L{oracle_best:<9} {oracle_auroc:<10.4f} {'100.0':<10}")
    pct0 = l0_auroc/oracle_auroc*100 if oracle_auroc > 0 else 0
    print(f"  {'Layer 0':<30} {'L0':<10} {l0_auroc:<10.4f} {pct0:<10.1f}")
    pctl = llast_auroc/oracle_auroc*100 if oracle_auroc > 0 else 0
    print(f"  {'Last Layer':<30} L{NUM_LAYERS-1:<9} {llast_auroc:<10.4f} {pctl:<10.1f}")

    print(f"\n  Per-layer AUROC (★=selected, ◇=oracle):")
    for l in range(NUM_LAYERS):
        sel = " ★" if l == best_layer else ""
        ora = " ◇" if l == oracle_best else ""
        bar = "+" if oracle_aurocs[l] > 0.9 else ("*" if oracle_aurocs[l] > 0.7 else " ")
        print(f"    L{l:2d}: {oracle_aurocs[l]:.4f} {bar}{sel}{ora}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "select_layer_result_qwen.json")
    with open(out_path, "w") as f:
        json.dump({"best_layer": best_layer, "sign": sign, "rel_change": rel_change.tolist(),
                   "selected_auroc": selected_auroc, "oracle_auroc": oracle_auroc,
                   "oracle_best_layer": oracle_best, "oracle_aurocs": oracle_aurocs,
                   "l0_auroc": l0_auroc, "llast_auroc": llast_auroc,
                   "mean_clean": mean_clean.tolist(), "mean_sta_calib": mean_sta.tolist()},
                  f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"\n{'='*70}\nDone.\n{'='*70}")

if __name__ == "__main__":
    main()
