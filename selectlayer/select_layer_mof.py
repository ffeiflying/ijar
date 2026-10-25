"""
Layer Selection — MoF-Models-7B (Clean vs STA, Relative ACF Change)
====================================================================
Select best detection layer: max |(mean_STA - mean_clean) / mean_clean|
Usage: python selectlayer/select_layer_mof.py
"""

import sys, os

_USER_SITES = [os.path.expanduser(p) for p in (
    "~/.local/lib/python3.10/site-packages",
    "~/.local/lib/python3.9/site-packages",
    "~/.local/lib/python3.8/site-packages")]
for p in _USER_SITES:
    while p in sys.path: sys.path.remove(p)

LLAVA_PATH = "/data/liuxiaofei/uncertainty/liuxiaofei/MMVP/LLaVA"
if LLAVA_PATH not in sys.path: sys.path.insert(0, LLAVA_PATH)

_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IJAR_DIR not in sys.path: sys.path.insert(0, _IJAR_DIR)

import json
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from myllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from myllava.conversation import conv_templates, SeparatorStyle
from myllava.model.builder import load_pretrained_model
from myllava.utils import disable_torch_init
from myllava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from envidence import EvidenceModel

# Config
MODEL_PATH = "/data/public/models/MoF_Models/MoF_Models/MoF_Models"
CONV_MODE = "llava_v1"
NUM_LAYERS = 40
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/liuxiaofei/MMVP/scripts/weight"
WEIGHT_PATTERN = "LLama_MLP_layer_{layer}_down_proj_weight.pth"
CLEAN_CSV = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/data.csv"
CLEAN_IMAGE_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean/images"
STA_JSON = "/data/liuxiaofei/liuxiaofei/IJAR/results/layerauroc/STA_layerauroc_mof.json"
OUTPUT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/selectlayer"
N_CLEAN = 50
N_CALIB_STA = 50
RANDOM_SEED = 42
MAX_NEW_TOKENS = 128

GEN_KWARGS = dict(do_sample=True, temperature=0.2, top_p=None, num_beams=1,
                  max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
                  output_hidden_states=True, output_scores=True)

def resolve_image_path(image_dir, image_id):
    for ext in [".png", ".jpg", ".jpeg"]:
        c = os.path.join(image_dir, image_id)
        if not image_id.lower().endswith(ext): c = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(c): return c
    import glob as g
    m = g.glob(os.path.join(image_dir, f"{image_id}*"))
    return m[0] if m else os.path.join(image_dir, image_id)

def build_prompt(model, question):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

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

def auroc(labels, scores):
    if len(set(labels)) < 2: return float('nan')
    return float(roc_auc_score(labels, scores))

def main():
    print("=" * 70)
    print("Layer Selection — MoF-Models-7B (Clean vs STA)")
    print("=" * 70)

    # 1. Load model
    print("\n[1/3] Loading model...")
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, "llava", device_map="cuda:0", device="cuda")
    print("  Model loaded.")

    for l in [0, NUM_LAYERS-1]:
        assert os.path.exists(os.path.join(WEIGHT_BASE, WEIGHT_PATTERN.format(layer=l)))
    print(f"  All {NUM_LAYERS} weight files present.")

    # 2. Run inference on clean samples
    print(f"\n[2/3] Running clean inference ({N_CLEAN} samples)...")
    rng = np.random.RandomState(RANDOM_SEED)
    df_clean = pd.read_csv(CLEAN_CSV).sample(n=N_CLEAN, random_state=RANDOM_SEED).reset_index(drop=True)
    clean_results = []

    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], None)
    conv.append_message(conv.roles[1], None)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Clean"):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        image_path = resolve_image_path(CLEAN_IMAGE_DIR, image_id)
        image = Image.open(image_path)

        prompt = build_prompt(model, question)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        with torch.inference_mode():
            (output_ids, all_probs, hidden_states_list,
             all_input_down_proj) = model.generate(
                input_ids, images=image_tensor.unsqueeze(0).half().cuda(), **GEN_KWARGS)

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:],
                                         skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str): outputs = outputs[:-len(stop_str)].strip()

        acf = compute_layerwise_acf(all_input_down_proj, WEIGHT_BASE, WEIGHT_PATTERN, NUM_LAYERS)
        cf = acf["per_layer_conflict"]
        print(f"  [{image_id}] out={outputs[:50]:<50} L0={cf[0]:.4f} L10={cf[10]:.4f} L20={cf[20]:.4f} L30={cf[30]:.4f} L39={cf[39]:.4f}")
        clean_results.append({"per_layer_conflict": cf, "model_output": outputs})

    # 3. Load STA + select
    print(f"\n[3/3] Loading STA data + selecting layer...")
    with open(STA_JSON) as f:
        sta_data = json.load(f)

    sta_idx = rng.permutation(len(sta_data))
    calib_sta = [sta_data[i] for i in sta_idx[:N_CALIB_STA]]

    mean_clean = np.mean([d["per_layer_conflict"] for d in clean_results], axis=0)
    mean_sta = np.mean([d["per_layer_conflict"] for d in calib_sta], axis=0)

    eps = 1e-8
    rel_change = (mean_sta - mean_clean) / (mean_clean + eps)
    best_layer = int(np.argmax(np.abs(rel_change)))
    sign = 1 if rel_change[best_layer] > 0 else -1

    print(f"\n  {'Layer':<8} {'Clean ACF':<14} {'STA ACF':<14} {'Rel Change %':<14}")
    for l in sorted(set([0, 5, 10, 15, 20, 25, 30, 35, 39, best_layer])):
        m = " <<< BEST" if l == best_layer else ""
        print(f"  L{l:<7} {mean_clean[l]:<14.8f} {mean_sta[l]:<14.8f} {rel_change[l]*100:<14.4f}{m}")

    direction = "high ACF → attacked" if sign > 0 else "low ACF → attacked"
    print(f"\n  => Selected: L{best_layer} (rel_change={rel_change[best_layer]*100:.4f}%, {direction})")

    # Evaluate
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
    out_path = os.path.join(OUTPUT_DIR, "select_layer_result_mof.json")
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
