"""
InternVL2_5-8B DINO Interleaving Inference (IJAR 3.1)
=======================================================
Runs inference on STA-5K-Mini with native_only / concat / interleaved modes.

Usage:
    python run_interleave_internvl.py
"""

import sys, os, json, re
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, average_precision_score

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

_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "internvl"))

from envidence import EvidenceModel

# ── Config ──────────────────────────────────────────────────────────────
MODEL_ID = "OpenGVLab/InternVL2_5-8B"
MODEL_CACHE = "/data/public/models/InternVL/InternVL2_5-8B"
GPU_ID = 0

LAYER_IDX = 0
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/InternVL/weight"
WEIGHT_PATTERN = "internvl_MLP_Layer{layer}_down_proj.pth"

STA_CSV = os.path.join(_IJAR_DIR, "datasets", "STA", "sta_mini.csv")
STA_IMAGE_DIR = os.path.join(_IJAR_DIR, "datasets", "STA", "images")
OUTPUT_DIR = os.path.join(_IJAR_DIR, "results", "interleave")
PROJECTOR_DIR = os.path.join(OUTPUT_DIR)

MAX_NEW_TOKENS = 128
GEN_TEMPERATURE = 1.0

MODES = ["native_only", "concat", "interleaved"]

# ── Image preprocessing (from run_typo_internvl.py) ─────────────────────
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
MAX_NUM = 1

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
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
        if i * j <= max_num and i * j >= min_num)
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
            ((i // (target_width // image_size)) + 1) * image_size)
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image(image_file, input_size=INPUT_SIZE, max_num=MAX_NUM):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# ── Helpers ─────────────────────────────────────────────────────────────
def _extract_choice(output_text):
    text = output_text.strip()
    if not text:
        return None
    first = text[0].lower()
    if first in ('a', 'b'):
        rest = text[1:3] if len(text) > 1 else ''
        if not rest or rest[0] in (' ', '.', ',', ')', ']', ':', '\n', '', '('):
            return first
    for pat in [r'\(([abAB])\)', r'\[([abAB])\]',
                r'([abAB])\)', r'\(([abAB])(?:\s|$|[\.\,])',
                r'\b([abAB])[\.\,](?:\s|$)',
                r'(?:^|\s)([abAB])(?:\s|$|\.|\,)']:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1].lower()
    m = re.search(r'\b([abAB])\b', text)
    if m:
        return m.group(1).lower()
    return None


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


def compute_metrics(samples):
    n_total = len(samples)
    y_true = [0 if s["choice"] == 'a' else 1 for s in samples]
    y_score = [s["acf"] for s in samples]
    n_correct = sum(1 for v in y_true if v == 0)
    n_incorrect = sum(1 for v in y_true if v == 1)
    mean_acf = float(np.mean(y_score))
    acc = n_correct / n_total if n_total > 0 else float('nan')

    if len(set(y_true)) < 2:
        auroc, aupr = float('nan'), float('nan')
    else:
        auroc = float(roc_auc_score(y_true, y_score))
        aupr = float(average_precision_score(y_true, y_score))

    return {
        "auroc": auroc if not np.isnan(auroc) else None,
        "aupr": aupr if not np.isnan(aupr) else None,
        "accuracy": acc,
        "mean_acf": mean_acf,
        "total": n_total, "correct": n_correct, "incorrect": n_incorrect,
    }


def flip_below_half(v):
    if v is None or np.isnan(v): return float('nan')
    return 1.0 - v if v < 0.5 else v


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("InternVL2_5-8B DINO Interleaving Inference")
    print(f"Modes: {MODES}")
    print("=" * 60)

    device = f"cuda:{GPU_ID}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from modeling_internvl_chat_dino import InternVLChatModelDINO

    acf = Layer0ACFComputer(WEIGHT_BASE, WEIGHT_PATTERN, LAYER_IDX, device)

    all_results = OrderedDict()

    for mode in MODES:
        print(f"\n{'='*50}\n  Mode: {mode}\n{'='*50}")

        # Load model for each mode (to get clean DINO initialization)
        print("  Loading model...")
        model = InternVLChatModelDINO.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            use_flash_attn=True,
            trust_remote_code=True, cache_dir=MODEL_CACHE,
            ignore_mismatched_sizes=True)  # DINO projector not in original ckpt
        model = model.to(device).eval()

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True, use_fast=False,
            cache_dir=MODEL_CACHE)

        # Set DINO mode
        model.set_dino_mode(mode)
        print(f"  DINO mode: {model.dino_mode}")

        # Load projector weights if available
        if mode != "native_only":
            projector_path = os.path.join(
                PROJECTOR_DIR, f"internvl_dino_projector_{mode}.pth")
            if os.path.exists(projector_path):
                state = torch.load(projector_path, map_location=device)
                model.dino_projector.load_state_dict(state['dino_projector'])
                print(f"  Loaded projector: {projector_path}")
            else:
                print(f"  [WARN] Projector not found: {projector_path}")
                print(f"  Using random projector weights!")

        # Run inference
        df = pd.read_csv(STA_CSV)
        samples = []

        generation_config = dict(max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=True, temperature=GEN_TEMPERATURE)

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"  STA (InternVL, {mode})"):
            image_id = str(row["index"])
            question = str(row["question"]).strip()
            image_path = os.path.join(STA_IMAGE_DIR, image_id)

            # Resolve image path
            for ext in [".png", ".jpg", ".jpeg"]:
                if os.path.exists(image_path):
                    break
                candidate = os.path.join(STA_IMAGE_DIR, f"{image_id}{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break

            pixel_values = load_image(image_path, max_num=MAX_NUM).to(
                torch.bfloat16).to(device)

            response, all_probs, llm_head_hidden_states, all_input_down_proj = \
                model.chat(tokenizer, pixel_values, question, generation_config)

            acf_value = acf.compute(all_input_down_proj)
            choice = _extract_choice(response)

            sample_id = os.path.basename(image_path)
            samples.append({"acf": acf_value, "choice": choice, "index": sample_id})

        metrics = compute_metrics(samples)
        asr = (metrics['incorrect'] / metrics['total']
               if metrics['total'] > 0 else float('nan'))

        auroc_f = flip_below_half(metrics['auroc'])
        aupr_f = flip_below_half(metrics['aupr'])

        print(f"  AUROC={metrics['auroc']:.4f} (flip={auroc_f:.4f}), "
              f"AUPR={metrics['aupr']:.4f} (flip={aupr_f:.4f}), "
              f"ACC={metrics['accuracy']:.4f}, ASR={asr:.4f}")

        all_results[mode] = {**metrics, "asr": asr,
                              "auroc_flipped": auroc_f, "aupr_flipped": aupr_f}

        del model
        torch.cuda.empty_cache()

    # Save
    out_path = os.path.join(OUTPUT_DIR, "interleave_internvl.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n{'='*55}")
    print(f"  {'Mode':<14s} {'AUROC':>8s} {'AUPR':>8s} {'ACC':>8s}")
    for mode in MODES:
        r = all_results[mode]
        print(f"  {mode:<14s} {flip_below_half(r['auroc']):>8.4f} "
              f"{flip_below_half(r['aupr']):>8.4f} {r['accuracy']:>8.4f}")
    print(f"  {'='*55}")


if __name__ == "__main__":
    main()
