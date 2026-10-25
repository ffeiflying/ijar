"""
DeepSeek-VL2-Tiny DINO Feature Fusion Inference (IJAR 3.1)
============================================================
Runs inference on STA-5K-Mini with native_only / fused modes.
DINO features are fused at channel level (concatenate → Linear → same token count).

Usage:
    python run_interleave_deepseek.py
"""

import sys, os, json, re
import numpy as np
import pandas as pd
import torch
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

# Add deepseek DINO model directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "deepseek"))

from envidence import EvidenceModel

# ── Config ──────────────────────────────────────────────────────────────
MODEL_PATH = ("/data/public/models/deepseek/"
              "models--deepseek-ai--deepseek-vl2-tiny/"
              "snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa")
GPU_ID = 0

LAYER_IDX = 0
WEIGHT_BASE = "/data/liuxiaofei/uncertainty/DeepSeek-VL2/weight"
WEIGHT_PATTERN = "DeepSeek_MLP_down_proj_{layer}_weight.pth"

STA_CSV = os.path.join(_IJAR_DIR, "datasets", "STA", "sta_mini.csv")
STA_IMAGE_DIR = os.path.join(_IJAR_DIR, "datasets", "STA", "images")
OUTPUT_DIR = os.path.join(_IJAR_DIR, "results", "interleave")
PROJECTOR_DIR = os.path.join(OUTPUT_DIR)

MAX_NEW_TOKENS = 128
GEN_TEMPERATURE = 1.0

MODES = ["native_only", "fused"]


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
    print("DeepSeek-VL2-Tiny DINO Feature Fusion Inference")
    print(f"Modes: {MODES}")
    print("=" * 60)

    device = f"cuda:{GPU_ID}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from modeling_deepseek_vl_v2_dino import DeepseekVLV2ForCausalLM
    from processing_deepseek_vl_v2 import DeepseekVLV2Processor
    from deepseek_vl2.utils.io import load_pil_images

    acf = Layer0ACFComputer(WEIGHT_BASE, WEIGHT_PATTERN, LAYER_IDX, device)

    all_results = OrderedDict()

    for mode in MODES:
        print(f"\n{'='*50}\n  Mode: {mode}\n{'='*50}")

        # Load model for each mode
        print("  Loading model...")
        model = DeepseekVLV2ForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16,
            device_map=device, ignore_mismatched_sizes=True,
            trust_remote_code=True)
        model = model.to(device).eval()

        processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)
        tokenizer = processor.tokenizer

        model.set_dino_mode(mode)
        print(f"  DINO mode: {model.dino_mode}")

        # Load projector weights if available
        if mode != "native_only":
            projector_path = os.path.join(
                PROJECTOR_DIR, "deepseek_dino_projector_fused.pth")
            if os.path.exists(projector_path):
                state = torch.load(projector_path, map_location=device)
                model.dino_projector.load_state_dict(state['dino_projector'])
                if 'dino_gate' in state:
                    model.dino_gate.data.fill_(state['dino_gate'])
                print(f"  Loaded projector: {projector_path}")
            else:
                print(f"  [WARN] Projector not found: {projector_path}")
                print(f"  Using random projector weights!")

        # Load STA data
        df = pd.read_csv(STA_CSV)
        samples = []
        debug_count = 0
        DEBUG_PRINT = 10  # Print raw output for first N samples

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"  STA (DeepSeek, {mode})"):
            image_id = str(row["index"])
            question = str(row["question"]).strip()
            image_path = os.path.join(STA_IMAGE_DIR, image_id)

            # Resolve image path
            for ext in [".png", ".jpg", ".jpeg"]:
                if os.path.exists(image_path + ext if not image_path.endswith(ext) else image_path):
                    if not image_path.endswith(ext):
                        image_path = image_path + ext
                    break

            if not os.path.exists(image_path):
                continue

            conversation = [
                {"role": "<|User|>",
                 "content": f"<image>\n{question}",
                 "images": [image_path]},
                {"role": "<|Assistant|>", "content": ""},
            ]

            pil_images = load_pil_images(conversation)
            prepare_inputs = processor(
                conversations=conversation, images=pil_images,
                force_batchify=True, system_prompt="").to(device)

            # Compute input embeddings (DINO fusion happens here)
            inputs_embeds = model.prepare_inputs_embeds(
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
            )

            with torch.inference_mode():
                outputs = model.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=GEN_TEMPERATURE,
                    use_cache=True,
                    output_hidden_states=True,
                )

            all_input_down_proj = outputs[-1]
            output_ids = outputs[0]

            # Decode
            full_text = tokenizer.decode(
                output_ids[0].cpu().tolist(), skip_special_tokens=False)
            marker = '<|Assistant|>'
            if marker in full_text:
                answer = full_text.split(marker)[-1]
            else:
                answer = full_text
            if answer.endswith('<｜end▁of▁sentence｜>'):
                answer = answer[:-len('<｜end▁of▁sentence｜>')]
            answer = answer.strip()

            acf_value = acf.compute(all_input_down_proj)
            choice = _extract_choice(answer)

            # Debug: print raw output for first few samples
            if debug_count < DEBUG_PRINT:
                debug_count += 1
                print(f"\n  --- Sample #{debug_count} ---")
                print(f"  Image: {image_id}")
                print(f"  Question: {question[:120]}...")
                print(f"  Full text (last 300 chars): ...{full_text[-300:]}")
                print(f"  Answer extracted: [{answer}]")
                print(f"  Choice parsed: [{choice}]")
                print(f"  ACF: {acf_value:.4f}")

            sample_id = os.path.basename(image_path)
            samples.append({"acf": acf_value, "choice": choice,
                           "index": sample_id})

        metrics = compute_metrics(samples)
        asr = (metrics['incorrect'] / metrics['total']
               if metrics['total'] > 0 else float('nan'))

        auroc_f = flip_below_half(metrics['auroc'])
        aupr_f = flip_below_half(metrics['aupr'])

        print(f"  AUROC={metrics['auroc']:.4f} (flip={auroc_f:.4f}), "
              f"AUPR={metrics['aupr']:.4f} (flip={aupr_f:.4f}), "
              f"ACC={metrics['accuracy']:.4f}, ASR={asr:.4f}")

        all_results[mode] = {**metrics, "asr": asr,
                             "auroc_flipped": auroc_f,
                             "aupr_flipped": aupr_f}

        del model
        torch.cuda.empty_cache()

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "interleave_deepseek.json")
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
