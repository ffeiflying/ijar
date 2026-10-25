"""
Layer-wise ACF extraction for the model-scaling experiment (see DESIGN.md).

Same method and dataset as IJAR/layerauroc/run_qwen.py, but:
  * down_proj inputs are captured with forward hooks instead of a patched
    transformers, so it runs in the clean `EUQ` env;
  * down_proj weights are read live off the model -- nothing is written to disk;
  * the ACF closed form in acf_core.py keeps 32B feasible;
  * output goes to scaling/results/, never over the existing results/.

Run (from this directory):
    /data/liuxiaofei/miniconda3/envs/EUQ/bin/python run_scaling.py \
        --model Qwen3-VL-8B-Instruct --datasets STA RTA
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
import time

import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Parent IJAR dir, for the original envidence.py
_IJAR_DIR = os.path.dirname(_HERE)
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

from acf_core import (  # noqa: E402
    DownProjCapture,
    precompute_layer_stats,
    layer_acf,
    resolve_text_layers,
)


# --------------------------------------------------------------------------
# Reuse the reference answer parser verbatim so labels stay comparable
# --------------------------------------------------------------------------

def _load_reference_parser():
    ref_path = os.path.join(_IJAR_DIR, "layerauroc", "layer_auroc.py")
    spec = importlib.util.spec_from_file_location("layer_auroc_ref", ref_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._extract_choice


_extract_choice = _load_reference_parser()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATASET_BASE = os.path.join(_IJAR_DIR, "datasets")
OUTPUT_BASE = os.path.join(_HERE, "results")

_QWEN_PUBLIC = "/data/public/models/Qwen"

MODELS = {
    "Qwen3-VL-8B-Instruct": {
        "path": f"{_QWEN_PUBLIC}/Qwen3-VL-8B-Instruct",
        "family": "qwen3vl",
        "params_b": 8.0,
    },
    "Qwen3-VL-32B-Instruct": {
        "path": f"{_QWEN_PUBLIC}/Qwen3-VL-32B-Instruct",
        "family": "qwen3vl",
        "params_b": 32.0,
    },
    # Qwen2.5-VL checkpoints live in HF-cache layout; point at the snapshot.
    "Qwen2.5-VL-3B-Instruct": {
        "path": (
            f"{_QWEN_PUBLIC}/Qwen2.5-VL-3B-Instruct/"
            "models--Qwen--Qwen2.5-VL-3B-Instruct/"
            "snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
        ),
        "family": "qwen25vl",
        "params_b": 3.0,
    },
    "Qwen2.5-VL-7B-Instruct": {
        "path": (
            f"{_QWEN_PUBLIC}/models--Qwen--Qwen2.5-VL-7B-Instruct/"
            "snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
        ),
        "family": "qwen25vl",
        "params_b": 7.0,
    },
    "Qwen2.5-VL-32B-Instruct": {
        "path": (
            f"{_QWEN_PUBLIC}/Qwen2.5-VL-32B-Instruct/"
            "models--Qwen--Qwen2.5-VL-32B-Instruct/"
            "snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da"
        ),
        "family": "qwen25vl",
        "params_b": 32.0,
    },
}

DATASETS = {
    "STA": {
        "image_dir": os.path.join(DATASET_BASE, "STA", "images"),
        "csv_path": os.path.join(DATASET_BASE, "STA", "sta_mini.csv"),
        "index_col": "index",
    },
    "RTA": {
        "image_dir": os.path.join(DATASET_BASE, "RTA", "images"),
        "csv_path": os.path.join(DATASET_BASE, "RTA", "rta_mini.csv"),
        "index_col": "index",
    },
}

MAX_NEW_TOKENS = 128  # same as the reference run


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_model(model_name, dtype, use_fast=False, gpu=0, device_map=None):
    cfg = MODELS[model_name]
    path = cfg["path"]
    if cfg["family"] == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration as Cls
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    from transformers import AutoProcessor

    model = Cls.from_pretrained(path, dtype=dtype, device_map=device_map or f"cuda:{gpu}")
    model.eval()
    if use_fast:
        processor = AutoProcessor.from_pretrained(path)
    else:
        # transformers >=5 loads the fast image processor by default even when
        # the checkpoint shipped a slow one, which changes the preprocessed
        # pixels.  The reference IJAR runs predate that switch, so default to
        # the slow processor to keep inputs comparable.
        try:
            processor = AutoProcessor.from_pretrained(path, use_fast=False)
            print("  processor: slow (use_fast=False)")
        except Exception as exc:  # pragma: no cover - depends on checkpoint
            print(f"  processor: use_fast=False unavailable ({exc}); using default")
            processor = AutoProcessor.from_pretrained(path)

    # transformers >=5 sets the processor's chat_template only from the
    # checkpoint's chat_template.json.  Qwen2.5-VL-32B ships the (byte-identical)
    # template inside tokenizer_config.json instead, so the processor comes back
    # without one and apply_chat_template() raises.  Fall back to the tokenizer's.
    if getattr(processor, "chat_template", None) is None:
        tmpl = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
        if tmpl:
            processor.chat_template = tmpl
            print("  processor: chat_template restored from the tokenizer")
        else:
            raise RuntimeError(f"{path} has no chat template anywhere")
    return model, processor


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_dataset(name, limit=None, seed=42):
    cfg = DATASETS[name]
    with open(cfg["csv_path"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]
    return rows


def resolve_image(image_dir, image_id):
    for ext in ("", ".png", ".jpg", ".jpeg"):
        cand = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(cand):
            return cand
    import glob as _glob
    matches = sorted(_glob.glob(os.path.join(image_dir, f"{image_id}*")))
    return matches[0] if matches else os.path.join(image_dir, image_id)


# --------------------------------------------------------------------------
# One sample
# --------------------------------------------------------------------------

def build_messages(image_path, question):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]


def run_sample(model, processor, image_path, question, layer_weights, layer_means):
    from qwen_vl_utils import process_vision_info

    messages = build_messages(image_path, question)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    # With device_map="auto" the weights are sharded, so model.device is not
    # necessarily where the embedding table lives; use the embedding's device.
    input_device = model.get_input_embeddings().weight.device
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(input_device)

    with torch.inference_mode(), DownProjCapture(model) as cap:
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    phis = cap.phi_per_layer()
    per_layer_conflict, per_layer_ignorance, per_token_conflict = [], [], []
    for i, phi in enumerate(phis):
        if phi is None:
            per_layer_conflict.append(float("nan"))
            per_layer_ignorance.append(float("nan"))
            per_token_conflict.append([])
            continue
        from acf_core import acf_from_phi
        kappa, ignorance = acf_from_phi(layer_weights[i], layer_means[i], phi)
        per_layer_conflict.append(float(kappa.mean().item()))
        per_layer_ignorance.append(float(ignorance.mean().item()))
        per_token_conflict.append([float(v) for v in kappa.tolist()])

    # one captured step per forward call: prefill + one per generated token
    n_tokens = len(cap.buf[0])

    return {
        "model_output": output_text,
        "per_layer_conflict": per_layer_conflict,
        "per_layer_ignorance": per_layer_ignorance,
        "per_token_conflict": per_token_conflict,
        "n_tokens": n_tokens,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def result_files(model_name, dataset):
    stem = f"{model_name}_{dataset}"
    return (
        os.path.join(OUTPUT_BASE, stem + ".jsonl"),
        os.path.join(OUTPUT_BASE, stem + ".json"),
    )


def already_done(jsonl_path):
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["index"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--datasets", nargs="+", default=["STA", "RTA"], choices=sorted(DATASETS))
    ap.add_argument("--limit", type=int, default=None, help="first N samples per dataset")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--use-fast", action="store_true",
                    help="use the fast image processor (reference runs used the slow one)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device-map", default=None,
                    help='e.g. "auto" to shard across all visible GPUs (needed for 32B)')
    args = ap.parse_args()

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    t0 = time.time()

    print("=" * 62)
    print(f"  model      : {args.model}")
    print(f"  path       : {MODELS[args.model]['path']}")
    print(f"  dtype      : {args.dtype}")
    print(f"  datasets   : {', '.join(args.datasets)}")
    print(f"  limit      : {args.limit}")
    print(f"  output     : {OUTPUT_BASE}")
    print("=" * 62)

    print("\n[1/3] Loading model ...")
    model, processor = load_model(
        args.model, dtype, use_fast=args.use_fast, gpu=args.gpu,
        device_map=args.device_map,
    )
    layers = resolve_text_layers(model)
    n_layers = len(layers)
    print(f"  {n_layers} text decoder layers, device={model.device}")

    print("[2/3] Reading down_proj weights off the model (nothing saved) ...")
    layer_weights = [l.mlp.down_proj.weight for l in layers]
    layer_means = [precompute_layer_stats(w)[0] for w in layer_weights]
    J, Ki = layer_weights[0].shape
    print(f"  layer0 weight {tuple(layer_weights[0].shape)} dtype={layer_weights[0].dtype}")

    print("[3/3] Running datasets ...")
    for ds in args.datasets:
        cfg = DATASETS[ds]
        rows = load_dataset(ds, limit=args.limit)
        jsonl_path, json_path = result_files(args.model, ds)
        done = set() if args.no_resume else already_done(jsonl_path)
        if done:
            print(f"\n  [{ds}] resuming, {len(done)} samples already done")

        with open(jsonl_path, "a", encoding="utf-8") as fout:
            for row in tqdm(rows, desc=f"{args.model}/{ds}"):
                index = str(row[cfg["index_col"]])
                if index in done:
                    continue
                question = str(row["question"]).strip()
                ground_truth = str(row.get("answer", "")).strip()
                image_path = resolve_image(cfg["image_dir"], index)

                rec = run_sample(
                    model, processor, image_path, question, layer_weights, layer_means
                )
                pred = _extract_choice(rec["model_output"])
                gt = ground_truth.strip().lower()[:1]
                correct = (pred is not None and pred == gt)

                rec = {
                    "index": index,
                    "question": question,
                    "ground_truth": ground_truth,
                    "pred": pred,
                    "correct": bool(correct),
                    **rec,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

        # consolidate jsonl -> json array
        with open(jsonl_path, encoding="utf-8") as f:
            all_recs = [json.loads(line) for line in f if line.strip()]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_recs, f, ensure_ascii=False, indent=2)

        n = len(all_recs)
        n_ok = sum(1 for r in all_recs if r["correct"])
        n_none = sum(1 for r in all_recs if r["pred"] is None)
        print(
            f"\n  [{ds}] N={n} acc={n_ok / n:.4f} "
            f"unparsed={n_none} layers={len(all_recs[0]['per_layer_conflict'])}"
        )
        print(f"  -> {json_path}")

    meta = {
        "model": args.model,
        "path": MODELS[args.model]["path"],
        "params_b": MODELS[args.model]["params_b"],
        "family": MODELS[args.model]["family"],
        "n_layers": n_layers,
        "hidden_size": int(J),
        "intermediate_size": int(Ki),
        "dtype": args.dtype,
        "max_new_tokens": MAX_NEW_TOKENS,
        "datasets": args.datasets,
        "limit": args.limit,
        "generation_config": json.loads(model.generation_config.to_json_string()),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "env": sys.prefix,
    }
    meta_path = os.path.join(OUTPUT_BASE, f"{args.model}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nmeta -> {meta_path}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
