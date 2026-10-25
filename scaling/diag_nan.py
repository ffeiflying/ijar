"""
Why does `_sample` see a probability tensor with inf/NaN on STA index 18?

Runs the exact same preprocessing path as run_scaling.py but stops before
generate(), so a bad logit cannot take the process down.  Prints, for the
last (next-token) position:

  * whether the fp16 logits themselves are finite
  * what `logits / temperature` does in fp16 vs fp32
    (Qwen2.5-VL ships do_sample=True with temperature=1e-6, i.e. a x1e6
     rescale -- harmless in fp32, an overflow in fp16)

Usage: python diag_nan.py [--row 18] [--models 3B 7B]
"""

import argparse
import csv
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from run_scaling import (  # noqa: E402
    DATASETS, MODELS, resolve_image, build_messages, load_model,
)

SHORT = {
    "Qwen2.5-VL-3B-Instruct": "3B",
    "Qwen2.5-VL-7B-Instruct": "7B",
    "Qwen2.5-VL-32B-Instruct": "32B",
}
BY_SHORT = {v: k for k, v in SHORT.items()}


def build_inputs(model, processor, image_path, question):
    from qwen_vl_utils import process_vision_info

    messages = build_messages(image_path, question)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    dev = model.get_input_embeddings().weight.device
    return processor(text=[text], images=image_inputs, videos=video_inputs,
                     padding=True, return_tensors="pt").to(dev)


def report(tag, logits, temperature):
    last = logits[0, -1].float()
    fin = torch.isfinite(last)
    print(f"  [{tag}] logits dtype={logits.dtype} shape={tuple(logits.shape)}")
    print(f"    finite: {bool(fin.all())}   n_nonfinite={int((~fin).sum())}")
    if fin.any():
        print(f"    max={last[fin].max().item():.4f}  min={last[fin].min().item():.4f}  "
              f"absmax={last[fin].abs().max().item():.4f}")
        k = min(5, int(fin.sum()))
        top = torch.topk(last[fin], k)
        print(f"    top{k} values={[round(v, 3) for v in top.values.tolist()]} "
              f"ids={top.indices.tolist()}")

    if temperature is None:
        return
    fp16 = (logits[0, -1] / temperature)
    fp32 = (logits[0, -1].float() / temperature)
    print(f"    /temp({temperature:g}) in fp16: "
          f"n_inf={int(torch.isinf(fp16).sum())} n_nan={int(torch.isnan(fp16).sum())}  "
          f"| fp32: n_inf={int(torch.isinf(fp32).sum())} "
          f"n_nan={int(torch.isnan(fp32).sum())}")
    for nm, x in (("fp16", fp16), ("fp32", fp32)):
        p = torch.softmax(x, dim=-1)
        print(f"      softmax({nm}) finite={bool(torch.isfinite(p).all())} "
              f"min={p.min().item():.4g} sum={p.sum().item():.4g}")


def scan_hidden(hs, tag):
    """First layer whose hidden states go non-finite -- locates the overflow."""
    print(f"  [{tag}] per-layer hidden-state finite scan:")
    bad = None
    for i, h in enumerate(hs):
        d = h.float()
        n_bad = int((~torch.isfinite(d)).sum())
        amax = d[torch.isfinite(d)].abs().max().item() if n_bad < d.numel() else float("nan")
        flag = "" if n_bad == 0 else "   <== NaN/Inf"
        if n_bad or i in (0, len(hs) - 1) or (bad is not None and i <= bad + 2):
            print(f"    h[{i:2d}] dtype={h.dtype} absmax={amax:.4g} n_bad={n_bad}{flag}")
        if n_bad and bad is None:
            bad = i
    if bad is None:
        print("    -> all layers finite")
    else:
        print(f"    -> first bad layer: {bad}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="STA")
    ap.add_argument("--row", type=int, default=18)
    ap.add_argument("--models", nargs="+", default=["3B", "7B"])
    ap.add_argument("--dtypes", nargs="+", default=["float16"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-generate", action="store_true",
                    help="skip the greedy decode (it dies on an all-NaN logits row)")
    ap.add_argument("--fp32-vision", action="store_true",
                    help="keep the LLM in --dtypes but run the visual tower in fp32 "
                         "(modeling casts pixel_values to visual.dtype and casts the "
                         "image embeds back to the LLM dtype before masked_scatter)")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    rows = list(csv.DictReader(open(cfg["csv_path"], encoding="utf-8")))
    row = rows[args.row]
    index = str(row[cfg["index_col"]])
    image_path = resolve_image(cfg["image_dir"], index)
    print(f"dataset={args.dataset} row={args.row} index={index}")
    print(f"image={image_path}")
    print(f"question={row['question'].strip()[:120]}")

    dtype = None
    for short in args.models:
        name = BY_SHORT[short]
        for dt in args.dtypes:
            dtype = getattr(torch, dt)
            print(f"\n{'='*60}\n{name} ({dt})\n{'='*60}")
            model, processor = load_model(name, dtype, use_fast=False, gpu=args.gpu)
            gc = model.generation_config
            print(f"  generation_config: do_sample={gc.do_sample} "
                  f"temperature={gc.temperature} top_p={gc.top_p} top_k={gc.top_k} "
                  f"rep_penalty={gc.repetition_penalty}")

            inputs = build_inputs(model, processor, image_path,
                                  row["question"].strip())
            n_img = int((inputs.input_ids == model.config.image_token_id).sum())
            grid = getattr(inputs, "image_grid_thw", None)
            print(f"  input tokens={int(inputs.input_ids.shape[1])} "
                  f"visual tokens={n_img} "
                  f"grid={grid.tolist() if grid is not None else None}")

            if args.fp32_vision:
                visual = getattr(model, "visual", None) or model.model.visual
                visual.to(torch.float32)
                visual_last = None
                def _grab(module, inp, out):
                    nonlocal visual_last
                    visual_last = out
                hook = visual.register_forward_hook(_grab)
                print(f"  [fp32-vision] {type(visual).__name__} cast to float32")

            with torch.inference_mode():
                out = model(**inputs, output_hidden_states=True)
            report("forward", out.logits, gc.temperature)
            scan_hidden(out.hidden_states, f"{short}/{dt}")
            if args.fp32_vision:
                hook.remove()
                feats = visual_last if isinstance(visual_last, torch.Tensor) else \
                    visual_last.last_hidden_state
                f = feats.float()
                print(f"  [fp32-vision] visual features dtype={feats.dtype} "
                      f"shape={tuple(feats.shape)} absmax={f.abs().max().item():.4g} "
                      f"finite={bool(torch.isfinite(f).all())}  "
                      f"(fp16 max is 65504 -- a cast here overflows if absmax exceeds it)")

            if not args.no_generate:
                finite = bool(torch.isfinite(out.logits[0, -1]).all())
                if finite:
                    with torch.inference_mode():
                        gen = model.generate(**inputs, max_new_tokens=8,
                                             do_sample=False)
                    print("  greedy decode: "
                          + str(processor.batch_decode(
                              gen[:, inputs.input_ids.shape[1]:],
                              skip_special_tokens=False)))
                else:
                    print("  greedy decode: skipped (logits non-finite)")

            del model, out, inputs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
