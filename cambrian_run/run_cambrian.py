"""Cambrian-8B — typographic-attack ACC + per-layer ACF (IJAR).

ACF definition is identical to the four existing models (qwen / internvl / mof /
deepseek): for every generated token, at the INPUT of each transformer block's
MLP down-projection (i.e. act_fn(gate_proj(x)) * up_proj(x)), we compute the
Dempster-Shafer conflict kappa with envidence.py, then average kappa over the
generated tokens -> one scalar per block per sample.

Differences from the four older runners (deliberate):
  * The four older models each needed a patched transformers `generation/utils.py`
    plus patched modeling code inside their own conda env. We touch no env: the
    down-proj inputs are captured with forward pre-hooks from this script only.
  * The per-layer down-projection weight is read live off the loaded model
    (module.weight) instead of being pre-saved to *.pth files.

Usage:
    python run_cambrian.py --n-sta 10 --n-clean 10 --out-suffix pilot
"""

import argparse
import faulthandler
import json
import os
import sys

# Print a stack trace every 5 min so a hang is diagnosable instead of silent.
faulthandler.dump_traceback_later(300, repeat=True)

VENDOR = "/data/liuxiaofei/liuxiaofei/IJAR/cambrian_run/vendor/site-packages"
CAMBRIDGE_SRC = "/data/liuxiaofei/liuxiaofei/IJAR/cambrian_src/cambrian"
IJAR_DIR = "/data/liuxiaofei/liuxiaofei/IJAR"
HF_CACHE = "/data/liuxiaofei/liuxiaofei/IJAR/cambrian_run/hf_cache"

# The shell already exports HF_HOME/HUGGINGFACE_HUB_CACHE pointing at the user's
# global cache; override (not setdefault) so the towers are read from the IJAR
# cache we populated, and nothing lands outside /data/liuxiaofei.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
# transformers 4.49 resolves from_pretrained() through its OWN TRANSFORMERS_CACHE
# constant, which beats HF_HUB_CACHE. The shell exports it to the global cache,
# where dinov2-giant is absent -> "resolved_archive_file is None". Override it.
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "hub")
# Every tower is already in the cache above; going offline avoids the Hub
# revision checks that hang on this network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# The older runners drop user-local site-packages to avoid a CUDA mismatch.
for _p in ("~/.local/lib/python3.10/site-packages",
           "~/.local/lib/python3.9/site-packages",
           "~/.local/lib/python3.8/site-packages"):
    _p = os.path.expanduser(_p)
    while _p in sys.path:
        sys.path.remove(_p)

# vendor first (our own open_clip/ezcolorlog), then the cambrian repo copy,
# then IJAR for envidence.py.
for _p in (IJAR_DIR, CAMBRIDGE_SRC, VENDOR):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, IJAR_DIR)
sys.path.insert(0, CAMBRIDGE_SRC)
sys.path.insert(0, VENDOR)

import torch  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from cambrian.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from cambrian.conversation import conv_templates  # noqa: E402
from cambrian.mm_utils import (  # noqa: E402
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from cambrian.model.builder import load_pretrained_model  # noqa: E402

from envidence import EvidenceModel  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "/data/public/models/cambrian-8b"
NUM_LAYERS = 32
CONV_MODE = "llama_3"          # cambrian-8b, per the official inference.py
DATASET_BASE = "/data/liuxiaofei/liuxiaofei/IJAR/datasets"
CLEAN_BASE = "/data/liuxiaofei/liuxiaofei/IJAR/typo-parameter/clean"
OUT_DIR = "/data/liuxiaofei/liuxiaofei/IJAR/results/cambrian"
DATASETS = {
    "STA": {"image_dir": f"{DATASET_BASE}/STA/images",
            "csv_path": f"{DATASET_BASE}/STA/sta_mini.csv"},
    "clean": {"image_dir": f"{CLEAN_BASE}/images",
              "csv_path": f"{CLEAN_BASE}/data.csv"},
}
MAX_NEW_TOKENS = 128
RANDOM_SEED = 42


def resolve_image_path(image_dir, image_id):
    for ext in (".png", ".jpg", ".jpeg"):
        cand = os.path.join(image_dir, image_id)
        if not image_id.lower().endswith(ext):
            cand = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(cand):
            return cand
    import glob as _glob
    matches = _glob.glob(os.path.join(image_dir, f"{image_id}*"))
    return matches[0] if matches else os.path.join(image_dir, image_id)


# ---------------------------------------------------------------------------
# ACF capture
# ---------------------------------------------------------------------------
class DownProjACF:
    """Forward pre-hooks on every block's MLP down_proj -> per-token kappa.

    Each model forward call (one generated token) fires the hooks in block order
    0..NUM_LAYERS-1; the conflicts of one pass are grouped into one token record,
    mirroring the old runners' `all_input_down_proj` per-token lists.
    """

    def __init__(self, model, num_layers, device, dtype=torch.float32):
        self.num_layers = num_layers
        self.steps = []          # list per generated token of [kappa per layer]
        self._cur = []
        self.handles = []
        self.evid = []
        blocks = model.model.layers
        assert len(blocks) == num_layers, f"got {len(blocks)} blocks"
        for i in range(num_layers):
            w = blocks[i].mlp.down_proj.weight
            self.evid.append(EvidenceModel(w.detach().to(device=device,
                                                           dtype=dtype)))
            self.handles.append(
                blocks[i].mlp.down_proj.register_forward_pre_hook(self._make(i))
            )

    def _make(self, i):
        def hook(module, args):
            x = args[0]
            if x.dim() == 3:
                x = x[:, -1:, :]
            elif x.dim() == 2:
                x = x[-1:, :]
            ev = self.evid[i]
            ev.get_evidence_weights(x.squeeze(0).T)
            k = ev.get_evidence_conflict().item()
            if i == 0:
                self._cur = []          # a new forward pass starts at block 0
            self._cur.append(k)
            if i == self.num_layers - 1:
                self.steps.append(self._cur)
        return hook

    def reset(self):
        self.steps = []
        self._cur = []

    def per_layer_conflict(self):
        n = len(self.steps)
        out = [0.0] * self.num_layers
        for step in self.steps:
            for l, k in enumerate(step):
                out[l] += k
        return [v / n for v in out] if n else out

    def _mean_steps(self, upto):
        """Mean kappa per layer over the first `upto` generated tokens."""
        steps = self.steps[:upto]
        if not steps:
            return [0.0] * self.num_layers
        out = [0.0] * self.num_layers
        for step in steps:
            for l, k in enumerate(step):
                out[l] += k
        return [v / len(steps) for v in out]

    def first_token_conflict(self):
        """Kappa at the very first generated token -- the decision token.

        Cambrian-8B emits ~100-token captions, so averaging kappa over every
        generated token dilutes the signal ~100x relative to the four
        letter-emitting models (1-3 tokens). Token 0 is the like-for-like
        analogue of what those models average over.
        """
        return list(self.steps[0]) if self.steps else [0.0] * self.num_layers

    def first_k_conflict(self, k=5):
        return self._mean_steps(k)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# Prompt building / generation (mirrors cambrian/inference.py)
# ---------------------------------------------------------------------------
def build_inputs(image, question, tokenizer, image_processor, model_config):
    if model_config.mm_use_im_start_end:
        from cambrian.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_sizes = [image.size]
    image_tensor = process_images([image], image_processor, model_config)
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    return input_ids, image_tensor, image_sizes


def run_dataset(name, model, tokenizer, image_processor, acf, n_samples, seed):
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["csv_path"])
    if n_samples is not None:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    print(f"  [{name}] {len(df)} samples", flush=True)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=name):
        image_id = str(row["index"])
        question = str(row["question"]).strip()
        gt = str(row.get("answer", "")).strip()
        image_path = resolve_image_path(cfg["image_dir"], image_id)
        image = Image.open(image_path).convert("RGB")

        input_ids, image_tensor, image_sizes = build_inputs(
            image, question, tokenizer, image_processor, model.config
        )
        input_ids = input_ids.to(device="cuda", non_blocking=True)

        acf.reset()
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=None,
                top_p=None,
                num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
            )
        text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        per_layer = acf.per_layer_conflict()
        first = acf.first_token_conflict()
        first5 = acf.first_k_conflict(5)

        cf_str = " | ".join(
            f"L{l}:{per_layer[l]:.6f}"
            for l in (0, NUM_LAYERS // 4, NUM_LAYERS // 2,
                      3 * NUM_LAYERS // 4, NUM_LAYERS - 1)
        )
        print(f"  [{image_id}] tok={len(acf.steps)} {cf_str} out={text[:50]!r}",
              flush=True)

        results.append({
            "index": image_id,
            "question": question,
            "ground_truth": gt,
            "model_output": text,
            # mean kappa over ALL generated tokens (kept: what the first run
            # produced, and what the four letter-emitting models effectively
            # compute since they only generate 1-3 tokens)
            "per_layer_conflict": per_layer,
            # decision-token statistics, to undo the ~100x averaging dilution
            "per_layer_conflict_first": first,
            "per_layer_conflict_first5": first5,
            "n_tokens": len(acf.steps),
        })
    return results


def main():
    global MAX_NEW_TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sta", type=int, default=None)
    ap.add_argument("--n-clean", type=int, default=None)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    MAX_NEW_TOKENS = args.max_new_tokens

    print(f"Loading {MODEL_PATH} ...", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, get_model_name_from_path(MODEL_PATH),
        device_map="cuda:0", device="cuda",
    )
    model.eval()
    dtype = next(model.parameters()).dtype
    print(f"  loaded, dtype={dtype}", flush=True)

    acf = DownProjACF(model, NUM_LAYERS, device="cuda:0",
                      dtype=torch.float32)
    print(f"  ACF hooks on {NUM_LAYERS} blocks", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, n in (("STA", args.n_sta), ("clean", args.n_clean)):
        if n == 0:
            continue
        res = run_dataset(name, model, tokenizer, image_processor, acf,
                          n, RANDOM_SEED)
        sfx = f"_{args.out_suffix}" if args.out_suffix else ""
        path = os.path.join(OUT_DIR, f"{name}_layerauroc_cambrian{sfx}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"  saved -> {path}", flush=True)

    acf.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
