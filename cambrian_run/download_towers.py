"""Download Cambrian-8B's four vision towers from hf-mirror into an IJAR-local cache.

Only downloads; writes nothing outside /data/liuxiaofei.
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
HF_HOME = "/data/liuxiaofei/liuxiaofei/IJAR/cambrian_run/hf_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")

from huggingface_hub import snapshot_download

REPOS = [
    # cambrian maps "siglip/CLIP-ViT-SO400M-14-384" -> this timm repo
    ("timm/ViT-SO400M-14-SigLIP-384",
     ["*.json", "*.txt", "*.py", "model.safetensors"]),
    ("openai/clip-vit-large-patch14-336", None),
    ("facebook/dinov2-giant",
     ["*.json", "*.txt", "model.safetensors"]),
    ("laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup",
     ["*.json", "*.txt", "open_clip_model.safetensors",
      "open_clip_pytorch_model.bin"]),
]

for repo, patterns in REPOS:
    print(f"\n=== {repo} ===", flush=True)
    try:
        p = snapshot_download(repo, allow_patterns=patterns)
        size = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(p) for f in fs
        )
        print(f"OK {p}  ({size / 1e9:.2f} GB)", flush=True)
    except Exception as e:  # keep going; report at the end
        print(f"FAIL {repo}: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
print("\nALL DONE", flush=True)
