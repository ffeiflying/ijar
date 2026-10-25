"""
Wait for enough free GPU memory on this shared box, then launch a run.

Qwen3-VL-32B needs ~62.1 GiB just for the weights (fp16, 14 shards), so it only
fits when the other users' jobs on this machine happen to be small.  Rather than
fail, poll `torch.cuda.mem_get_info` and start the run as soon as it fits.

The probe runs in a *subprocess* so this waiter never holds a CUDA context of
its own (a resident context would steal ~300 MiB from the child).

On OOM the run resumes from its .jsonl (run_scaling.py skips finished indices),
so retrying is cheap.

Usage:
    /data/liuxiaofei/miniconda3/envs/EUQ/bin/python wait_for_gpu.py \
        --model Qwen3-VL-32B-Instruct --datasets STA RTA
    ... --dry-run          # just report what is free right now, never launch
"""

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = "/data/liuxiaofei/miniconda3/envs/EUQ/bin/python"

# A single card must have this much free to run un-sharded (fastest).
SINGLE_GIB = 68.0
# Otherwise shard across all cards, which needs this much free in total.
SHARD_GIB = 67.0

PROBE = (
    "import torch,json;"
    "print(json.dumps([torch.cuda.mem_get_info(i)[0]/2**30"
    " for i in range(torch.cuda.device_count())]))"
)


def free_per_gpu():
    """Free GiB per visible GPU, probed from a throwaway subprocess."""
    out = subprocess.check_output([PYTHON, "-c", PROBE], text=True).strip()
    import json
    return json.loads(out)


def describe(free):
    per = "  ".join(f"gpu{i} {g:.1f}" for i, g in enumerate(free))
    return f"{per}  |  total {sum(free):.1f} GiB free"


def pick(free):
    """Return the extra CLI args for run_scaling.py, or None if it does not fit."""
    if not free:
        return None
    best = max(range(len(free)), key=lambda i: free[i])
    if free[best] >= SINGLE_GIB:
        return ["--gpu", str(best)], f"single card gpu{best} ({free[best]:.1f} GiB)"
    if sum(free) >= SHARD_GIB:
        return ["--device-map", "auto"], f"sharded across {len(free)} cards"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=["STA", "RTA"])
    ap.add_argument("--poll", type=int, default=45, help="seconds between checks")
    ap.add_argument("--max-tries", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--logdir", default=os.path.join(_HERE, "logs"))
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, f"run_{args.model}.log")
    wait_log = os.path.join(args.logdir, f"wait_{args.model}.log")

    def note(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(wait_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    note(f"waiting for {args.model}; need {SINGLE_GIB:.0f} GiB on one card "
         f"or {SHARD_GIB:.0f} GiB total")

    if args.dry_run:
        free = free_per_gpu()
        note(describe(free))
        sel = pick(free)
        note(f"would {'launch' if sel else 'keep waiting'} "
             + (sel[1] if sel else ""))
        return 0

    tries = 0
    while tries < args.max_tries:
        free = free_per_gpu()
        sel = pick(free)
        if sel is None:
            note("not enough free memory: " + describe(free))
            time.sleep(args.poll)
            continue

        extra, why = sel
        tries += 1
        note(f"launching ({why}), attempt {tries}/{args.max_tries}")

        cmd = (
            [PYTHON, os.path.join(_HERE, "run_scaling.py"),
             "--model", args.model, "--datasets", *args.datasets, *extra]
        )
        env = dict(os.environ)
        env.update(
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            HF_HUB_OFFLINE="1",
            MPLBACKEND="Agg",
            PYTHONUNBUFFERED="1",
        )
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n{'=' * 70}\n[{time.strftime('%F %T')}] "
                      f"{' '.join(cmd)}\n{'=' * 70}\n")
            log.flush()
            rc = subprocess.call(cmd, cwd=_HERE, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)

        if rc == 0:
            note(f"run finished cleanly -> {log_path}")
            return 0
        note(f"run exited rc={rc}; will retry after {args.poll}s "
             f"(resumes from .jsonl)")
        time.sleep(args.poll)

    note("gave up after too many failed attempts")
    return 1


if __name__ == "__main__":
    sys.exit(main())
