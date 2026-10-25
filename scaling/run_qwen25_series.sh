#!/bin/bash
# Finish the Qwen scaling ladder, strictly serially.
#
# Running the 3B and the 32B at the same time previously blew up the 3B with a
# CUDA device-side assert (the 32B's sharded load starved it of memory), so the
# 3B goes first and the 32B's memory waiter only starts once the 3B is done.
#
# Both stages are resumable: run_scaling.py skips indices already in its .jsonl.

set -u
cd "$(dirname "$0")"

PY=/data/liuxiaofei/miniconda3/envs/EUQ/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "[$(date '+%F %T')] stage 1/2: Qwen2.5-VL-3B"
$PY run_scaling.py --model Qwen2.5-VL-3B-Instruct --datasets STA RTA --gpu "${THREE_B_GPU:-1}" \
    > logs/run_qwen25vl_3b.log 2>&1
echo "[$(date '+%F %T')] stage 1 exit=$?"

echo "[$(date '+%F %T')] stage 2/2: Qwen2.5-VL-32B (waits for free memory)"
$PY wait_for_gpu.py --model Qwen2.5-VL-32B-Instruct --datasets STA RTA \
    > logs/wait_console_32b25.log 2>&1
echo "[$(date '+%F %T')] stage 2 exit=$?"

echo "[$(date '+%F %T')] all stages done"
