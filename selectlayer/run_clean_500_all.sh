#!/usr/bin/env bash
# ============================================================================
# Run clean-set layerwise ACF on ALL 4 models, serially (N=500 each)
# ============================================================================
# Each model needs its own conda env (different torch/transformers). This
# driver calls each env's python directly (conda run misresolves here), so
# you can just:  bash run_clean_500_all.sh
#
# After it finishes, regenerate the four-line figures with:
#   /data/liuxiaofei/miniconda3/envs/qwen/bin/python four_line_plot_zoom.py
# ============================================================================
set -e
# HF mirror for faster model/tokenizer loading (env inherits to all children)
export HF_ENDPOINT=https://hf-mirror.com
BASE=/data/liuxiaofei/liuxiaofei/IJAR/selectlayer
QENV=/data/liuxiaofei/miniconda3/envs/qwen/bin/python
IENV=/data/liuxiaofei/miniconda3/envs/internvl/bin/python
MENV=/data/liuxiaofei/miniconda3/envs/mmvp/bin/python
DENV=/data/liuxiaofei/miniconda3/envs/deepseek/bin/python

echo "======================================================================"
echo "[1/4] Qwen2.5-VL-7B   clean N=500   (env: qwen)"
echo "======================================================================"
# $QENV "$BASE/run_clean_pilot_qwen.py" 500\


echo "======================================================================"
echo "[2/4] InternVL2.5-8B  clean N=500   (env: internvl)"
echo "======================================================================"
$IENV "$BASE/run_clean_pilot_internvl.py" 500

echo "======================================================================"
echo "[3/4] MoF-Models-7B   clean N=500   (env: mmvp)"
echo "======================================================================"
$MENV "$BASE/run_clean_pilot_mof.py" 500

echo "======================================================================"
echo "[4/4] DeepSeek-VL2-Tiny clean N=500 (env: deepseek)"
echo "======================================================================"
$DENV "$BASE/run_clean_pilot_deepseek.py" 500

echo ""
echo "All 4 models done. Results in:"
echo "  results/selectlayer/clean_layerauroc_{qwen,internvl,mof,deepseek}.json"
echo ""
echo "Regenerate figures:"
echo "  $QENV $BASE/four_line_plot_zoom.py"
