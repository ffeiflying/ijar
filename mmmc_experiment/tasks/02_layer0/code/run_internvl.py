"""Isolated MMMC layer-selectable ACF pilot; uses existing internvl env and local model."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

import sys
import time

ROOT = TASK2 / "results"
# Keep any HF dynamic-code cache writes within this experiment, not the environment.
os.environ['HF_MODULES_CACHE'] = str(CACHE / 'hf_modules')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(PROJECT))
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0,str(TASK5/'code'))
from evidence_v2 import EvidenceModel

MODEL_PATH = '/data/public/models/InternVL/InternVL2_5-8B/models--OpenGVLab--InternVL2_5-8B/snapshots/e9e4c0dc1db56bfab10458671519b7fa3dd29463'
WEIGHT = '/data/liuxiaofei/uncertainty/InternVL/weight/internvl_MLP_Layer0_down_proj.pth'

spec = importlib.util.spec_from_file_location('existing_internvl_helpers', PROJECT / 'layerauroc/run_internvl.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--output')
    parser.add_argument('--feature-mean', help='Fixed development/calibration mean .npy; absent uses explicit zero-mean gauge')
    parser.add_argument('--layer', type=int, default=0, help='Decoder block index; -1 selects final block')
    args = parser.parse_args()
    records = json.loads((DATA / 'pilot.json').read_text())
    if args.limit:
        records = records[:args.limit]
    out = (TASK2 if args.layer == 0 else TASK3) / 'results' / (args.output or ('internvl_v2_pilot.jsonl' if args.layer == 0 else f'internvl_v2_layer{args.layer}_pilot.jsonl'))
    completed = {}
    if out.exists():
        completed = {r['row_id']:r for r in map(json.loads, out.read_text().splitlines())}
    if all(row['row_id'] in completed for row in records):
        print('All requested records already completed:', out)
        return
    torch.manual_seed(42)
    assert torch.cuda.is_available(), 'Run outside sandbox to access GPU'
    model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True, local_files_only=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False, local_files_only=True)
    layers = model.language_model.model.layers
    layer = args.layer if args.layer >= 0 else len(layers) + args.layer
    assert 0 <= layer < len(layers), layer
    down_proj = layers[layer].feed_forward.w2
    weight_path = WEIGHT.replace('Layer0_', f'Layer{layer}_')
    beta = torch.load(weight_path, map_location='cuda')
    assert beta.shape == down_proj.weight.shape, (beta.shape, down_proj.weight.shape)
    assert torch.allclose(beta.float(), down_proj.weight.float(), atol=0.01, rtol=0.01), f'Saved evidence weight mismatches model layer {layer}'
    feature_mean = __import__('numpy').load(args.feature_mean) if args.feature_mean else None
    evidence = EvidenceModel(beta, feature_mean=feature_mean)
    config = {'evidence_version':EvidenceModel.VERSION, 'feature_mean':args.feature_mean, 'model':MODEL_PATH, 'layer':layer, 'module':f'language_model.model.layers.{layer}.feed_forward.w2', 'weight':weight_path, 'dtype':'bfloat16', 'max_new_tokens':args.max_new_tokens, 'do_sample':False, 'max_image_tiles':1, 'seed':42, 'aggregation':f'mean over last-position layer{layer} MLP inputs for each generation forward step, including prefill predicting first token', 'pilot_sha256':hashlib.sha256((DATA/'pilot.json').read_bytes()).hexdigest(), 'python':sys.executable, 'torch':torch.__version__, 'output':str(out)}
    config_path = out.with_suffix('.config.json')
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Resume configuration differs'
    else:
        config_path.write_text(json.dumps(config, indent=2))
    captured = []
    def capture(module, inputs):
        captured.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1])[-1:].clone())
    handle = down_proj.register_forward_pre_hook(capture)
    with out.open('a', buffering=1) as f, torch.inference_mode():
        for row in records:
            if row['row_id'] in completed:
                continue
            captured.clear()
            start = time.time()
            pixels = helpers.load_image(str(resolve_record_path(row['image_path'])), max_num=1).to(dtype=torch.bfloat16, device='cuda')
            returned = model.chat(tokenizer, pixels, row['question'], {'max_new_tokens':args.max_new_tokens, 'do_sample':False})
            response = returned[0] if isinstance(returned, tuple) else returned
            assert captured, f'Layer{layer} hook did not execute'
            # Independent hook avoids relying on patched generation layer indexing.
            scores, ignorance = [], []
            for phi in captured:
                evidence.get_evidence_weights(phi.T)
                scores.append(evidence.get_evidence_conflict().item())
                ignorance.append(evidence.get_evidence_ignorance().item())
            assert all(math.isfinite(x) for x in scores + ignorance), 'Nonfinite evidence'
            result = {**row, 'response':response, 'evidence_version':EvidenceModel.VERSION, 'layer':layer, 'acf':sum(scores)/len(scores), 'ignorance':sum(ignorance)/len(ignorance), 'token_acf':scores, 'n_forward_steps':len(scores), 'seconds':time.time()-start}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(json.dumps({k:result[k] for k in ['row_id','label','acf','n_forward_steps','seconds','response']}, ensure_ascii=False), flush=True)
            del returned
    handle.remove()
    print('DONE', str(out), flush=True)
if __name__ == '__main__':
    main()
