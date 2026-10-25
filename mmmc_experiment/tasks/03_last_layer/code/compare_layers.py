"""Validate identical layer0/final pilot and compare fixed-score AUROC by image."""
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = TASK3 / "results"
original = json.loads((DATA/'pilot.json').read_text())
results = {}
for layer, filename in [(0,'internvl_pilot.jsonl'), (31,'internvl_last_pilot.jsonl')]:
    rows = [json.loads(x) for x in result_path(filename).read_text().splitlines()]
    assert len(rows)==len(original)==200
    by_id = {r['row_id']:r for r in rows}
    assert len(by_id)==200
    for source in original:
        row = by_id[source['row_id']]
        assert all(row[k]==v for k,v in source.items())
        assert row['layer']==layer and np.isfinite(row['acf'])
        assert len(row['token_acf'])==row['n_forward_steps']>0
        assert abs(row['acf']-np.mean(row['token_acf']))<1e-12
    results[layer] = by_id
groups = defaultdict(list)
for row in original:
    groups[row['image_id']].append(row['row_id'])
pairs = [[i for i in ids] for ids in groups.values()]
assert all(len(pair)==2 for pair in pairs)
y = np.array([results[0][i]['label'] for pair in pairs for i in pair])
scores = {layer:np.array([[rs[i]['acf'] for i in pair] for pair in pairs]) for layer,rs in results.items()}
delta = roc_auc_score(y,scores[31].ravel())-roc_auc_score(y,scores[0].ravel())
rng = np.random.default_rng(42)
boot=[]
for _ in range(1000):
    idx = rng.integers(0,len(pairs),len(pairs))
    boot.append(roc_auc_score(y,scores[31][idx].ravel())-roc_auc_score(y,scores[0][idx].ravel()))
report = {'rows_each':200,'layers':[0,31],'source_records_exact_match':True,'all_scores_finite':True,'aggregation_verified':True,'matching_responses':sum(results[0][i]['response']==results[31][i]['response'] for i in results[0]),'matching_forward_counts':sum(results[0][i]['n_forward_steps']==results[31][i]['n_forward_steps'] for i in results[0]),'auroc_last_minus_first':float(delta),'auroc_delta_95ci':np.quantile(boot,[.025,.975]).tolist(),'bootstrap':'1000 resamples of identical image pairs across both layers, seed42'}
(ROOT/'layer_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
