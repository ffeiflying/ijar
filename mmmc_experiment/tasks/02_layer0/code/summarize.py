"""Evaluate fixed high-ACF-positive detection, with image-pair bootstrap."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
ROOT = TASK2 / "results"
parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, default=0)
parser.add_argument('--input', default='internvl_pilot.jsonl')
args = parser.parse_args()
p = result_path(args.input)
rows = [json.loads(line) for line in p.read_text().splitlines()]
assert len({r['row_id'] for r in rows}) == len(rows)
assert all(r['layer'] == args.layer and np.isfinite(r['acf']) for r in rows)
groups = defaultdict(dict)
for r in rows:
    assert r['label'] not in groups[r['image_id']], 'Duplicate label in image pair'
    groups[r['image_id']][r['label']] = r
assert all(set(g) == {0,1} for g in groups.values()), 'Incomplete image pairs'
pairs = np.array([[g[0]['acf'],g[1]['acf']] for g in groups.values()])
y = np.tile([0,1], len(pairs))
scores = pairs.ravel()
fpr, tpr, thresholds = roc_curve(y, scores)
report = {'rows':len(rows), 'image_pairs':len(pairs), 'layer':args.layer, 'score_direction':'higher ACF predicts modality conflict (fixed before seeing results)', 'auroc':float(roc_auc_score(y,scores)), 'auprc':float(average_precision_score(y,scores)), 'tpr_at_fpr_le_0.05':float(max(tpr[fpr<=.05])), 'mean_clean':float(pairs[:,0].mean()), 'mean_conflict':float(pairs[:,1].mean()), 'mean_paired_delta':float((pairs[:,1]-pairs[:,0]).mean()), 'paired_conflict_higher_fraction':float((pairs[:,1]>pairs[:,0]).mean()), 'response_accuracy':'Not measured: this experiment evaluates conflict detection, not answer grading', 'by_type':{}}
rng = np.random.default_rng(42)
bootstrap = []
for _ in range(1000):
    sample = pairs[rng.integers(0,len(pairs),len(pairs))]
    bootstrap.append([roc_auc_score(y,sample.ravel()), (sample[:,1]-sample[:,0]).mean()])
for i,name in enumerate(['auroc','mean_paired_delta']):
    report[name+'_95ci'] = np.quantile(np.array(bootstrap)[:,i],[.025,.975]).tolist()
for kind in sorted({r['conflict_type'] for r in rows if r['label']}):
    relevant = [g for g in groups.values() if g[1]['conflict_type']==kind]
    sub = np.array([[g[0]['acf'],g[1]['acf']] for g in relevant])
    report['by_type'][kind] = {'pairs':len(sub),'auroc':float(roc_auc_score(np.tile([0,1],len(sub)),sub.ravel())),'mean_paired_delta':float((sub[:,1]-sub[:,0]).mean())}
output = p.with_suffix('.summary.json')
output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
