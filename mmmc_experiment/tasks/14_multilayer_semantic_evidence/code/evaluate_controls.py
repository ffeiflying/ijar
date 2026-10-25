"""Frozen-method mechanism audit on balanced image/question crossed controls."""
import collections
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
TASK=Path(__file__).resolve().parent.parent
dest=TASK/'results/controls'
rows=json.loads((dest/'balanced_records.json').read_text())
values=[]
for row in rows:
    with np.load(dest/f'features/{row["row_id"]}.npz') as z:
        assert int(z['language_forward_count'])==1
        values.append({k:float(z[k]) for k in z.files})
score={k:np.array([v[k] for v in values]) for k in values[0]}
frozen=json.loads((TASK/'results/frozen_dev_selection.json').read_text())
keys=[frozen['best_evidence'],frozen['best_matching_baseline']]
keys += [s['key'] for name,s in frozen['selections'].items() if name.endswith('source_DS')]
methods={k:score[k] for k in keys}
sys.path.insert(0,str(TASK.parent/'15_calibrated_internal_evidence/code'))
from calibrate import predict, fuse
cal=json.loads((TASK.parent/'15_calibrated_internal_evidence/results/frozen_calibrators.json').read_text())
key=cal['selected']
group,discount,temp,statistic=key.split('/')
model=cal['calibrators'][group]
ratios=np.column_stack([predict(e['linear'],np.column_stack([score[k] for k in e['keys']])) for e in model['experts']])
methods['calibrated/'+key]=fuse(ratios,model['reliability'][discount],float(temp[1:]))[statistic]
result={}
y=np.array([r['label'] for r in rows])
groups=collections.defaultdict(dict)
for i,row in enumerate(rows):
    groups[row['question_group']][row['label']]=i
for name,s in methods.items():
    deltas=[float(s[ix[1]]-s[ix[0]]) for ix in groups.values()]
    result[name]={'auroc':float(roc_auc_score(y,s)),'question_paired_positive_deltas':sum(v>0 for v in deltas),'question_pairs':len(deltas),'per_concept_delta':{rows[ix[0]]['concept']:float(s[ix[1]]-s[ix[0]]) for ix in groups.values()}}
report={'n_rows':len(rows),'n_questions':8,'n_images':8,'labels':'visually checked synthetic false-premise controls, not transferred official labels','single_forward_verified_for_every_row':True,'development_images_used':'mechanism check only, not independent generalization','methods':result}
(dest/'metrics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
