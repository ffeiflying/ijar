import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

ROOT = TASK2 / "results"
os.environ['MPLCONFIGDIR'] = str(CACHE / 'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
rows = [json.loads(x) for x in (ROOT / 'internvl_pilot.jsonl').read_text().splitlines()]
y = np.array([r['label'] for r in rows])
s = np.array([r['acf'] for r in rows])
fpr, tpr, _ = roc_curve(y,s)
fig, ax = plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
bins = np.linspace(s.min(), s.max(), 23)
for label, name, color in [(0,'Normal','#277DA1'),(1,'Conflict','#F3722C')]:
    ax[0].hist(s[y==label], bins=bins, alpha=.55, label=name,color=color)
ax[0].set(xlabel='Layer 0 ACF', ylabel='Questions',title='MMMC paired pilot (100 images)')
ax[0].legend()
ax[1].plot(fpr,tpr,label=f'AUROC = {roc_auc_score(y,s):.3f}',color='#277DA1')
ax[1].plot([0,1],[0,1],ls='--',color='gray',lw=1)
ax[1].set(xlabel='False positive rate',ylabel='True positive rate',title='Higher ACF predicts conflict',xlim=(0,1),ylim=(0,1))
ax[1].legend()
fig.savefig(ROOT/'internvl_pilot.png',dpi=180)
fig.savefig(ROOT/'internvl_pilot.pdf')
