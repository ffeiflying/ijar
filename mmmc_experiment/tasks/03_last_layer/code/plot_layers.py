import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

ROOT=TASK3 / "results"
os.environ['MPLCONFIGDIR']=str(CACHE/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
for idx,(layer,filename) in enumerate([(0,'internvl_pilot.jsonl'),(31,'internvl_last_pilot.jsonl')]):
    rows=[json.loads(x) for x in result_path(filename).read_text().splitlines()]
    y=np.array([r['label'] for r in rows]); s=np.array([r['acf'] for r in rows])
    bins=np.linspace(s.min(),s.max(),23)
    for label,name,color in [(0,'Normal','#277DA1'),(1,'Conflict','#F3722C')]:
        axs[idx].hist(s[y==label],bins=bins,alpha=.55,color=color,label=name)
    axs[idx].set(xlabel='ACF',ylabel='Questions',title=f'Layer {layer}')
    axs[idx].legend()
    fpr,tpr,_=roc_curve(y,s)
    axs[2].plot(fpr,tpr,label=f'L{layer}: AUROC {roc_auc_score(y,s):.3f}')
axs[2].plot([0,1],[0,1],'--',color='gray',lw=1)
axs[2].set(xlabel='False positive rate',ylabel='True positive rate',title='Identical 100 image pairs',xlim=(0,1),ylim=(0,1))
axs[2].legend()
fig.savefig(ROOT/'layer_comparison.png',dpi=180)
fig.savefig(ROOT/'layer_comparison.pdf')
