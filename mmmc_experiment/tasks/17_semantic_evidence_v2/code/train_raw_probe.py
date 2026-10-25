"""Full-state linear probe control; official train/calibration selection only."""
import json,pickle,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from train_mil import records,TASK,unit
OUT=TASK/'results/raw_probe'

def load(split,layer):
    path=OUT/f'{split}_L{layer}.npz'
    rows=records(split)
    if path.exists():
        with np.load(path) as z:return rows,z['x'],z['y']
    values=[]
    for row in rows:
        with np.load(TASK/f'results/features/{split}/{row["row_id"]}.npz') as z:
            values.append(np.concatenate([unit(z[f'L{layer}_{k}']) for k in ['qpost_pred','qpost_mean','image_mean']]))
    x=np.stack(values);y=np.array([r['label'] for r in rows])
    np.savez_compressed(path,x=x,y=y)
    return rows,x,y

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'frozen.json').exists(),'Preserve freeze'
    grid=[];best=-1
    for layer in [24,31]:
        _,x,y=load('train',layer);_,cx,cy=load('calibration',layer)
        for c in [.0001,.001,.01,.1,1.]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='liblinear',dual=True,max_iter=2000,random_state=20261026))
            model.fit(x,y);auc=float(roc_auc_score(cy,model.decision_function(cx)))
            entry={'layer':layer,'C':c,'calibration_auroc':auc};grid.append(entry);print(entry,flush=True)
            if auc>best:best=auc;winner=entry;pickle.dump(model,open(OUT/'model.pkl','wb'))
    frozen={'winner':winner,'grid':grid,'selection':'official train/calibration only, not validation or confirmation'}
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2))
    rows,x,y=load('validation',winner['layer']);model=pickle.load(open(OUT/'model.pkl','rb'));s=model.decision_function(x)
    sys.path.insert(0,str(TASK.parent/'14_multilayer_semantic_evidence/code'))
    from evaluate import metrics
    report=metrics(rows,s,bootstrap=True);(OUT/'validation_metrics.json').write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/'validation_scores.npz',row_ids=[r['row_id'] for r in rows],score=s)
    print('validation',report,flush=True)
if __name__=='__main__':main()
