"""Question-only shortcut control, never consumes image or answers."""
import json,pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from train_mil import TASK,records
import sys
sys.path.insert(0,str(TASK.parent/'14_multilayer_semantic_evidence/code'))
from evaluate import metrics
out=TASK/'results/text_shortcut';out.mkdir(parents=True,exist_ok=True)
assert not (out/'frozen.json').exists()
train=records('train');cal=records('calibration');val=records('validation')
x=[r['question'] for r in train];y=[r['label'] for r in train]
grid=[];best=-1
for c in [.1,1.,10.]:
    model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=20000),LogisticRegression(C=c,max_iter=1000))
    model.fit(x,y);auc=float(roc_auc_score([r['label'] for r in cal],model.decision_function([r['question'] for r in cal])))
    grid.append({'C':c,'calibration_auroc':auc})
    if auc>best:best=auc;winner=model;entry=grid[-1]
(out/'frozen.json').write_text(json.dumps({'winner':entry,'grid':grid,'selection':'official train/calibration only'},indent=2))
pickle.dump(winner,open(out/'model.pkl','wb'))
report=metrics(val,winner.decision_function([r['question'] for r in val]),bootstrap=True)
(out/'validation_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
