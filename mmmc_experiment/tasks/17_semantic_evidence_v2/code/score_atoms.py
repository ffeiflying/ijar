"""Score preserved semantic witnesses; tune on dev only, never task16 confirmation."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
import spacy
from evidence import head_evidence,proposition_pushforward,verify
TASK=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(TASK.parents[1]/'shared'))
from layout import SPLITS
sys.path.insert(0,str(TASK.parent/'14_multilayer_semantic_evidence/code'))
from evaluate import metrics
LAYERS=[0,16,20,24,27,31]

def atoms_mask(meta,nlp,objects_only=False):
    doc=nlp(meta['question'])
    question_slots={t.idx for t in doc if t.pos_ in ['NOUN','PROPN'] and any(c.lower_ in ['what','which','whose'] for c in t.children)}
    indices=[i for i,a in enumerate(meta['atoms']) if a['start'] not in question_slots and (a['kind']=='object' if objects_only else True)]
    return indices if indices else list(range(len(meta['atoms'])))

def compute(split):
    rows=json.loads((SPLITS/f'{split}.json').read_text())
    nlp=spacy.load('en_core_web_sm',disable=['ner'])
    with np.load(TASK.parent/'16_reference_semantic_evidence/results/background.npz') as z:
        bg={k:np.log(np.maximum(z[k],1e-300)) for k in z.files}
    result=[]
    for row in rows:
        path=TASK/f'results/features/{split}/{row["row_id"]}.npz'
        assert path.exists(),path
        meta=json.loads(path.with_suffix('.atoms.json').read_text())
        masks={'premises':atoms_mask(meta,nlp),'objects':atoms_mask(meta,nlp,objects_only=True)}
        values={}
        with np.load(path) as z:
            assert z['language_forward_count']==1
            qids=z['qids'].tolist()
            drop=len(meta['qids'])-len(qids)
            for layer in LAYERS:
                pfx=f'L{layer}_'
                refs=[]
                positions=[]
                for atom in meta['atoms']:
                    ix=[j-drop for j in atom['qindices'] if j>=drop]
                    positions.append(ix if ix else list(range(len(qids))))
                    refs.append(logsumexp(bg[f'L{layer}'][atom['aliases']]) if atom['aliases'] else np.mean(bg[f'L{layer}'][np.array(qids)[positions[-1]]]))
                llr=z[pfx+'atom_head_logp'].astype(np.float64)-np.array(refs)[None,:]
                source_mass=np.stack([z[pfx+'head_visual_mass'][:,ix].mean(-1) for ix in positions],-1)
                for heads in [8,32]:
                    chosen=np.argsort(source_mass,axis=0)[-heads:]
                    selected=np.take_along_axis(llr,chosen,axis=0)
                    mass=np.take_along_axis(source_mass,chosen,axis=0)
                    for discount in ['none','vision']:
                        for temperature in [1.,4.]:
                            st=head_evidence(selected,None if discount=='none' else mass,temperature)
                            for role,ix in masks.items():
                                name=f'L{layer}_H{heads}_{discount}_T{temperature:g}_{role}_'
                                opp=st['opposition'][ix]
                                u=st['ignorance'][ix]
                                betp=st['opposite_betp'][ix]
                                logcf=-np.log1p(-np.minimum(opp,1-1e-15))
                                bag=proposition_pushforward(st['support'][ix],opp,u)
                                local={'cf_mean':logcf.mean(),'cf_max':logcf.max(),'cf_sum':logcf.sum(),'betp_mean':betp.mean(),'betp_max':betp.max(),'bag_betp':bag['opposite_betp'],'internal_cf':st['internal_cf'][ix].mean(),'ignorance':u.mean(),'net_contrast':(-selected[:,ix]).mean()}
                                for key,value in local.items():
                                    values[name+key]=float(value)
        result.append(values)
    keys=set(result[0])
    assert all(set(v)==keys for v in result)
    scores={k:np.array([v[k] for v in result]) for k in sorted(keys)}
    assert all(np.isfinite(v).all() for v in scores.values())
    return rows,scores

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['freeze','validation'],required=True)
    a=p.parse_args()
    dest=TASK/'results/atomic_evidence'
    dest.mkdir(parents=True,exist_ok=True)
    fp=dest/'frozen_dev.json'
    if a.mode=='freeze':
        rows,scores=compute('dev')
        y=[r['label'] for r in rows]
        auc={k:float(roc_auc_score(y,v)) for k,v in scores.items()}
        candidates=[k for k in auc if '_cf_' in k and not k.endswith('internal_cf')]
        best=max(candidates,key=lambda k:(auc[k],k))
        overall=max((k for k in auc if not k.endswith('ignorance')),key=lambda k:(auc[k],k))
        bylayer={str(l):max((k for k in candidates if k.startswith(f'L{l}_')),key=lambda k:(auc[k],k)) for l in LAYERS}
        frozen={'best_cf':best,'best_overall':overall,'dev_cf_auroc':auc[best],'dev_overall_auroc':auc[overall],'by_layer':bylayer,'configurations':len(auc),'supervised_training':False,'selection_labels':'dev only','math':verify(),'semantic_limitations':'binary image-vs-background grounding hypotheses, not proven real-world negation; correlated head sources tempered by mean weights'}
        assert not fp.exists(),'Do not overwrite dev freeze'
        fp.write_text(json.dumps(frozen,indent=2))
        (dest/'dev_metrics.json').write_text(json.dumps(auc,indent=2))
        print(json.dumps(frozen,indent=2))
        return
    frozen=json.loads(fp.read_text())
    rows,scores=compute('validation')
    keys=set([frozen['best_cf'],frozen['best_overall']]+list(frozen['by_layer'].values()))
    report={k:metrics(rows,scores[k],bootstrap=k in [frozen['best_cf'],frozen['best_overall']]) for k in sorted(keys)}
    (dest/'validation_metrics.json').write_text(json.dumps({'frozen':frozen,'methods':report},indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
