"""No training: local prequential visual counterevidence and same-feature controls."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import spacy
TASK=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(TASK.parent/'17_semantic_evidence_v2/code'))
from score_atoms import atoms_mask
from evidence import head_evidence
sys.path.insert(0,str(TASK.parent/'14_multilayer_semantic_evidence/code'))
from evaluate import metrics
SPLITS=TASK.parents[1]/'shared/data/splits';LAYERS=[0,8,16,20,24,27,31]

def compute(split):
    rows=json.loads((SPLITS/f'{split}.json').read_text());nlp=spacy.load('en_core_web_sm',disable=['ner']);out=[];errors=[];latency=[]
    for r in rows:
        p=TASK/f'results/prequential/{split}/{r["row_id"]}.npz';assert p.exists(),p
        meta=json.loads((TASK.parent/f'17_semantic_evidence_v2/results/features/{split}/{r["row_id"]}.atoms.json').read_text())
        roles={'objects':atoms_mask(meta,nlp,True),'premises':atoms_mask(meta,nlp)};values={}
        with np.load(p) as z:
            assert z['vision_forward_count']==z['language_forward_count']==1
            drop=int(z['qindex_drop']);count=len(z['qids'])
            positions=[[j-drop for j in a['qindices'] if 0<=j-drop<count] for a in meta['atoms']]
            positions=[ix if ix else list(range(count)) for ix in positions]
            def atoms(x):return np.array([x[ix].mean() for ix in positions])
            dv=[];dt=[];mass=[]
            for l in LAYERS:
                pf=f'L{l}_';joint=z[pf+'joint_logp'].astype(float);visual=z[pf+'visual_logp'].astype(float);text=z[pf+'text_logp'].astype(float);base=z[pf+'base_logp'].astype(float)
                dv.append(atoms(joint-text));dt.append(atoms(joint-visual));mass.append(atoms(z[pf+'visual_mass'].astype(float)))
                controls={'joint_nll':atoms(-joint),'text_nll':atoms(-text),'visual_nll':atoms(-visual),'visual_countercontrast':-dv[-1],'text_dominance':dt[-1]-dv[-1],'joint_prior_contrast':atoms(base-joint)}
                for role,ix in roles.items():
                    for name,score in controls.items():
                        for reduce in ['mean','max']:values[f'L{l}_{role}_{name}_{reduce}']=float(getattr(np,reduce)(score[ix]))
                errors.append(float(z[pf+'reconstruction_error']))
            dv=np.stack(dv);dt=np.stack(dt);mass=np.stack(mass)
            selections={f'L{l}':[i] for i,l in enumerate(LAYERS)}
            selections.update(all=list(range(7)),mid=list(range(2,6)),late=list(range(3,7)))
            for name,indices in selections.items():
                for discount in ['none','visual_mass']:
                    vr=1. if discount=='none' else mass[indices]
                    # Correlated layer views tempered by mean effective simple-support weights.
                    v=head_evidence(dv[indices],None if discount=='none' else vr)
                    t=head_evidence(dt[indices])
                    vp=v['support'];vn=v['opposition'];tp=t['support'];tn=t['opposition']
                    cf=vp*tn+vn*tp
                    measures={'assertion_cf':vn,'source_cross_cf':cf,'visual_internal_cf':v['internal_cf'],'visual_ignorance':v['ignorance'],'negative_visual_betp':v['opposite_betp']}
                    for role,ix in roles.items():
                        for m,score in measures.items():
                            for reduce in ['mean','max']:values[f'{name}_{discount}_{role}_{m}_{reduce}']=float(getattr(np,reduce)(score[ix]))
            latency.append(float(z['seconds']))
        out.append(values)
    keys=sorted(out[0]);assert all(set(v)==set(keys) for v in out)
    return rows,{k:np.array([v[k] for v in out]) for k in keys},{'rows':len(rows),'vision_calls_per_row':1,'language_calls_per_row':1,'generated_tokens':0,'max_attention_decomposition_error':max(errors),'median_extraction_seconds':float(np.median(latency)),'latency_scope':'all7layers/4localreadouts, excluding model loading; not minimal detector latency'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['freeze','validation'],required=True);a=p.parse_args();dest=TASK/'results/prequential_scores';dest.mkdir(parents=True,exist_ok=True)
    if a.mode=='freeze':
        assert not (dest/'frozen.json').exists();rows,s,audit=compute('dev');auc={k:float(roc_auc_score([r['label'] for r in rows],v)) for k,v in s.items()}
        cf=[k for k in auc if '_cf_' in k];raw=[k for k in auc if '_nll_' in k or 'contrast' in k or 'text_dominance' in k]
        f={'best_cf':max(cf,key=lambda k:auc[k]),'best_raw':max(raw,key=lambda k:auc[k]),'block0_cf':max((k for k in cf if k.startswith('L0_')),key=lambda k:auc[k]),'best_betp':max((k for k in auc if 'negative_visual_betp' in k),key=lambda k:auc[k]),'training':False,'selection':'dev fixed-formula selection only, no fitted mass allocator/calibration','configurations':len(auc),'audit':audit,'limitations':'local causal-path deletion readout, not full no-image model evaluation; signed likelihood change is operational counterevidence, not calibrated truth likelihood'}
        (dest/'frozen.json').write_text(json.dumps(f,indent=2));(dest/'dev_metrics.json').write_text(json.dumps(auc,indent=2));print(json.dumps({**f,'dev_auc':{k:auc[f[k]] for k in ['best_cf','best_raw','block0_cf','best_betp']}},indent=2))
    else:
        f=json.loads((dest/'frozen.json').read_text());rows,s,audit=compute('validation');report={f[k]:metrics(rows,s[f[k]],bootstrap=k=='best_cf') for k in ['best_cf','best_raw','block0_cf','best_betp']}
        (dest/'validation_metrics.json').write_text(json.dumps({'methods':report,'audit':audit},indent=2));print(json.dumps({'methods':report,'audit':audit},indent=2))
if __name__=='__main__':main()
