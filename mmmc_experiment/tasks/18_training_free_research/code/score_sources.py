"""No fitting: fixed semantic readouts, set masses, dev-only formula selection."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
import spacy
from semantic_sets import nested,conflict,against_assertion,verify
TASK=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(TASK.parent/'17_semantic_evidence_v2/code'))
from score_atoms import atoms_mask
sys.path.insert(0,str(TASK.parent/'14_multilayer_semantic_evidence/code'))
from evaluate import metrics
SPLITS=TASK.parents[1]/'shared/data/splits'
LAYERS=[0,16,24,31]
SOURCES=['visual','text','visual_residual','text_residual','base','joint']
PAIRS=[('visual','text'),('visual_residual','text_residual'),('visual_residual','base'),('joint','base')]

def compute(split):
    rows=json.loads((SPLITS/f'{split}.json').read_text());nlp=spacy.load('en_core_web_sm',disable=['ner']);scores=[];audit=[]
    for row in rows:
        path=TASK/f'results/source_v3/{split}/{row["row_id"]}.npz';assert path.exists(),path
        meta=json.loads(path.with_suffix('.atoms.json').read_text());masks={'objects':atoms_mask(meta,nlp,True),'premises':atoms_mask(meta,nlp)};values={}
        with np.load(path) as z:
            assert z['language_forward_count']==1
            audit.append(max(float(z[f'L{l}_source_reconstruction_relative_error']) for l in LAYERS))
            for layer in LAYERS:
                pf=f'L{layer}_source_';frame_mask=z[pf+'frame_mask'];na=len(frame_mask)
                arrays={s:z[pf+s+'_frame_logp'].astype(float) for s in SOURCES}
                coverage={s:np.exp(z[pf+s+'_coverage_logp'].astype(float)).clip(0,1) for s in SOURCES}
                for temp in [1.,4.]:
                    for discount in ['none','coverage']:
                        evidence={s:[nested(softmax(arrays[s][i,frame_mask[i]]/temp),1. if discount=='none' else coverage[s][i]) for i in range(na)] for s in SOURCES}
                        local={}
                        for s in SOURCES:
                            m=np.array([against_assertion(e) for e in evidence[s]])
                            local[s+'_assertion_cf']=m[:,0]
                            local[s+'_ignorance']=m[:,1]
                            local[s+'_negative_betp']=1-m[:,2]
                        for a,b in PAIRS:
                            c=np.array([conflict(x,y) for x,y in zip(evidence[a],evidence[b])])
                            local[a+'_vs_'+b+'_minimum_cf']=c[:,0]
                            local[a+'_vs_'+b+'_independent_cf']=c[:,1]
                            # Directional loss of target support, not symmetric divergence.
                            pa=np.array([against_assertion(e)[2] for e in evidence[a]])
                            pb=np.array([against_assertion(e)[2] for e in evidence[b]])
                            local[a+'_vs_'+b+'_target_suppression']=np.maximum(pb-pa,0)
                        for role,ix in masks.items():
                            for name,val in local.items():
                                for reduce in ['mean','max']:
                                    key=f'L{layer}_T{temp:g}_{discount}_{role}_{name}_{reduce}'
                                    values[key]=float(getattr(np,reduce)(val[ix]))
                for a,b in PAIRS:
                    contrast=z[pf+b+'_target_logp'].astype(float)-z[pf+a+'_target_logp'].astype(float)
                    for role,ix in masks.items():values[f'L{layer}_{role}_{a}_vs_{b}_raw_contrast_mean']=float(contrast[ix].mean())
                for s in SOURCES:
                    for role,ix in masks.items():values[f'L{layer}_{role}_{s}_raw_negative_matching_mean']=float(-z[pf+s+'_target_logp'][ix].mean())
        scores.append(values)
    keys=sorted(scores[0]);assert all(set(s)==set(keys) for s in scores)
    return rows,{k:np.array([s[k] for s in scores]) for k in keys},{'rows':len(rows),'language_forward_count_per_row':1,'max_attention_reconstruction_relative_error':max(audit),'no_generated_answer':True,'training':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['freeze','validation'],required=True);a=p.parse_args()
    out=TASK/'results/semantic_sources';out.mkdir(parents=True,exist_ok=True)
    if a.mode=='freeze':
        assert not (out/'frozen.json').exists();rows,s,audit=compute('dev');auc={k:float(roc_auc_score([r['label'] for r in rows],v)) for k,v in s.items()}
        cf=[k for k in auc if '_cf_' in k];robust=[k for k in cf if '_minimum_cf_' in k];raw=[k for k in auc if '_raw_' in k]
        f={'best_cf':max(cf,key=lambda k:auc[k]),'best_dependence_robust_cf':max(robust,key=lambda k:auc[k]),'best_raw':max(raw,key=lambda k:auc[k]),'by_layer_cf':{str(l):max((k for k in cf if k.startswith(f'L{l}_')),key=lambda k:auc[k]) for l in LAYERS},'configurations':len(auc),'selection':'dev fixed-formula selection only; no fitted coefficients or mass assignment','training':False,'math':verify(),'audit':audit,'limitations':'local token-source views, not complete modality-isolated reruns; dictionary affinities not calibrated truth; multiple objects in an image not mutually exclusive, frame describes queried local entity'}
        (out/'frozen.json').write_text(json.dumps(f,indent=2));(out/'dev_metrics.json').write_text(json.dumps(auc,indent=2));print(json.dumps({**f,'dev_auc':{k:auc[f[k]] for k in ['best_cf','best_dependence_robust_cf','best_raw']}},indent=2))
    else:
        f=json.loads((out/'frozen.json').read_text());rows,s,audit=compute('validation');keys=set([f[k] for k in ['best_cf','best_dependence_robust_cf','best_raw']]+list(f['by_layer_cf'].values()));r={k:metrics(rows,s[k],bootstrap=k==f['best_cf']) for k in sorted(keys)}
        (out/'validation_metrics.json').write_text(json.dumps({'methods':r,'audit':audit},indent=2));print(json.dumps({'methods':r,'audit':audit},indent=2))
if __name__=='__main__':main()
