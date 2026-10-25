"""Training-free spatial random sets and semantic-affinity controls from one prefill.
No fitting, weights, supervised calibration or new VLM calls. Dev selects formulas only.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
import spacy
TASK=Path(__file__).resolve().parent.parent
PREV=TASK.parent/'17_semantic_evidence_v2'
sys.path.insert(0,str(PREV/'code'))
from score_atoms import atoms_mask
sys.path.insert(0,str(TASK.parent/'14_multilayer_semantic_evidence/code'))
from evaluate import metrics
SPLITS=TASK.parents[1]/'shared/data/splits'
LAYERS=[0,16,20,24,27,31]

def spatial_cf(a,b):
    """Nested random sets on patch indices. Independent vs shared uniform cuts."""
    pa=a/np.maximum(a.max(-1,keepdims=True),1e-300)
    pb=b/np.maximum(b.max(-1,keepdims=True),1e-300)
    order=np.argsort(-pa,axis=-1)
    sa=np.take_along_axis(pa,order,-1)
    masses=sa-np.concatenate([sa[:,1:],np.zeros((len(sa),1))],-1)
    overlap=np.maximum.accumulate(np.take_along_axis(pb,order,-1),axis=-1)
    independent=np.sum(masses*(1-overlap),-1)
    shared=1-np.minimum(pa,pb).max(-1)
    return independent,shared,pa.min(-1),pb.min(-1)

def compute(split):
    rows=json.loads((SPLITS/f'{split}.json').read_text())
    nlp=spacy.load('en_core_web_sm',disable=['ner']);all_scores=[]
    for row in rows:
        p=PREV/f'results/features/{split}/{row["row_id"]}.npz'
        meta=json.loads(p.with_suffix('.atoms.json').read_text());score={}
        masks={'objects':atoms_mask(meta,nlp,True),'premises':atoms_mask(meta,nlp)}
        with np.load(p) as z:
            for layer in LAYERS:
                prefix=f'L{layer}_';lp=z[prefix+'atom_patch_logp'].astype(float)
                att=z[prefix+'atom_attention'].astype(float)
                for correction in ['original','relative','no_bottom']:
                    a=att.copy()
                    if correction=='relative':a/=np.maximum(a.mean(0,keepdims=True),1e-12)
                    if correction=='no_bottom':a[:,-16:]=0
                    a/=np.maximum(a.sum(-1,keepdims=True),1e-12)
                    for temperature in [1.,4.]:
                        b=np.exp((lp-lp.max(-1,keepdims=True))/temperature)
                        independent,shared,ua,ub=spatial_cf(a,b)
                        # Semantic lift of attention over within-image average; no dataset reference.
                        log_ground=logsumexp(lp+np.log(np.maximum(a,1e-300)),axis=-1)
                        log_global=logsumexp(lp,axis=-1)-np.log(lp.shape[-1])
                        local={'spatial_independent_cf':independent,'spatial_shared_cf':shared,'spatial_bayesian_disagreement':1-(a*b/b.sum(-1,keepdims=True)).sum(-1),'attention_ignorance':ua,'semantic_ignorance':ub,'negative_lift':log_global-log_ground,'negative_matching':-log_ground,'negative_max_patch':-lp.max(-1)}
                        for role,ix in masks.items():
                            for name,values in local.items():
                                for reduce in ['mean','max']:
                                    score[f'L{layer}_{correction}_T{temperature:g}_{role}_{name}_{reduce}']=float(getattr(np,reduce)(values[ix]))
                # Geometry controls avoid any classifier or vocabulary output readout.
                v=z[prefix+'atom_vcontext'].astype(float);q=z[prefix+'atom_qcontext'].astype(float)
                cos=(v*q).sum(-1)/np.maximum(np.linalg.norm(v,axis=-1)*np.linalg.norm(q,axis=-1),1e-12)
                for role,ix in masks.items():
                    score[f'L{layer}_{role}_cosine_mismatch_mean']=float((1-cos[ix]).mean())
                pre=z[prefix+'qpost_pred'].astype(float);post=z[prefix+'qblock_pred'].astype(float)
                score[f'L{layer}_ffn_direction_change']=float(1-np.dot(pre,post)/(np.linalg.norm(pre)*np.linalg.norm(post)))
        all_scores.append(score)
    keys=sorted(all_scores[0]);assert all(set(s)==set(keys) for s in all_scores)
    return rows,{k:np.array([s[k] for s in all_scores]) for k in keys}

def verify():
    rng=np.random.default_rng(9);error=0
    for _ in range(20):
        a=rng.random(5);a/=a.max();b=rng.random(5);b/=b.max()
        def focal(pi):
            ix=np.argsort(-pi);s=pi[ix];return [(set(ix[:j+1]),s[j]-(s[j+1] if j+1<len(s) else 0)) for j in range(len(s))]
        exact=sum(ma*mb for A,ma in focal(a) for B,mb in focal(b) if not A&B)
        fast=spatial_cf(a[None],b[None])[0][0];error=max(error,abs(exact-fast))
        assert abs(spatial_cf(a[None],a[None])[1][0])<1e-12
    return {'independent_random_set_max_error':error,'shared_cut_identical_source_cf':0.,'scope':'spatial-view disagreement, not proof of semantic negation; shared cut is a dependence assumption'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['freeze','validation'],required=True);a=p.parse_args()
    out=TASK/'results/cached_candidates';out.mkdir(parents=True,exist_ok=True)
    if a.mode=='freeze':
        assert not (out/'frozen.json').exists()
        rows,scores=compute('dev');y=[r['label'] for r in rows];auc={k:float(roc_auc_score(y,s)) for k,s in scores.items()}
        cf=[k for k in auc if '_cf_' in k];controls=[k for k in auc if 'negative_matching' in k or 'negative_max_patch' in k or 'negative_lift' in k]
        frozen={'best_cf':max(cf,key=lambda k:auc[k]),'best_matching':max(controls,key=lambda k:auc[k]),'best_geometry':max((k for k in auc if 'cosine_mismatch' in k or 'ffn_direction' in k),key=lambda k:auc[k]),'by_layer_cf':{str(l):max((k for k in cf if k.startswith(f'L{l}_')),key=lambda k:auc[k]) for l in LAYERS},'configurations':len(auc),'training':False,'selection':'dev formula selection only, no fitted coefficients','math':verify()}
        (out/'frozen.json').write_text(json.dumps(frozen,indent=2));(out/'dev_metrics.json').write_text(json.dumps(auc,indent=2));print(json.dumps({**frozen,'selected_dev_auc':{k:auc[frozen[k]] for k in ['best_cf','best_matching','best_geometry']}},indent=2))
    else:
        f=json.loads((out/'frozen.json').read_text());rows,s=compute('validation');keys=set([f[k] for k in ['best_cf','best_matching','best_geometry']]+list(f['by_layer_cf'].values()));report={k:metrics(rows,s[k],bootstrap=k==f['best_cf']) for k in sorted(keys)}
        (out/'validation_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
