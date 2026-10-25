"""Training-free typed visual-semantic random sets; no learned mass allocator."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp
from sklearn.metrics import roc_auc_score
import spacy
from transformers import AutoTokenizer
from semantic_sets import nested,conflict,verify
TASK=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(TASK.parent/'17_semantic_evidence_v2/code'))
from score_atoms import atoms_mask
sys.path.insert(0,str(TASK.parent/'14_multilayer_semantic_evidence/code'))
from evaluate import metrics
SPLITS=TASK.parents[1]/'shared/data/splits'
LAYERS=[0,16,24,31]
FAMILIES=[set(s.split()) for s in ['red blue green yellow orange purple pink brown black white gray grey silver golden','wooden metal plastic','large small','tall short','long short','round square','old young','empty full','open closed','wet dry','left right','above below under over','front behind','inside outside']]
HIERARCHY={'person':set('person man woman boy girl'.split()),'animal':set('animal dog cat horse sheep cow bird elephant bear zebra giraffe fish chicken duck'.split()),'vehicle':set('vehicle bicycle car motorcycle airplane bus train truck boat'.split()),'furniture':set('furniture chair couch bed bench table'.split()),'fruit':set('fruit apple orange banana'.split()),'food':set('food fruit apple orange banana bread fish chicken sandwich broccoli carrot pizza donut cake'.split())}
SYN={'bike':'bicycle','sofa':'couch','tv':'television','automobile':'car','motorbike':'motorcycle','grey':'gray','plane':'airplane'}

def frame(meta,i,token_ids,mask,tokenizer):
    atom=meta['atoms'][i];lemma=SYN.get(atom['lemma'],atom['lemma'])
    original=np.where(mask)[0]
    labels=[lemma if j==0 else tokenizer.decode([int(next(v for v in token_ids[j] if v))]).strip().lower() for j in original]
    labels=[SYN.get(v,v) for v in labels]
    allowed=list(range(len(original)))
    if atom['kind'] in ['attribute','relation']:
        family=next((f for f in FAMILIES if lemma in f),None)
        allowed=[j for j,name in enumerate(labels) if name in family or j==0] if family else [0]
    accepted=set(HIERARCHY.get(lemma,{lemma}))
    # A generic superclass cannot refute a more specific premise.
    accepted|={name for name,desc in HIERARCHY.items() if lemma in desc}
    target={k for k,j in enumerate(allowed) if j==0 or labels[j] in accepted}
    return original[allowed],target

def assertion(a,target):
    order,mass,pi=a;seen=False;cf=0.;betp=0.
    for i,j in enumerate(order):
        if j in target:seen=True
        if not seen:cf+=mass[i]
        betp+=mass[i]*len(set(order[:i+1])&target)/(i+1)
    return cf,float(mass[-1]),1-betp

def compute(split):
    rows=json.loads((SPLITS/f'{split}.json').read_text());nlp=spacy.load('en_core_web_sm',disable=['ner'])
    config=json.loads((TASK/'results/patch_v4_config.json').read_text())
    tok=AutoTokenizer.from_pretrained(config['model'],trust_remote_code=True,use_fast=False,local_files_only=True)
    results=[]
    for n,row in enumerate(rows):
        path=TASK/f'results/patch_v4/{split}/{row["row_id"]}.npz';assert path.exists(),path
        meta=json.loads(path.with_suffix('.atoms.json').read_text());masks={'objects':atoms_mask(meta,nlp,True),'premises':atoms_mask(meta,nlp)};values={}
        with np.load(path) as z:
            assert z['language_forward_count']==1
            for layer in LAYERS:
                pf=f'L{layer}_';lp=z[pf+'semantic_patch_logp'].astype(float);heads=z[pf+'semantic_head_logp'].astype(float)
                attention=z[pf+'atom_attention'].astype(float);na=len(lp)
                for mode in ['attention','global','target_max','attention_top8']:
                    if mode=='attention':pooled=logsumexp(heads,axis=0)-np.log(heads.shape[0])
                    elif mode=='global':pooled=logsumexp(lp,axis=-1)-np.log(lp.shape[-1])
                    elif mode=='target_max':
                        ix=lp[:,0,:].argmax(-1);pooled=np.stack([lp[i,:,j] for i,j in enumerate(ix)])
                    else:
                        ix=np.argsort(attention,axis=-1)[:,-8:];pooled=np.stack([logsumexp(lp[i][:,ix[i]],axis=-1)-np.log(8) for i in range(na)])
                    prepared=[frame(meta,i,z[pf+'semantic_frame_token_ids'][i],z[pf+'semantic_frame_mask'][i],tok) for i in range(na)]
                    for temperature in [1.,4.]:
                        for discount in ['none','coverage']:
                            stats=[]
                            for i,(ix,target) in enumerate(prepared):
                                logits=pooled[i,ix];rho=1. if discount=='none' else min(1.,float(np.exp(logsumexp(logits))))
                                if len(ix)==1:
                                    stats.append([0.,1.,.5,.5,-float(logits[0])]);continue
                                a=nested(softmax(logits/temperature),rho)
                                cf,u,nb=assertion(a,target)
                                probs=softmax(logits/temperature)
                                matching=-float(logsumexp(logits[list(target)]))
                                stats.append([cf,u,nb,1-float(probs[list(target)].sum()),matching])
                            stats=np.array(stats)
                            for role,ix in masks.items():
                                for j,name in enumerate(['assertion_cf','ignorance','negative_betp','negative_conditional_probability','negative_matching']):
                                    for reduce in ['mean','max']:
                                        values[f'L{layer}_{mode}_T{temperature:g}_{discount}_{role}_{name}_{reduce}']=float(getattr(np,reduce)(stats[ix,j]))
                # Head disagreement on the same semantic frame, without learned weights.
                for role,ix_atoms in masks.items():
                    stats=[]
                    for i in ix_atoms:
                        ix,target=prepared[i]
                        if len(ix)==1:
                            stats.append([0.,0.,0.]);continue
                        hs=[nested(softmax(heads[h,i,ix])) for h in range(0,heads.shape[0],4)]
                        pair=np.array([conflict(hs[a],hs[b]) for a in range(len(hs)) for b in range(a)])
                        local=np.array([assertion(h,target) for h in hs])
                        stats.append([pair[:,0].mean(),pair[:,1].mean(),local[:,0].mean()])
                    for j,name in enumerate(['head_minimum_cf','head_independent_cf','head_assertion_cf']):values[f'L{layer}_{role}_{name}_mean']=float(np.mean(stats,axis=0)[j])
        results.append(values)
        if (n+1)%50==0:print('scored',split,n+1,flush=True)
    keys=sorted(results[0]);assert all(set(v)==set(keys) for v in results)
    return rows,{k:np.array([v[k] for v in results]) for k in keys}

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['freeze','validation'],required=True);a=p.parse_args()
    out=TASK/'results/patch_semantic_sets_v2';out.mkdir(parents=True,exist_ok=True)
    if a.mode=='freeze':
        assert not (out/'frozen.json').exists();rows,s=compute('dev');auc={k:float(roc_auc_score([r['label'] for r in rows],v)) for k,v in s.items()}
        cf=[k for k in auc if '_cf_' in k];baselines=[k for k in auc if 'negative_conditional_probability' in k or 'negative_matching' in k]
        f={'best_cf':max(cf,key=lambda k:auc[k]),'best_baseline':max(baselines,key=lambda k:auc[k]),'by_layer_cf':{str(l):max((k for k in cf if k.startswith(f'L{l}_')),key=lambda k:auc[k]) for l in LAYERS},'training':False,'selection':'dev fixed formulas only, no weights fitted; semantic compatibility rules are label-free','configurations':len(auc),'math':verify(),'limits':'frozen unembedding affinities not calibrated semantic probabilities; fixed ontology incomplete; local description alternatives not global object-existence classes'}
        (out/'frozen.json').write_text(json.dumps(f,indent=2));(out/'dev_metrics.json').write_text(json.dumps(auc,indent=2));print(json.dumps({**f,'dev_auc':{k:auc[f[k]] for k in ['best_cf','best_baseline']}},indent=2))
    else:
        f=json.loads((out/'frozen.json').read_text());rows,s=compute('validation');keys=set([f['best_cf'],f['best_baseline']]+list(f['by_layer_cf'].values()));report={k:metrics(rows,s[k],bootstrap=k==f['best_cf']) for k in sorted(keys)}
        (out/'validation_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
        np.savez_compressed(out/'validation_predictions.npz',row_ids=[r['row_id'] for r in rows],**{k:s[k] for k in keys})
        bytype={}
        for typ in ['object','attribute','relation']:
            ids={r['image_id'] for r in rows if r.get('conflict_type')==typ}
            ix=[i for i,r in enumerate(rows) if r['image_id'] in ids]
            if ix:bytype[typ]={k:metrics([rows[i] for i in ix],s[k][ix]) for k in [f['best_cf'],f['best_baseline']]}
        signatures=[]
        for r in rows:
            meta=json.loads((TASK/f'results/patch_v4/validation/{r["row_id"]}.atoms.json').read_text())
            signatures.append(sorted((a['lemma'],tuple(a['aliases'])) for a in meta['atoms'] if a['kind']=='object'))
        invariance={}
        for typ in ['object','attribute','relation']:
            pairs=[]
            for imageid in {r['image_id'] for r in rows if r.get('conflict_type')==typ}:
                ix=[i for i,r in enumerate(rows) if r['image_id']==imageid]
                if len(ix)==2 and signatures[ix[0]]==signatures[ix[1]]:pairs.append(ix)
            invariance[typ]={'same_object_signature_pairs':len(pairs),'tied_cf_within_1e-6':sum(abs(float(s[f['best_cf']][a]-s[f['best_cf']][b]))<1e-6 for a,b in pairs)}
        (out/'validation_by_type.json').write_text(json.dumps({'paired_type_metrics':bytype,'same_object_invariance':invariance},indent=2))
if __name__=='__main__':main()
