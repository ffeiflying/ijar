"""No-fit entity-bound attribute evidence and random-set spatial predicates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
import spacy
TASK=Path(__file__).resolve().parent.parent;SOURCE=TASK.parent/'18_training_free_research'
sys.path.insert(0,str(SOURCE/'code'))
from score_patch_frames import frame,assertion
from semantic_sets import nested
sys.path.insert(0,str(TASK.parent/'17_semantic_evidence_v2/code'))
from score_atoms import atoms_mask
# Avoid collision with this module's filename.
import importlib.util
spec=importlib.util.spec_from_file_location('semantic_eval14',TASK.parent/'14_multilayer_semantic_evidence/code/evaluate.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
SPLITS=TASK.parents[1]/'shared/data/splits';LAYERS=[16,24,31]


def push_binary(e,target):
    order,mass,_=e;support=0.;opposition=0.
    for i,m in enumerate(mass):
        A=set(order[:i+1])
        if A<=target:support+=m
        elif not A&target:opposition+=m
    return support,opposition,max(0.,1-support-opposition)


def location(q):
    pi=q/max(float(q.max()),1e-30);order=np.argsort(-pi);s=pi[order];mass=s-np.r_[s[1:],0.]
    side=round(len(q)**.5);assert side*side==len(q)
    xy=np.stack([order%side,order//side],-1)/(side-1)
    lower=np.minimum.accumulate(xy,axis=0);upper=np.maximum.accumulate(xy,axis=0)
    return mass,lower,upper


def spatial(a,b,axis):
    ma,la,ua=location(a);mb,lb,ub=location(b);joint=ma[:,None]*mb[None,:]
    forward=float((joint*(ua[:,axis,None]<lb[None,:,axis])).sum())
    reverse=float((joint*(la[:,axis,None]>ub[None,:,axis])).sum())
    return forward,reverse,max(0.,1-forward-reverse)


def compute(split):
    rows=json.loads((SPLITS/f'{split}.json').read_text());nlp=spacy.load('en_core_web_sm',disable=['ner'])
    cfg=json.loads((SOURCE/'results/patch_v4_config.json').read_text());tok=AutoTokenizer.from_pretrained(cfg['model'],trust_remote_code=True,use_fast=False,local_files_only=True)
    results=[];coverage=[]
    for n,r in enumerate(rows):
        path=SOURCE/f'results/patch_v4/{split}/{r["row_id"]}.npz';meta=json.loads(path.with_suffix('.atoms.json').read_text());atoms=meta['atoms'];active=atoms_mask(meta,nlp);values={}
        with np.load(path) as z:
            assert z['language_forward_count']==1
            for layer in LAYERS:
                pf=f'L{layer}_';lp=z[pf+'semantic_patch_logp'].astype(float);att=z[pf+'atom_attention'].astype(float);na=len(lp)
                prepared=[frame(meta,i,z[pf+'semantic_frame_token_ids'][i],z[pf+'semantic_frame_mask'][i],tok) for i in range(na)]
                for power in [1.,4.]:
                    for pool in ['semantic','semantic_attention']:
                        for temp in [1.,4.]:
                            nodes=[];objects={};maps={};counts={'object':0,'attribute_bound':0,'spatial_bound':0,'unknown':0}
                            for i in active:
                                if atoms[i]['kind']!='object':continue
                                ix,target=prepared[i];labels=lp[i,ix]
                                global_lp=logsumexp(labels,axis=-1)-np.log(labels.shape[-1])
                                objects[i]=push_binary(nested(softmax(global_lp/temp)),target)
                                nodes.append(objects[i]);counts['object']+=1
                                probability=softmax(labels,axis=0)[list(target)].sum(0)
                                w=np.maximum(probability,1e-30)**power
                                if pool=='semantic_attention':w*=att[i]
                                maps[i]=w/np.maximum(w.sum(),1e-30)
                            for i in active:
                                atom=atoms[i]
                                if atom['kind']=='object':continue
                                node=(0.,0.,1.)
                                ix,target=prepared[i]
                                if atom['kind']=='attribute' and len(ix)>1 and atom['anchor'] in maps:
                                    j=atom['anchor'];labels=lp[i,ix]
                                    # Semantic logits at noun-localized patches; no label from dataset.
                                    logits=logsumexp(labels+np.log(np.maximum(maps[j],1e-300))[None,:],axis=-1)
                                    p,o,u=push_binary(nested(softmax(logits/temp)),target)
                                    rho=objects[j][0]
                                    node=(rho*p,rho*o,1-rho*(p+o));counts['attribute_bound']+=1
                                elif atom['kind']=='relation' and atom['subject'] in maps and atom['object'] in maps:
                                    word=atom['lemma'];axis=0 if word in ['left','right'] else (1 if word in ['above','below','under','over','on'] else None)
                                    if axis is not None:
                                        left,right,u=spatial(maps[atom['subject']],maps[atom['object']],axis)
                                        if word in ['right','below','under']:left,right=right,left
                                        rho=min(objects[atom['subject']][0],objects[atom['object']][0])
                                        node=(rho*left,rho*right,1-rho*(left+right));counts['spatial_bound']+=1
                                if node==(0.,0.,1.):counts['unknown']+=1
                                nodes.append(node)
                            if not nodes:nodes=[(0.,0.,1.)]
                            node=np.array(nodes);assert np.allclose(node.sum(-1),1)
                            opposition=node[:,1]
                            # Frechet lower belief of any false premise: max marginal belief.
                            cf=float(opposition.max());normal=max(0.,float(node[:,0].sum()-(len(node)-1)));u=max(0.,1-cf-normal)
                            prefix=f'L{layer}_P{power:g}_{pool}_T{temp:g}_'
                            values[prefix+'global_lower_cf']=cf
                            values[prefix+'acf_mean']=float(opposition.mean())
                            values[prefix+'global_betp']=float(cf+.5*u)
                            values[prefix+'independent_union_cf']=float(1-np.prod(1-opposition))
                            values[prefix+'objects_acf']=float(np.mean([x[1] for x in objects.values()])) if objects else 0.
                            if layer==31 and power==1 and pool=='semantic' and temp==1:coverage.append(counts)
        results.append(values)
        if (n+1)%50==0:print('scored',split,n+1,flush=True)
    keys=sorted(results[0]);assert all(set(v)==set(keys) for v in results)
    return rows,{k:np.array([v[k] for v in results]) for k in keys},{k:sum(c[k] for c in coverage) for k in coverage[0]}


def verify():
    a=np.zeros(16);a[4]=1;b=np.zeros(16);b[7]=1
    left,right,u=spatial(a,b,0);assert abs(left-1)<1e-12 and right==0 and u==0
    left,right,u=spatial(np.ones(16),np.ones(16),0);assert left==right==0 and abs(u-1)<1e-12
    return {'localized_left_truth_verified':True,'uninformative_location_is_unknown':True,'assumptions':'independent random location sets for spatial pushforward, entity-belief discount is an uncalibrated conservative proxy; global union lower bound needs no premise independence'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['freeze','validation'],required=True);a=p.parse_args();out=TASK/'results';out.mkdir(exist_ok=True)
    if a.mode=='freeze':
        assert not (out/'frozen.json').exists();rows,s,coverage=compute('dev');auc={k:float(roc_auc_score([r['label'] for r in rows],v)) for k,v in s.items()}
        eligible=[k for k in auc if k.endswith('global_lower_cf') or k.endswith('acf_mean')]
        f={'best_cf':max(eligible,key=lambda k:auc[k]),'best_global_cf':max((k for k in eligible if k.endswith('global_lower_cf')),key=lambda k:auc[k]),'best_objects':max((k for k in auc if k.endswith('objects_acf')),key=lambda k:auc[k]),'training':False,'selection':'dev fixed formulas only, no fitted quality coefficients','math':verify(),'coverage':coverage,'configurations':len(auc)}
        (out/'frozen.json').write_text(json.dumps(f,indent=2));(out/'dev_metrics.json').write_text(json.dumps(auc,indent=2));print(json.dumps({**f,'dev_auc':{k:auc[f[k]] for k in ['best_cf','best_global_cf','best_objects']}},indent=2))
    else:
        f=json.loads((out/'frozen.json').read_text());rows,s,coverage=compute('validation');keys=set([f['best_cf'],f['best_global_cf'],f['best_objects']]);report={k:legacy.metrics(rows,s[k],bootstrap=k==f['best_cf']) for k in keys}
        (out/'validation_metrics.json').write_text(json.dumps({'methods':report,'coverage':coverage},indent=2));print(json.dumps({'methods':report,'coverage':coverage},indent=2))
if __name__=='__main__':main()
