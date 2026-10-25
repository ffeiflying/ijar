"""Supervised one-prefill proposition evidence; calibration selects, test only reports.

Identical local features and MLP are used for evidential and ordinary MIL controls.
Fixed random projection is label independent. No generated answers or metadata labels
enter the features. Bag supervision never marks every atom in a conflict bag false.
"""
import argparse, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import spacy
from score_atoms import atoms_mask

TASK=Path(__file__).resolve().parent.parent
SPLITS=TASK.parents[1]/'shared/data/splits'
OUT=TASK/'results/supervised_mil'
torch.set_num_threads(4)
threadpool_limits(4)

def records(split):
    if split=='confirmation': return json.loads((TASK/'results/confirmation/confirmation_records.json').read_text())
    p=(TASK/'data' if split in ['train','calibration'] else SPLITS)/f'{split}.json'
    return json.loads(p.read_text())

def unit(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

def load(split,layer,projection,nlp):
    cache=OUT/f'inputs_{split}_L{layer}.npz'
    rows=records(split)
    if cache.exists():
        with np.load(cache) as z:
            assert np.array_equal(z['ids'],[r['row_id'] for r in rows])
            return rows,{k:z[k] for k in z.files if k!='ids'}
    atoms=[]; global_values=[]; masks=[]
    for n,row in enumerate(rows):
        p=TASK/f'results/features/{split}/{row["row_id"]}.npz'
        meta=json.loads(p.with_suffix('.atoms.json').read_text())
        ix=atoms_mask(meta,nlp)
        with np.load(p) as z:
            prefix=f'L{layer}_'
            v=unit(z[prefix+'atom_vcontext'][ix])@projection
            q=unit(z[prefix+'atom_qcontext'][ix])@projection
            h=z[prefix+'atom_head_logp'][:,ix].astype(np.float32)
            kind=np.array([[float(meta['atoms'][i]['kind']==k) for k in ['object','attribute','relation','count','global']] for i in ix])
            stats=np.stack([h.mean(0),h.std(0),h.max(0),h.min(0),z[prefix+'atom_head_entropy'][:,ix].mean(0)],-1)
            local=np.concatenate([v,q,v*q,np.abs(v-q),stats,kind],-1).astype(np.float32)
            atoms.append(local[:16])
            g=np.concatenate([unit(z[prefix+'qpost_pred'])@projection,unit(z[prefix+'qpost_mean'])@projection,unit(z[prefix+'image_mean'])@projection,unit(z['text_mean'])@projection])
            global_values.append(g)
        if (n+1)%500==0: print('load',split,layer,n+1,flush=True)
    width=atoms[0].shape[1]
    x=np.zeros((len(rows),16,width),np.float32); mask=np.zeros((len(rows),16),bool)
    for i,a in enumerate(atoms): x[i,:len(a)]=a; mask[i,:len(a)]=True
    data={'x':x,'mask':mask,'global':np.stack(global_values),'y':np.array([r['label'] for r in rows],np.float32)}
    np.savez_compressed(cache,ids=[r['row_id'] for r in rows],**data)
    return rows,data

class Model(nn.Module):
    def __init__(self,dim,kind,hidden=64):
        super().__init__(); self.kind=kind
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(.15),nn.Linear(hidden,2 if kind.endswith('evidence') else 1))
    def forward(self,x,mask=None):
        out=self.net(x)
        if self.kind.startswith('global'): return out[:,0],None
        if self.kind.endswith('evidence'):
            e=nn.functional.softplus(out)
            total=e.sum(-1)+2
            opposition=e[...,1]/total; ignorance=2/total
            # Independent product random sets, pushed to "any false premise".
            log_nonconflict=(torch.log1p(-opposition.clamp(max=1-1e-7))*mask).sum(-1)
            prob=-torch.expm1(log_nonconflict)
            return torch.logit(prob.clamp(1e-6,1-1e-6)),(ignorance*mask).sum(-1)/mask.sum(-1)
        probability=torch.sigmoid(out[...,0])
        log_nonconflict=(torch.log1p(-probability.clamp(max=1-1e-7))*mask).sum(-1)
        prob=-torch.expm1(log_nonconflict)
        return torch.logit(prob.clamp(1e-6,1-1e-6)),None

def tensors(data,kind,mean=None,std=None):
    x=data['global'] if kind.startswith('global') else data['x']
    if kind=='global_text': x=x[:,-128:]
    if kind.startswith('context_'):
        x=np.concatenate([x,np.broadcast_to(data['global'][:,None,:],(len(x),x.shape[1],data['global'].shape[1]))],-1)
    if mean is None:
        flat=x if kind.startswith('global') else x[data['mask']]
        mean=flat.mean(0); std=np.maximum(flat.std(0),.01)
    x=(x-mean)/std
    return (torch.from_numpy(x.astype(np.float32)),torch.from_numpy(data['mask']),torch.from_numpy(data['y'])),mean,std

def run(train,cal,kind,decay,seed):
    torch.manual_seed(seed)
    (x,m,y),mean,std=tensors(train,kind)
    (cx,cm,cy),_,_=tensors(cal,kind,mean,std)
    model=Model(x.shape[-1],kind)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=decay)
    best=-1; state=None; best_epoch=0
    for epoch in range(1,161):
        model.train(); order=torch.randperm(len(y))
        for ids in order.split(128):
            logits,u=model(x[ids],m[ids]); loss=nn.functional.binary_cross_entropy_with_logits(logits,y[ids])
            # Discourage vacuity only for known normal bags; no fake atom labels.
            if kind.endswith('evidence'): loss=loss+.01*((1-y[ids])*u).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch%5==0:
            model.eval()
            with torch.no_grad(): score=model(cx,cm)[0].numpy()
            auc=float(roc_auc_score(cy.numpy(),score))
            if auc>best: best=auc; state=copy.deepcopy(model.state_dict()); best_epoch=epoch
        if epoch-best_epoch>=35: break
    return {'state':state,'mean':mean,'std':std,'kind':kind,'weight_decay':decay,'seed':seed,'epoch':best_epoch,'calibration_auroc':best}

def predict(data,fit):
    (x,m,y),_,_=tensors(data,fit['kind'],fit['mean'],fit['std'])
    model=Model(x.shape[-1],fit['kind']); model.load_state_dict(fit['state']); model.eval()
    with torch.no_grad(): logits,u=model(x,m)
    return logits.numpy()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['fit','evaluate'],required=True); p.add_argument('--split',choices=['validation','confirmation'],default='validation'); a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261026)
    projection=rng.normal(size=(4096,128)).astype(np.float32)
    nlp=spacy.load('en_core_web_sm',disable=['ner'])
    if a.mode=='fit':
        assert not (OUT/'frozen.json').exists(),'Preserve frozen model'
        report=[]; winners={}
        for layer in [24,31]:
            _,train=load('train',layer,projection,nlp); _,cal=load('calibration',layer,projection,nlp)
            for kind in ['global','global_text','ordinary_mil','evidence','context_ordinary_mil','context_evidence']:
                for decay in [.01,.1]:
                    fit=run(train,cal,kind,decay,20261026)
                    key=f'L{layer}_{kind}_wd{decay}'
                    entry={k:v for k,v in fit.items() if k not in ['state','mean','std']}; entry.update(key=key,layer=layer)
                    print(json.dumps(entry),flush=True); report.append(entry)
                    if kind not in winners or entry['calibration_auroc']>winners[kind]['calibration_auroc']:
                        winners[kind]=entry; torch.save(fit,OUT/f'{kind}.pt')
        frozen={'selection':'official train calibration only; no official test labels used to fit or select','train_rows':2000,'calibration_rows':400,'projection_seed':20261026,'projection_dim':128,'winners':winners,'grid':report,'limitations':'supervised weak bag labels; independence approximation; projection and parser overhead; not proof DST improves rank'}
        (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2)); np.save(OUT/'projection.npy',projection)
    else:
        frozen=json.loads((OUT/'frozen.json').read_text()); report={}
        sys.path.insert(0,str(TASK.parent/'14_multilayer_semantic_evidence/code'))
        from evaluate import metrics
        for kind,entry in frozen['winners'].items():
            fit=torch.load(OUT/f'{kind}.pt',map_location='cpu',weights_only=False)
            rows,data=load(a.split,entry['layer'],projection,nlp)
            score=predict(data,fit)
            report[kind]=metrics(rows,score,bootstrap=True)
            report[kind]['layer']=entry['layer']
            report[kind]['per_conflict_type']={}
            for t in ['object','attribute','relation']:
                ids={r['image_id'] for r in rows if r.get('conflict_type')==t}
                ix=[i for i,r in enumerate(rows) if r['image_id'] in ids]
                if ix: report[kind]['per_conflict_type'][t]=metrics([rows[i] for i in ix],score[ix])
            np.savez_compressed(OUT/f'{a.split}_{kind}.npz',row_ids=[r['row_id'] for r in rows],score=score)
        (OUT/f'{a.split}_metrics.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
