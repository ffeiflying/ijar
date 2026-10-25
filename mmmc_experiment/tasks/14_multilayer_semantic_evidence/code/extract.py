"""One prefill; internal visual semantic evidence at fixed candidate layers.

No gold metadata enters features; no answer tokens are generated. A lexical
readout frame is an experimental approximation, not a validated semantic frame.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import CACHE, EXPERIMENT, SPLITS, TASK2, TASK5, resolve_record_path
os.environ['HF_MODULES_CACHE']=str(CACHE/'hf_modules')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(TASK2/'code'))
sys.path.insert(0,str(TASK5/'code'))
import run_internvl as existing
from evidence_v2 import combine_evidence, fuse_evidence_sources
import numpy as np
import torch

TASK=EXPERIMENT/'tasks/14_multilayer_semantic_evidence'
LAYERS=[0,8,16,20,24,27,31]
STOP=set('what which where how is are was were the a an of in on to does do this that there it and with can you see shown image picture photo photograph at many color kind type made from why who whose has have doing looking look shown taken take else visible above below behind front left right wearing standing'.split())

class Captured(Exception):
    pass

def lexical(tok,tokenid):
    text=tok.decode([tokenid]).strip().lower()
    return bool(re.fullmatch('[a-z]{3,}',text)) and text not in STOP

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['dev','validation'],default='dev')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--stop',type=int,default=0)
    parser.add_argument('--records',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--audit-forward',action='store_true')
    args=parser.parse_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    model=existing.AutoModel.from_pretrained(existing.MODEL_PATH,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True,local_files_only=True).eval().cuda()
    tok=existing.AutoTokenizer.from_pretrained(existing.MODEL_PATH,trust_remote_code=True,use_fast=False,local_files_only=True)
    lm=model.language_model
    dev=json.loads((SPLITS/'dev.json').read_text())
    token_ids=sorted({i for row in dev for i in tok.encode(row['question'],add_special_tokens=False) if lexical(tok,i)})
    frame={'token_ids':token_ids,'tokens':[tok.decode([i]).strip() for i in token_ids],'source':'development questions only, no labels or answers','limitations':'lexical token labels, not validated semantic equivalence classes'}
    frame_path=TASK/'results/frame.json'
    frame_path.parent.mkdir(parents=True,exist_ok=True)
    if frame_path.exists():
        assert json.loads(frame_path.read_text())==frame
    else:
        frame_path.write_text(json.dumps(frame,ensure_ascii=False,indent=2))
    mapping={v:i for i,v in enumerate(token_ids)}
    beta=lm.output.weight.detach()[token_ids].float()
    beta=beta-beta.mean(0,keepdim=True)
    K=len(token_ids)
    state={}
    if args.audit_forward:
        def count_forward(module,inputs):
            state['forward_count']=state.get('forward_count',0)+1
        lm.register_forward_pre_hook(count_forward)
    original_generate=model.generate

    def generate(*pos,**kw):
        ids=kw['input_ids'][0]
        seq=ids.tolist()
        needle=tok.encode(state['question'],add_special_tokens=False)
        matches=[i for i in range(len(seq)-len(needle)+1) if seq[i:i+len(needle)]==needle]
        if not matches:
            needle=needle[1:]
            matches=[i for i in range(len(seq)-len(needle)+1) if seq[i:i+len(needle)]==needle]
        assert matches,'Question span absent'
        qpos=list(range(matches[-1],matches[-1]+len(needle)))
        content=[j for j,i in enumerate(needle) if lexical(tok,i)]
        if not content:
            content=list(range(len(needle)))
        state.update(visual=ids==model.img_context_token_id,qpos=qpos,qids=needle,content=content)
        state['scores']['content_count']=np.array(len(content))
        state['scores']['frame_coverage']=np.array(sum(needle[j] in mapping for j in content)/len(content))
        return original_generate(*pos,**kw)
    model.generate=generate

    def install(layer):
        block=lm.model.layers[layer]
        attn=block.attention
        helper=sys.modules[attn.__class__.__module__]
        buf={}
        def block_pre(module,inputs,kw):
            buf['residual']=inputs[0]
        def pre(module,inputs,kw):
            buf['position_ids']=kw['position_ids']
            buf['mask']=kw.get('attention_mask')
        def qkv_capture(module,inputs,out):
            buf['qkv']=out
        def postattn(module,inputs):
            buf['hidden']=inputs[0][0,state['visual']].detach()
        def finish(module,inputs,out):
            qkv=buf.pop('qkv')
            B,S,_=qkv.shape
            qkv=qkv.reshape(B,S,attn.num_key_value_heads,attn.num_key_value_groups+2,attn.head_dim)
            q=qkv[...,:attn.num_key_value_groups,:].reshape(B,S,attn.num_heads,attn.head_dim).transpose(1,2)
            k=qkv[...,-2,:].transpose(1,2)
            v=qkv[...,-1,:].transpose(1,2)
            cos,sin=attn.rotary_emb(v,seq_len=S)
            q,k=helper.apply_rotary_pos_emb(q,k,cos,sin,buf['position_ids'])
            k=helper.repeat_kv(k,attn.num_key_value_groups)
            v=helper.repeat_kv(v,attn.num_key_value_groups)
            idx=torch.tensor([state['qpos'][j] for j in state['content']],device=q.device)
            logits=q.index_select(2,idx).float()@k.float().transpose(2,3)/math.sqrt(attn.head_dim)
            if buf['mask'] is not None:
                logits+=buf['mask'].index_select(2,idx).float()
            else:
                logits.masked_fill_(torch.arange(S,device=q.device)[None,None,None,:]>idx[None,None,:,None],float('-inf'))
            full_weights=logits.softmax(-1)[0]
            visual_weights=full_weights[:,:,state['visual']]
            weights=visual_weights.mean(0)
            head_conditional=visual_weights/visual_weights.sum(-1,keepdim=True).clamp_min(1e-30)
            conditional=weights/weights.sum(-1,keepdim=True).clamp_min(1e-30)
            topv,topi=conditional.topk(3,dim=-1)
            topw=topv/topv.sum(-1,keepdim=True)
            prefix=f'L{layer}_'
            state['scores'][prefix+'attention_mass']=np.array(weights.sum(-1).mean().item())
            query_ids=torch.tensor([state['qids'][j] for j in state['content']],device=q.device)
            known=[j for j,i in enumerate(query_ids.tolist()) if i in mapping]
            # Image/text attention contributions to the same queried residual.
            vc=torch.einsum('hqs,hsd->qhd',visual_weights,v[0,:,state['visual']].float())
            total=torch.einsum('hqs,hsd->qhd',full_weights,v[0].float())
            tc=total-vc
            residual=buf.pop('residual')[0,idx]
            source_hidden=torch.stack([residual+attn.wo(vc.reshape(len(idx),-1).to(attn.wo.weight.dtype)),residual+attn.wo(tc.reshape(len(idx),-1).to(attn.wo.weight.dtype))],dim=1)
            source_h=lm.model.norm(source_hidden).float()
            source_logits=source_h@beta.T
            source_absolute=source_h.abs()@beta.abs().T
            source_ap=((source_absolute+source_logits)/2).clamp_min(0)
            source_am=((source_absolute-source_logits)/2).clamp_min(0)
            for temperature in [1.,4.,16.]:
                fused=fuse_evidence_sources(source_ap,source_am,temperature=temperature,source_dim=1)
                base=prefix+f'source_T{temperature:g}_'
                state['scores'][base+'cross_cf_mean']=np.array(fused['cf_logscore'].mean().item())
                state['scores'][base+'cross_cf_max']=np.array(fused['cf_logscore'].max().item())
                state['scores'][base+'negative_log_ig']=np.array((-fused['log_ignorance']).mean().item())
                probability=torch.softmax(source_logits/temperature,-1)
                entropy=-(probability*probability.clamp_min(1e-30).log()).sum(-1)/math.log(K)
                confidence=(1-entropy).clamp(0,1)
                kappa=confidence[:,0]*confidence[:,1]*(1-(probability[:,0]*probability[:,1]).sum(-1))
                state['scores'][base+'discounted_cf_mean']=np.array(kappa.mean().item())
                state['scores'][base+'discounted_cf_max']=np.array(kappa.max().item())
            # Post-attention residual is the fixed primary position.
            for position,hidden in [('attn',buf.pop('hidden'))]:
                h=lm.model.norm(hidden).float()
                compact_logits=h@beta.T
                absolute=h.abs()@beta.abs().T
                ap=((absolute+compact_logits)/2).clamp_min(0)
                am=((absolute-compact_logits)/2).clamp_min(0)
                if not state.get('numeric_checked'):
                    explicit=beta*h[0][None,:]
                    error=max((explicit.relu().sum(-1)-ap[0]).abs().max().item(),((-explicit).relu().sum(-1)-am[0]).abs().max().item())
                    assert error<.01,('Positive/negative matrix identity',error)
                    state['scores']['evidence_total_identity_error']=np.array(error)
                pfx=prefix+position+'_'
                # Full-vocabulary likelihood baseline, with no answer generation.
                full=lm.output(h.to(lm.output.weight.dtype)).float()
                query_logp=full[:,query_ids]-full.logsumexp(-1,keepdim=True)
                top_ids=full.topk(10,dim=-1).indices
                match=(top_ids[topi]==query_ids[:,None,None]).any(-1).any(-1)
                state['scores'][pfx+'lexical_hits']=np.array(match.sum().item())
                # Log pooling is stable even for very small vocabulary probabilities.
                baseline={
                    'allpatch':query_logp.max(0).values,
                    'attention':(query_logp.T+conditional.clamp_min(1e-30).log()).logsumexp(-1),
                    'top3':(query_logp.T.gather(1,topi)+topw.clamp_min(1e-30).log()).logsumexp(-1),
                }
                for name,vals in baseline.items():
                    state['scores'][pfx+'nll_'+name+'_mean']=np.array((-vals).mean().item())
                    state['scores'][pfx+'nll_'+name+'_max']=np.array((-vals).max().item())
                for temperature in [1.,4.,16.]:
                    ds=combine_evidence(ap,am,temperature)
                    reconstruction=(ds['normalized_plausibility']-torch.softmax(compact_logits.double()/temperature,-1)).abs().max().item()
                    assert reconstruction<1e-4,('Plausibility reconstruction',layer,reconstruction)
                    state['scores'][pfx+f'T{temperature:g}_reconstruction_error']=np.array(reconstruction)
                    logcf=ds['cf_logscore']
                    all_queries=(conditional.double()*logcf[None,:]).sum(-1)
                    selected=(logcf[topi]*topw.double()).sum(-1)
                    base=pfx+f'T{temperature:g}_'
                    for name,values in [('acf_attention',all_queries),('acf_top3',selected)]:
                        state['scores'][base+name+'_mean']=np.array(values.mean().item())
                        state['scores'][base+name+'_max']=np.array(values.max().item())
                    state['scores'][base+'image_only_acf']=np.array(logcf.mean().item())
                    state['scores'][base+'source_ignorance']=np.array((conditional.double()*ds['ignorance'][None,:]).sum(-1).mean().item())
                    # Explicit singleton+Theta BPA, mixed across correlated patches.
                    # Confidence discount is a heuristic, not semantic calibration.
                    prob=torch.softmax(compact_logits/temperature,-1)
                    entropy=-(prob*prob.clamp_min(1e-30).log()).sum(-1)/math.log(K)
                    reliability=(1-entropy).clamp(0,1)
                    head_prob=torch.einsum('hqv,vk->hqk',head_conditional,prob)
                    head_entropy=-(head_prob*head_prob.clamp_min(1e-30).log()).sum(-1)/math.log(K)
                    head_r=(1-head_entropy).clamp(0,1)
                    masses=head_r[...,None]*head_prob
                    H=head_prob.shape[0]
                    all_pairs=(head_r.sum(0).square()-head_r.square().sum(0))-(masses.sum(0).square().sum(-1)-masses.square().sum((0,2)))
                    interhead=(all_pairs/(H*(H-1))).clamp(0,1)
                    state['scores'][base+'interhead_cf_mean']=np.array(interhead.mean().item())
                    state['scores'][base+'interhead_cf_max']=np.array(interhead.max().item())
                    if known:
                        qi=torch.tensor(known,device=q.device)
                        ci=torch.tensor([mapping[int(query_ids[j])] for j in known],device=q.device)
                        support=prob[:,ci].T
                        for selector,w in [('attention',conditional[qi]),('top3',torch.zeros_like(conditional[qi]).scatter_(1,topi[qi],topw[qi]))]:
                            # Text BPA is dogmatic on the queried lexical label.
                            cf=(w*reliability[None,:]*(1-support)).sum(-1)
                            uncertainty=(w*entropy[None,:]).sum(-1)
                            state['scores'][base+'grounding_cf_'+selector+'_mean']=np.array(cf.mean().item())
                            state['scores'][base+'grounding_cf_'+selector+'_max']=np.array(cf.max().item())
                            state['scores'][base+'grounding_source_ig_'+selector]=np.array(uncertainty.mean().item())
                    else:
                        # Uncovered lexical premise remains unknown, not contradictory.
                        for selector in ['attention','top3']:
                            for pool in ['mean','max']:
                                state['scores'][base+'grounding_cf_'+selector+'_'+pool]=np.array(0.)
                            state['scores'][base+'grounding_source_ig_'+selector]=np.array(1.)
                if state['example']:
                    patches=topi[0].tolist()
                    values,ids=full[patches].topk(5,-1)
                    state['examples'].append({'layer':layer,'position':position,'query':tok.decode([int(query_ids[0])]),'patches':patches,'decoded':[[tok.decode([i]) for i in row] for row in ids.tolist()]})
                del full
            state['numeric_checked']=True
            if layer==max(LAYERS):
                raise Captured()
        block.register_forward_pre_hook(block_pre,with_kwargs=True)
        attn.register_forward_pre_hook(pre,with_kwargs=True)
        attn.wqkv.register_forward_hook(qkv_capture)
        block.ffn_norm.register_forward_pre_hook(postattn)
        block.register_forward_hook(finish)
    for layer in LAYERS:
        install(layer)
    config={'version':2,'layers':LAYERS,'model':existing.MODEL_PATH,'positions':['post-attention residual'],'frame_sha256':hashlib.sha256(frame_path.read_bytes()).hexdigest(),'temperatures':[1,4,16],'query':'alphabetic non-stopword question token; original question','image_tiles':1,'sampling':False,'early_exit':'after final selected block, before final answer logits','mass_assignment':'exact linear-feature DS; exact image/text source DS; entropy-discounted lexical grounding and pairwise head conflict','source_features':'RMSNorm(query residual + W_O(attention contribution from image or text)); shared query residual, correlated sources','dtype':'bfloat16 model, float32 TF32 linear totals, float64 DST','scope':'exploratory dev selection; validation only after freeze'}
    config_path=TASK/'results/config_v2.json'
    if config_path.exists():
        assert json.loads(config_path.read_text())==config
    else:
        config_path.write_text(json.dumps(config,indent=2))
    rows=json.loads((args.records if args.records else SPLITS/f'{args.split}.json').read_text())
    if args.limit:
        rows=rows[:args.limit]
    rows=rows[args.start:args.stop or None]
    dest=args.output_dir if args.output_dir else TASK/f'results/features_v2/{args.split}'
    dest.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for n,row in enumerate(rows):
            path=dest/f'{row["row_id"]}.npz'
            if path.exists():
                continue
            state.clear()
            state.update(question=row['question'],scores={},example=n<4,examples=[])
            start=time.perf_counter()
            pixels=existing.helpers.load_image(str(resolve_record_path(row['image_path'])),max_num=1).to(dtype=torch.bfloat16,device='cuda')
            try:
                model.chat(tok,pixels,row['question'],{'max_new_tokens':1,'do_sample':False})
                raise RuntimeError('Early exit absent')
            except Captured:
                pass
            torch.cuda.synchronize()
            if args.audit_forward:
                assert state['forward_count']==1,('Expected one language forward',state['forward_count'])
                state['scores']['language_forward_count']=np.array(state['forward_count'])
            state['scores']['seconds_all_layers']=np.array(time.perf_counter()-start)
            assert all(np.isfinite(v).all() for v in state['scores'].values()),'Nonfinite scores'
            tmp=path.with_suffix('.partial.npz')
            np.savez_compressed(tmp,**state['scores'])
            tmp.replace(path)
            if state['example']:
                path.with_suffix('.examples.json').write_text(json.dumps({'row_id':row['row_id'],'question':row['question'],'readouts':state['examples']},ensure_ascii=False,indent=2))
            if (n+1)%10==0 or n<2:
                print(args.split,n+1,'/',len(rows),'seconds',round(float(state['scores']['seconds_all_layers']),3),flush=True)
    print('DONE',args.split,len(rows),'frame',K,flush=True)

if __name__=='__main__':
    main()
