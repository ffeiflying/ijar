"""One VLM prefill: atom/head semantic evidence plus probe control features."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import CACHE,SPLITS,TASK2,resolve_record_path
os.environ['HF_MODULES_CACHE']=str(CACHE/'hf_modules')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(TASK2/'code'))
import run_internvl as existing
import numpy as np
import spacy
import torch
from parse_atoms import parse
TASK=Path(__file__).resolve().parent.parent
LAYERS=[0,16,20,24,27,31]

class Captured(Exception):
    pass

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['dev','validation','train','calibration','confirmation'],default='dev')
    parser.add_argument('--records',type=Path)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--stop',type=int,default=0)
    parser.add_argument('--layers',default=','.join(map(str,LAYERS)))
    args=parser.parse_args()
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(42)
    layers=[int(v) for v in args.layers.split(',')]
    assert layers==sorted(set(layers)) and all(0<=v<32 for v in layers)
    tokenizer=existing.AutoTokenizer.from_pretrained(existing.MODEL_PATH,trust_remote_code=True,use_fast=False,local_files_only=True)
    nlp=spacy.load('en_core_web_sm',disable=['ner'])
    records_path=args.records if args.records else SPLITS/f'{args.split}.json'
    rows=json.loads(records_path.read_text())
    if args.limit:
        rows=rows[:args.limit]
    rows=rows[args.start:args.stop or None]
    dest=TASK/f'results/features/{args.split}'
    dest.mkdir(parents=True,exist_ok=True)
    metadata={}
    for row in rows:
        path=dest/f'{row["row_id"]}.atoms.json'
        if path.exists():
            meta=json.loads(path.read_text())
            assert meta['question']==row['question']
        else:
            meta=parse(row['question'],tokenizer,nlp)
            assert meta['alignment_fraction']>.9,('Question alignment',meta)
            path.write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        metadata[row['row_id']]=meta
    del nlp
    model=existing.AutoModel.from_pretrained(existing.MODEL_PATH,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True,local_files_only=True).eval().cuda()
    lm=model.language_model
    state={}
    old_generate=model.generate
    def generate(*pos,**kw):
        ids=kw['input_ids'][0]
        seq=ids.tolist()
        needle=state['meta']['qids']
        drop=0
        matches=[i for i in range(len(seq)-len(needle)+1) if seq[i:i+len(needle)]==needle]
        if not matches:
            needle=needle[1:]
            drop=1
            matches=[i for i in range(len(seq)-len(needle)+1) if seq[i:i+len(needle)]==needle]
        assert matches,'Question span not found'
        qpos=list(range(matches[-1],matches[-1]+len(needle)))
        positions=[]
        for atom in state['meta']['atoms']:
            indices=[j-drop for j in atom['qindices'] if j>=drop]
            positions.append(indices if indices else list(range(len(needle))))
        state.update(visual=ids==model.img_context_token_id,qpos=qpos,qids=needle,atom_positions=positions,prediction=len(seq)-1)
        embed=lm.get_input_embeddings().weight[needle].detach()
        state['data']['text_tokens']=embed.cpu().to(torch.float16).numpy()
        state['data']['text_mean']=embed.float().mean(0).cpu().to(torch.float16).numpy()
        state['data']['qids']=np.array(needle,dtype=np.int32)
        return old_generate(*pos,**kw)
    model.generate=generate
    def count(module,inputs):
        state['forwards']=state.get('forwards',0)+1
    lm.register_forward_pre_hook(count)
    def install(layer):
        block=lm.model.layers[layer]
        attn=block.attention
        helper=sys.modules[attn.__class__.__module__]
        buf={}
        def pre(module,inputs,kw):
            buf['position_ids']=kw['position_ids']
            buf['mask']=kw.get('attention_mask')
        def qkv_hook(module,inputs,out):
            buf['qkv']=out
        def post_attention(module,inputs):
            all_hidden=inputs[0][0]
            visual=state['visual']
            qpos=state['qpos']
            h=lm.model.norm(all_hidden[visual])
            qh=lm.model.norm(all_hidden[qpos])
            pfx=f'L{layer}_'
            data=state['data']
            data[pfx+'qtokens']=qh.detach().cpu().to(torch.float16).numpy()
            data[pfx+'qpost_pred']=lm.model.norm(all_hidden[state['prediction']]).detach().cpu().to(torch.float16).numpy()
            data[pfx+'qpost_mean']=qh.float().mean(0).cpu().to(torch.float16).numpy()
            data[pfx+'image_mean']=h.float().mean(0).cpu().to(torch.float16).numpy()
            B,S,_=buf['qkv'].shape
            qkv=buf.pop('qkv').reshape(B,S,attn.num_key_value_heads,attn.num_key_value_groups+2,attn.head_dim)
            q=qkv[...,:attn.num_key_value_groups,:].reshape(B,S,attn.num_heads,attn.head_dim).transpose(1,2)
            k=qkv[...,-2,:].transpose(1,2)
            v=qkv[...,-1,:].transpose(1,2)
            cos,sin=attn.rotary_emb(v,seq_len=S)
            q,k=helper.apply_rotary_pos_emb(q,k,cos,sin,buf['position_ids'])
            k=helper.repeat_kv(k,attn.num_key_value_groups)
            idx=torch.tensor(qpos,device=q.device)
            logits=q.index_select(2,idx).float()@k.float().transpose(2,3)/math.sqrt(attn.head_dim)
            if buf['mask'] is not None:
                logits+=buf['mask'].index_select(2,idx).float()
            else:
                logits.masked_fill_(torch.arange(S,device=q.device)[None,None,None,:]>idx[None,None,:,None],float('-inf'))
            weights=logits.softmax(-1)[0,:,:,visual]
            conditional=weights/weights.sum(-1,keepdim=True).clamp_min(1e-30)
            mean_conditional=weights.mean(0)/weights.mean(0).sum(-1,keepdim=True).clamp_min(1e-30)
            full=lm.output(h).float()
            norm=full.logsumexp(-1)
            qids=torch.tensor(state['qids'],device=q.device)
            patch_logp=(full[:,qids]-norm[:,None]).T
            head_logp=(patch_logp[None,:,:]+conditional.clamp_min(1e-30).log()).logsumexp(-1)
            data[pfx+'head_logp']=head_logp.cpu().numpy()
            data[pfx+'patch_logp']=patch_logp.cpu().numpy()
            data[pfx+'mean_attention']=mean_conditional.cpu().numpy()
            data[pfx+'head_visual_mass']=weights.sum(-1).cpu().numpy()
            data[pfx+'vcontext']=(mean_conditional@h.float()).cpu().to(torch.float16).numpy()
            atom_conditional=torch.stack([conditional[:,indices].mean(1) for indices in state['atom_positions']],dim=1)
            atom_logp=[]
            alias_ids=[]
            for atom,indices in zip(state['meta']['atoms'],state['atom_positions']):
                aliases=atom['aliases']
                if aliases:
                    value=full[:,aliases].logsumexp(-1)-norm
                else:
                    value=patch_logp[indices].mean(0)
                atom_logp.append(value)
                alias_ids.append(aliases)
            atom_logp=torch.stack(atom_logp)
            atom_head=(atom_logp[None,:,:]+atom_conditional.clamp_min(1e-30).log()).logsumexp(-1)
            atom_mean=atom_conditional.mean(0)
            data[pfx+'atom_head_logp']=atom_head.cpu().numpy()
            data[pfx+'atom_patch_logp']=atom_logp.cpu().numpy()
            data[pfx+'atom_attention']=atom_mean.cpu().numpy()
            data[pfx+'atom_qcontext']=torch.stack([qh[indices].float().mean(0) for indices in state['atom_positions']]).cpu().to(torch.float16).numpy()
            data[pfx+'atom_vcontext']=(atom_mean@h.float()).cpu().to(torch.float16).numpy()
            side=round(h.shape[0]**.5)
            assert side*side==h.shape[0]
            grid=torch.stack(torch.meshgrid(torch.linspace(0,1,side,device=h.device),torch.linspace(0,1,side,device=h.device),indexing='ij'),-1).reshape(-1,2)
            data[pfx+'atom_head_centroids']=(atom_conditional@grid).cpu().numpy()
            data[pfx+'atom_head_entropy']=(-(atom_conditional*atom_conditional.clamp_min(1e-30).log()).sum(-1)/math.log(h.shape[0])).cpu().numpy()
        def block_end(module,inputs,out):
            hidden=out[0][0]
            pfx=f'L{layer}_'
            state['data'][pfx+'qblock_pred']=lm.model.norm(hidden[state['prediction']]).detach().cpu().to(torch.float16).numpy()
            state['data'][pfx+'qblock_mean']=lm.model.norm(hidden[state['qpos']]).float().mean(0).cpu().to(torch.float16).numpy()
            if layer==max(layers):
                raise Captured()
        attn.register_forward_pre_hook(pre,with_kwargs=True)
        attn.wqkv.register_forward_hook(qkv_hook)
        block.ffn_norm.register_forward_pre_hook(post_attention)
        block.register_forward_hook(block_end)
    for layer in layers:
        install(layer)
    config={'version':1,'layers':layers,'position':'post-attention semantic lens; post-block probe controls','parse':'spaCy en_core_web_sm; object/attribute/relation/count atoms, no labels','aliases':'single-token surface/lemma and small fixed synonym map; otherwise subword mean affinity','forward':'one vision+language prefill; stop after block31, no final-answer logits','source':'original image/question','model':existing.MODEL_PATH,'environment':'existing internvl, unchanged','max_image_tiles':1}
    cp=TASK/('results/feature_config.json' if args.split in ['dev','validation'] else f'results/feature_config_{args.split}.json')
    if cp.exists():
        assert json.loads(cp.read_text())==config
    else:
        cp.write_text(json.dumps(config,indent=2))
    with torch.inference_mode():
        for n,row in enumerate(rows):
            path=dest/f'{row["row_id"]}.npz'
            if path.exists():
                continue
            state.clear()
            state.update(meta=metadata[row['row_id']],data={})
            start=time.perf_counter()
            pixels=existing.helpers.load_image(str(resolve_record_path(row['image_path'])),max_num=1).to(dtype=torch.bfloat16,device='cuda')
            try:
                model.chat(tokenizer,pixels,row['question'],{'max_new_tokens':1,'do_sample':False})
                raise RuntimeError('Early exit absent')
            except Captured:
                pass
            torch.cuda.synchronize()
            assert state['forwards']==1
            state['data']['language_forward_count']=np.array(state['forwards'])
            state['data']['seconds']=np.array(time.perf_counter()-start)
            assert all(np.isfinite(v).all() for v in state['data'].values())
            tmp=path.with_suffix('.partial.npz')
            np.savez_compressed(tmp,**state['data'])
            tmp.replace(path)
            if (n+1)%25==0 or n<2:
                print(args.split,n+1,'/',len(rows),'seconds',round(float(state['data']['seconds']),3),'atoms',len(state['meta']['atoms']),flush=True)
    print('DONE',args.split,len(rows),flush=True)

if __name__=='__main__':
    main()
