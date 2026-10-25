"""Frozen one-prefill local visual counterevidence at each input token's predecessor.
Reads internal attention states, never generates or evaluates an answer.
"""
import argparse,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
TASK=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(TASK.parents[1]/'shared'))
from layout import CACHE,SPLITS,TASK2,resolve_record_path
os.environ['HF_MODULES_CACHE']=str(CACHE/'hf_modules');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(TASK2/'code'));import run_internvl as existing
class Captured(Exception):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','validation'],default='dev');p.add_argument('--gpu',type=int,default=1);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    torch.cuda.set_device(a.gpu);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    layers=[0,8,16,20,24,27,31]
    tok=existing.AutoTokenizer.from_pretrained(existing.MODEL_PATH,trust_remote_code=True,use_fast=False,local_files_only=True)
    rows=json.loads((SPLITS/f'{a.split}.json').read_text());rows=rows[:a.limit] if a.limit else rows
    dest=TASK/f'results/prequential/{a.split}';dest.mkdir(parents=True,exist_ok=True)
    model=existing.AutoModel.from_pretrained(existing.MODEL_PATH,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True,local_files_only=True).eval().cuda();lm=model.language_model
    state={};original=model.generate
    def generate(*args,**kw):
        ids=kw['input_ids'][0];seq=ids.tolist();needle=tok.encode(state['row']['question'],add_special_tokens=False);drop=0
        matches=[i for i in range(len(seq)-len(needle)+1) if seq[i:i+len(needle)]==needle]
        if not matches:
            needle=needle[1:];drop=1;matches=[i for i in range(len(seq)-len(needle)+1) if seq[i:i+len(needle)]==needle]
        assert matches
        positions=[i-1 for i in range(matches[-1],matches[-1]+len(needle))]
        assert min(positions)>=0
        state.update(qpos=positions,qids=needle,drop=drop,visual=ids==model.img_context_token_id)
        state['data']['qids']=np.array(needle);state['data']['qindex_drop']=np.array(drop)
        return original(*args,**kw)
    model.generate=generate
    def count_lm(module,inputs):state['lm_calls']=state.get('lm_calls',0)+1
    def count_vision(module,inputs):state['vision_calls']=state.get('vision_calls',0)+1
    lm.register_forward_pre_hook(count_lm);model.vision_model.register_forward_pre_hook(count_vision)
    def install(layer):
        block=lm.model.layers[layer];attn=block.attention;helper=sys.modules[attn.__class__.__module__];buf={}
        def block_pre(module,inputs,kw):
            h=inputs[0] if inputs else kw['hidden_states'];buf['base']=h[0,state['qpos']].detach().float()
        def attn_pre(module,inputs,kw):buf.update(position_ids=kw['position_ids'],mask=kw.get('attention_mask'))
        def qkv_hook(module,inputs,out):buf['qkv']=out
        def attn_out(module,inputs,out):buf['attention']=out[0][0,state['qpos']].detach().float()
        def read(module,inputs):
            hpost=inputs[0][0,state['qpos']];B,S,_=buf['qkv'].shape
            qkv=buf.pop('qkv').reshape(B,S,attn.num_key_value_heads,attn.num_key_value_groups+2,attn.head_dim)
            q=qkv[...,:attn.num_key_value_groups,:].reshape(B,S,attn.num_heads,attn.head_dim).transpose(1,2)
            k=qkv[...,-2,:].transpose(1,2);v=qkv[...,-1,:].transpose(1,2)
            cos,sin=attn.rotary_emb(v,seq_len=S);q,k=helper.apply_rotary_pos_emb(q,k,cos,sin,buf['position_ids'])
            k=helper.repeat_kv(k,attn.num_key_value_groups);v=helper.repeat_kv(v,attn.num_key_value_groups)[0].float()
            idx=torch.tensor(state['qpos'],device=q.device);logits=q.index_select(2,idx).float()@k.float().transpose(2,3)/math.sqrt(attn.head_dim)
            if buf['mask'] is not None:logits+=buf['mask'].index_select(2,idx).float()
            else:logits.masked_fill_(torch.arange(S,device=q.device)[None,None,None,:]>idx[None,None,:,None],float('-inf'))
            weights=logits.softmax(-1)[0];vis=state['visual']
            iv=weights[:,:,vis]@v[:,vis];it=weights[:,:,~vis]@v[:,~vis]
            def project(x):return torch.nn.functional.linear(x.transpose(0,1).reshape(len(idx),-1),attn.wo.weight.float(),None)
            ov=project(iv);ot=project(it)
            if attn.wo.bias is not None:ot+=attn.wo.bias.float()
            actual=buf.pop('attention');error=(ov+ot-actual).norm()/actual.norm().clamp_min(1e-12);assert error<.03
            prefix=f'L{layer}_';data=state['data'];data[prefix+'reconstruction_error']=error.cpu().numpy()
            base=buf.pop('base')
            views={'base':base,'visual':base+ov,'text':base+ot,'joint':hpost.float()}
            target=torch.tensor(state['qids'],device=q.device);row=torch.arange(len(target),device=q.device)
            for name,value in views.items():
                readout=lm.output(lm.model.norm(value.to(hpost.dtype))).float()
                logp=readout[row,target]-readout.logsumexp(-1)
                data[prefix+name+'_logp']=logp.cpu().numpy()
            data[prefix+'visual_mass']=weights[:,:,vis].sum(-1).mean(0).cpu().numpy()
            if layer==max(layers):raise Captured()
        block.register_forward_pre_hook(block_pre,with_kwargs=True);attn.register_forward_pre_hook(attn_pre,with_kwargs=True);attn.wqkv.register_forward_hook(qkv_hook);attn.register_forward_hook(attn_out);block.ffn_norm.register_forward_pre_hook(read)
    for layer in layers:install(layer)
    config={'version':1,'layers':layers,'readout_position':'post-attention at predecessor of each literal input-question token','hypothesis':'local visual contribution raises/lowers observed-token plausibility; not full image/no-image causal likelihood','views':'base,base+actual visual AV,base+actual nonvisual AV,actual post-attention state','forward':'one vision and one language prefill; stop before block31 FFN; no generated answers','training':False,'model':existing.MODEL_PATH}
    cp=TASK/'results/prequential/config.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    else:cp.parent.mkdir(parents=True,exist_ok=True);cp.write_text(json.dumps(config,indent=2))
    with torch.inference_mode():
        for i,r in enumerate(rows):
            out=dest/f'{r["row_id"]}.npz'
            if out.exists():continue
            state.clear();state.update(row=r,data={});start=time.perf_counter()
            image=existing.helpers.load_image(str(resolve_record_path(r['image_path'])),max_num=1).to(dtype=torch.bfloat16,device='cuda')
            try:model.chat(tok,image,r['question'],{'max_new_tokens':1,'do_sample':False});raise RuntimeError('Early exit absent')
            except Captured:pass
            torch.cuda.synchronize();assert state['lm_calls']==state['vision_calls']==1
            data=state['data'];data.update(language_forward_count=np.array(state['lm_calls']),vision_forward_count=np.array(state['vision_calls']),seconds=np.array(time.perf_counter()-start))
            assert all(np.isfinite(x).all() for x in data.values())
            tmp=out.with_suffix('.partial.npz');np.savez_compressed(tmp,**data);tmp.replace(out)
            if i<2 or (i+1)%25==0:print(a.split,i+1,'/',len(rows),'seconds',round(float(data['seconds']),3),flush=True)
    print('DONE',a.split,len(rows),flush=True)
if __name__=='__main__':main()
