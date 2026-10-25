"""Deploy frozen background-referenced CF: one vision+language prefill.

Stops after the selected block's attention. Never generates answer tokens.
No task labels/answers enter prediction. Full model checkpoint stays unchanged.
"""
import argparse
import hashlib
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
import torch
TASK=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(TASK.parent/'14_multilayer_semantic_evidence/code'))
from extract import lexical

class Captured(Exception):
    pass

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['dev','validation'],default='validation')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--records',type=Path)
    parser.add_argument('--baseline-layer0',action='store_true')
    args=parser.parse_args()
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(42)
    frozen_path=TASK/'results/frozen_dev_selection.json'
    frozen=json.loads(frozen_path.read_text())
    key=frozen['best_cf']
    layer_str,readout,alpha_str,statistic=key.split('/')
    layer=int(layer_str[1:])
    alpha=float(alpha_str[1:])
    assert readout=='attention' and statistic=='cf','This deployment implements the frozen attention CF winner'
    if args.baseline_layer0:
        layer=0
        key=f'L0/{readout}/{alpha_str}/{statistic}'
    bgpath=TASK/'results/background.npz'
    assert hashlib.sha256(bgpath.read_bytes()).hexdigest()==frozen['background_sha256']
    with np.load(bgpath) as z:
        prior=torch.tensor(z[f'L{layer}'],dtype=torch.float64,device='cuda').clamp_min(1e-300).log()*alpha
    prior-=prior.logsumexp(0)
    model=existing.AutoModel.from_pretrained(existing.MODEL_PATH,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True,local_files_only=True).eval().cuda()
    tok=existing.AutoTokenizer.from_pretrained(existing.MODEL_PATH,trust_remote_code=True,use_fast=False,local_files_only=True)
    lm=model.language_model
    block=lm.model.layers[layer]
    attn=block.attention
    helper=sys.modules[attn.__class__.__module__]
    state={}
    old_generate=model.generate
    def generate(*a,**kw):
        seq=kw['input_ids'][0].tolist()
        state['visual']=kw['input_ids'][0]==model.img_context_token_id
        needle=tok.encode(state['question'],add_special_tokens=False)
        matches=[i for i in range(len(seq)-len(needle)+1) if seq[i:i+len(needle)]==needle]
        if not matches:
            needle=needle[1:]
            matches=[i for i in range(len(seq)-len(needle)+1) if seq[i:i+len(needle)]==needle]
        assert matches,'Question span absent'
        content=[j for j,i in enumerate(needle) if lexical(tok,i)]
        if not content:
            content=list(range(len(needle)))
        state['qpos']=[matches[-1]+j for j in content]
        state['qids']=[needle[j] for j in content]
        return old_generate(*a,**kw)
    model.generate=generate
    def count_lm(module,inputs):
        state['lm_forwards']=state.get('lm_forwards',0)+1
    def count_vision(module,inputs):
        state['vision_forwards']=state.get('vision_forwards',0)+1
    lm.register_forward_pre_hook(count_lm)
    model.vision_model.register_forward_pre_hook(count_vision)
    def attn_pre(module,inputs,kw):
        state['position_ids']=kw['position_ids']
        state['mask']=kw.get('attention_mask')
    def qkv_capture(module,inputs,out):
        state['qkv']=out
    def finish(module,inputs):
        hidden=inputs[0][0,state['visual']]
        B,S,_=state['qkv'].shape
        qkv=state.pop('qkv').reshape(B,S,attn.num_key_value_heads,attn.num_key_value_groups+2,attn.head_dim)
        q=qkv[...,:attn.num_key_value_groups,:].reshape(B,S,attn.num_heads,attn.head_dim).transpose(1,2)
        k=qkv[...,-2,:].transpose(1,2)
        v=qkv[...,-1,:].transpose(1,2)
        cos,sin=attn.rotary_emb(v,seq_len=S)
        q,k=helper.apply_rotary_pos_emb(q,k,cos,sin,state['position_ids'])
        k=helper.repeat_kv(k,attn.num_key_value_groups)
        idx=torch.tensor(state['qpos'],device=q.device)
        logits=q.index_select(2,idx).float()@k.float().transpose(2,3)/math.sqrt(attn.head_dim)
        if state['mask'] is not None:
            logits+=state['mask'].index_select(2,idx).float()
        else:
            logits.masked_fill_(torch.arange(S,device=q.device)[None,None,None,:]>idx[None,None,:,None],float('-inf'))
        weights=logits.softmax(-1)[0,:,:,state['visual']].mean(0)
        conditional=weights/weights.sum(-1,keepdim=True).clamp_min(1e-30)
        visual_logits=lm.output(lm.model.norm(hidden)).float()
        ids=torch.tensor(state['qids'],device=q.device)
        logp=visual_logits[:,ids]-visual_logits.logsumexp(-1,keepdim=True)
        log_support=(logp.T+conditional.clamp_min(1e-30).log()).logsumexp(-1)
        # Same operation order as offline evaluation (float32 mean NLL, float64 reference).
        contrast=(-log_support).mean().double()+prior[ids].mean()
        cf_log=contrast.clamp_min(0)
        cf=-torch.expm1(-cf_log)
        ignorance=torch.exp(-contrast.abs())
        support=-torch.expm1(contrast.clamp_max(0))
        assert torch.allclose(cf+ignorance+support,torch.ones_like(cf),atol=1e-10)
        state['prediction']={'cf':cf.item(),'cf_logscore':cf_log.item(),'ignorance':ignorance.item(),'support_premise':support.item(),'contrast':contrast.item(),'query_tokens':[tok.decode([i]) for i in state['qids']]}
        raise Captured()
    attn.register_forward_pre_hook(attn_pre,with_kwargs=True)
    attn.wqkv.register_forward_hook(qkv_capture)
    block.ffn_norm.register_forward_pre_hook(finish)
    rows=json.loads((args.records if args.records else SPLITS/f'{args.split}.json').read_text())
    if args.limit:
        rows=rows[:args.limit]
    output=args.output if args.output else TASK/f'results/detector_{args.split}.jsonl'
    output.parent.mkdir(parents=True,exist_ok=True)
    done={json.loads(line)['row_id'] for line in output.read_text().splitlines()} if output.exists() else set()
    config={'selected':key,'frozen_selection_sha256':hashlib.sha256(frozen_path.read_bytes()).hexdigest(),'split':args.split,'model':existing.MODEL_PATH,'early_exit':f'block{layer} post-attention before FFN','one_forward_audited':True,'answer_generation':False,'max_image_tiles':1,'label_use':False}
    if args.records:
        config.update(split=args.records.stem,records_sha256=hashlib.sha256(args.records.read_bytes()).hexdigest())
    if args.baseline_layer0:
        config['role']='fixed-configuration block0 baseline, no test tuning'
    config_path=output.with_suffix('.config.json')
    if config_path.exists():
        assert json.loads(config_path.read_text())==config
    else:
        config_path.write_text(json.dumps(config,indent=2))
    with output.open('a',buffering=1) as stream,torch.inference_mode():
        for n,row in enumerate(rows):
            if row['row_id'] in done:
                continue
            state.clear()
            state['question']=row['question']
            torch.cuda.synchronize()
            start=time.perf_counter()
            pixels=existing.helpers.load_image(str(resolve_record_path(row['image_path'])),max_num=1).to(dtype=torch.bfloat16,device='cuda')
            try:
                model.chat(tok,pixels,row['question'],{'max_new_tokens':1,'do_sample':False})
                raise RuntimeError('Early exit absent')
            except Captured:
                pass
            torch.cuda.synchronize()
            assert state['lm_forwards']==state['vision_forwards']==1
            result={'row_id':row['row_id'],'image_id':row['image_id'],'layer':layer,'seconds':time.perf_counter()-start,'language_forwards':state['lm_forwards'],'vision_forwards':state['vision_forwards'],**state['prediction']}
            stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            if (n+1)%25==0 or n<2:
                print('detector',n+1,'/',len(rows),'seconds',round(result['seconds'],3),flush=True)
    print('DONE',output,flush=True)

if __name__=='__main__':
    main()
