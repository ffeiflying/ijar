"""Single prefill, block0 only; frozen intermediate visual logit lens diagnostic."""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'shared'))
from layout import CACHE, EXPERIMENT, SPLITS, TASK2, resolve_record_path
os.environ['HF_MODULES_CACHE'] = str(CACHE / 'hf_modules')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(TASK2 / 'code'))
import run_internvl as existing
import torch

TASK = EXPERIMENT / 'tasks/13_layer0_semantic_lens'
STOP = set('what which where how is are was were the a an of in on to does do this that there it and with can you see shown image picture photo photograph at many color kind type made from why who whose has have doing'.split())

class Captured(Exception):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='dev', choices=['dev', 'validation'])
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--readout', choices=['logit','cosine','centered'], default='logit')
    parser.add_argument('--all-patches', action='store_true')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(42)
    model = existing.AutoModel.from_pretrained(existing.MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True, local_files_only=True).eval().cuda()
    tok = existing.AutoTokenizer.from_pretrained(existing.MODEL_PATH, trust_remote_code=True, use_fast=False, local_files_only=True)
    lm = model.language_model
    block = lm.model.layers[0]
    attn = block.attention
    helper = sys.modules[attn.__class__.__module__]
    if args.readout != 'logit':
        cosine_weights = {}
        for name,weight in [('input_embedding',lm.get_input_embeddings().weight),('unembedding',lm.output.weight)]:
            w = weight.detach().float()
            if args.readout == 'centered':
                w = w-w.mean(0,keepdim=True)
            cosine_weights[name] = torch.nn.functional.normalize(w,dim=-1).to(torch.bfloat16)
    state = {}
    original_generate = model.generate

    def generate(*pos, **kw):
        ids = kw['input_ids'][0]
        state['visual'] = ids == model.img_context_token_id
        seq = ids.tolist()
        needle = tok.encode(state['question'], add_special_tokens=False)
        matches = [i for i in range(len(seq)-len(needle)+1) if seq[i:i+len(needle)] == needle]
        if not matches:
            # Allow only leading tokenizer boundary difference, never a short suffix.
            needle = needle[1:]
            matches = [i for i in range(len(seq)-len(needle)+1) if seq[i:i+len(needle)] == needle]
        assert matches, 'Exact question span absent'
        state['qpos'] = list(range(matches[-1], matches[-1]+len(needle)))
        state['qids'] = seq[matches[-1]:matches[-1]+len(needle)]
        return original_generate(*pos, **kw)
    model.generate = generate

    def block_pre(module, inputs, kw):
        state['input'] = inputs[0][0, state['visual']].detach()

    def attn_pre(module, inputs, kw):
        state['position_ids'] = kw['position_ids']
        state['mask'] = kw.get('attention_mask')

    def qkv_hook(module, inputs, out):
        state['qkv'] = out

    def postattn(module, inputs):
        state['post_attention'] = inputs[0][0, state['visual']].detach()

    def finish(module, inputs, out):
        state['block_output'] = out[0][0, state['visual']].detach()
        B,S,_ = state['qkv'].shape
        qkv = state.pop('qkv').reshape(B,S,attn.num_key_value_heads,attn.num_key_value_groups+2,attn.head_dim)
        q = qkv[...,:attn.num_key_value_groups,:].reshape(B,S,attn.num_heads,attn.head_dim).transpose(1,2)
        k = qkv[...,-2,:].transpose(1,2)
        v = qkv[...,-1,:].transpose(1,2)
        cos,sin = attn.rotary_emb(v,seq_len=S)
        q,k = helper.apply_rotary_pos_emb(q,k,cos,sin,state['position_ids'])
        k = helper.repeat_kv(k,attn.num_key_value_groups)
        idx = torch.tensor(state['qpos'],device=q.device)
        logits = q.index_select(2,idx).float() @ k.float().transpose(2,3) / math.sqrt(attn.head_dim)
        if state['mask'] is not None:
            logits += state['mask'].index_select(2,idx).float()
        else:
            logits.masked_fill_(torch.arange(S,device=q.device)[None,None,None,:] > idx[None,None,:,None], float('-inf'))
        weights = logits.softmax(-1)[0,:,:,state['visual']].mean(0)
        records = []
        selected = set()
        for qi, tokenid in enumerate(state['qids']):
            text = tok.decode([tokenid]).strip().lower()
            if not re.search('[a-zA-Z]',text) or text in STOP:
                continue
            vals, patches = weights[qi].topk(3)
            records.append({'token':text,'token_id':tokenid,'patches':patches.tolist(),'attention':vals.tolist(),'visual_attention_mass':weights[qi].sum().item()})
            selected.update(patches.tolist())
        _, patches = weights.mean(0).topk(3)
        selected.update(patches.tolist())
        selected = sorted(selected)
        if args.all_patches:
            selected = list(range(weights.shape[-1]))
        result = {'layer':0,'question':state['question'],'content_tokens':records,'question_mean_patches':patches.tolist(),'visual_token_count':weights.shape[-1],'positions':{}}
        ix = torch.tensor(selected,device=q.device)
        for position in ['input','post_attention','block_output']:
            hidden = state[position].index_select(0,ix)
            if args.readout != 'logit':
                if args.readout == 'centered':
                    hidden = hidden-state[position].mean(0,keepdim=True)
                normalized = torch.nn.functional.normalize(hidden.float(),dim=-1).to(torch.bfloat16)
                for name,weight in cosine_weights.items():
                    values,ids = (normalized @ weight.T).float().topk(10,dim=-1)
                    result['positions'][position+'_'+name] = {str(patch):{'top_tokens':[tok.decode([i]) for i in ids[j].tolist()],'top_ids':ids[j].tolist(),'top_cosines':values[j].tolist()} for j,patch in enumerate(selected)}
                continue
            # Architecture-matched final RMSNorm; no final layer computation.
            lens = lm.output(lm.model.norm(hidden)).float()
            probs = lens.softmax(-1)
            top_probs, top_ids = probs.topk(10,dim=-1)
            entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
            result['positions'][position] = {str(patch):{'top_tokens':[tok.decode([i]) for i in top_ids[j].tolist()],'top_ids':top_ids[j].tolist(),'top_probs':top_probs[j].tolist(),'normalized_entropy':entropy[j].item()/math.log(probs.shape[-1])} for j,patch in enumerate(selected)}
        state['result'] = result
        raise Captured()

    block.register_forward_pre_hook(block_pre,with_kwargs=True)
    attn.register_forward_pre_hook(attn_pre,with_kwargs=True)
    attn.wqkv.register_forward_hook(qkv_hook)
    block.ffn_norm.register_forward_pre_hook(postattn)
    block.register_forward_hook(finish)
    rows = json.loads((SPLITS/f'{args.split}.json').read_text())
    if args.limit:
        rows = rows[:args.limit]
    variant = 'diagnostics' if args.readout == 'logit' else args.readout+'_diagnostics'
    if args.all_patches:
        variant = 'allpatch_'+variant
    dest = TASK/f'results/{variant}/{args.split}'
    dest.mkdir(parents=True,exist_ok=True)
    config = {'model':existing.MODEL_PATH,'layer':0,'positions':['input','post_attention','block_output'],'lens':'frozen output(final RMSNorm(visual residual state))','query':'non-stopword question subword tokens; mean over heads; top3 visual patches','max_image_tiles':1,'answer_sampling':False,'early_exit':'block0 output','labels_used_for_extraction':False,'purpose':'semantic recovery diagnostic, not validated DS masses'}
    if args.readout != 'logit':
        config['lens'] = 'cosine similarity to frozen input embeddings and unembedding rows; no probability or mass interpretation'
    if args.readout == 'centered':
        config['centering'] = 'image-wise visual patch mean and vocabulary-wise token embedding mean; no fitted parameters'
    config_path = TASK/('results/config.json' if args.readout == 'logit' else f'results/{args.readout}_config.json')
    if args.all_patches:
        config['coverage_control'] = 'decode all visual patches, retaining original attention selection for paired comparison'
        config_path = TASK/f'results/{variant}_config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config
    else:
        config_path.write_text(json.dumps(config,indent=2))
    with torch.inference_mode():
        for n,row in enumerate(rows):
            path = dest/f'{row["row_id"]}.json'
            if path.exists():
                continue
            state.clear()
            state['question'] = row['question']
            start = time.perf_counter()
            pixels = existing.helpers.load_image(str(resolve_record_path(row['image_path'])),max_num=1).to(dtype=torch.bfloat16,device='cuda')
            try:
                model.chat(tok,pixels,row['question'],{'max_new_tokens':1,'do_sample':False})
                raise RuntimeError('Block0 hook did not terminate forward')
            except Captured:
                pass
            torch.cuda.synchronize()
            result = state.pop('result')
            result.update(row_id=row['row_id'],image_id=row['image_id'],seconds=time.perf_counter()-start)
            tmp = path.with_suffix('.partial')
            tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2))
            tmp.replace(path)
            print(json.dumps({'n':n+1,'row_id':row['row_id'],'seconds':round(result['seconds'],3)},ensure_ascii=False),flush=True)
    print('DONE',args.split,len(rows),flush=True)

if __name__ == '__main__':
    main()
