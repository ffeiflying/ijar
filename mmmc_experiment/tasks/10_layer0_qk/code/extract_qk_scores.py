"""Exact attention sum partition at block0; no output-head score."""
import argparse,json,math,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import SPLITS,CACHE,TASK2,TASK5,EXPERIMENT,resolve_record_path
TASK=EXPERIMENT/'tasks/10_layer0_qk'
os.environ['HF_MODULES_CACHE']=str(CACHE/'hf_modules');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(TASK2/'code'))
import numpy as np
import torch
import run_internvl as existing
sys.path.insert(0,str(TASK5/'code'))
from evidence_v2 import combine_evidence
p=argparse.ArgumentParser();p.add_argument('--splits',default='dev,validation');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
model=existing.AutoModel.from_pretrained(existing.MODEL_PATH,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True,local_files_only=True).eval().cuda()
tokenizer=existing.AutoTokenizer.from_pretrained(existing.MODEL_PATH,trust_remote_code=True,use_fast=False,local_files_only=True)
attn=model.language_model.model.layers[0].attention;helper=sys.modules[attn.__class__.__module__]
state={};current_question=[''];output={}
original_generate=model.generate

def wrapped_generate(*args,**kwargs):
 ids=kwargs['input_ids'][0];state['input_ids']=ids.tolist();state['visual_mask']=ids==model.img_context_token_id
 seq=ids.tolist();needle=tokenizer.encode(current_question[0],add_special_tokens=False);found=None
 for tail in [needle,needle[1:],needle[-3:],needle[-2:]]:
  if not tail:continue
  matches=[i for i in range(len(seq)-len(tail)+1) if seq[i:i+len(tail)]==tail]
  if matches:found=list(range(matches[-1],matches[-1]+len(tail)));break
 assert found is not None,'Question token span not found'
 state['question_positions']=found;state['question_exact']=len(found)>=len(needle)-1
 return original_generate(*args,**kwargs)
model.generate=wrapped_generate

def pre_hook(module,args,kwargs):
 state['position_ids']=kwargs['position_ids'];state['mask']=kwargs.get('attention_mask')
def qkv_hook(module,args,out):state['qkv']=out
attn.register_forward_pre_hook(pre_hook,with_kwargs=True);attn.wqkv.register_forward_hook(qkv_hook)

def split_hook(module,args):
 qkv=state.pop('qkv');B,S,_=qkv.shape;assert B==1
 qkv=qkv.reshape(B,S,attn.num_key_value_heads,attn.num_key_value_groups+2,attn.head_dim)
 q=qkv[...,:attn.num_key_value_groups,:].reshape(B,S,attn.num_heads,attn.head_dim).transpose(1,2)
 k=qkv[...,-2,:].transpose(1,2);v=qkv[...,-1,:].transpose(1,2)
 cos,sin=attn.rotary_emb(v,seq_len=S)
 q,k=helper.apply_rotary_pos_emb(q,k,cos,sin,state['position_ids'])
 k=helper.repeat_kv(k,attn.num_key_value_groups);v=helper.repeat_kv(v,attn.num_key_value_groups)
 question=state['question_positions'];indices=question+[S-1]
 idx=torch.tensor(indices,device=q.device);q=q.index_select(2,idx)
 logits=(q@k.transpose(2,3))/math.sqrt(attn.head_dim)
 if state['mask'] is not None:logits=logits+state['mask'].index_select(2,idx)
 else:logits=logits.masked_fill(torch.arange(S,device=q.device)[None,None,None,:]>idx[None,None,:,None],torch.finfo(logits.dtype).min)
 weights=logits.softmax(-1,dtype=torch.float32).to(q.dtype)
 visual_keys=k[0,:,state['visual_mask'],:]
 # Keys are the categorical attention targets; features are query coordinates.
 beta=visual_keys.float()-visual_keys.float().mean(1,keepdim=True)
 terms=q[0].float()[:,:,None,:]*beta[:,None,:,:]/math.sqrt(attn.head_dim)
 positive=terms.relu().sum(-1,dtype=torch.float64)
 negative=(-terms).relu().sum(-1,dtype=torch.float64)
 exact=torch.softmax((q[0].double()@visual_keys.double().transpose(1,2))/math.sqrt(attn.head_dim),-1)
 stop={'what','which','where','how','is','are','the','a','an','of','in','on','to','does','do','this','that','there','it','and','with','can','you','see','shown','image','picture','photo','photograph','at','many','color','kind','type','made','from'}
 content=[]
 for j,position in enumerate(question):
  text=tokenizer.decode([state['input_ids'][position]],skip_special_tokens=True).lower().strip(' ▁Ġ.,?!:;\n')
  if any(c.isalpha() for c in text) and text not in stop:content.append(j)
 if not content:content=list(range(len(question)))
 output['question_exact']=np.array(state['question_exact'])
 output['visual_targets']=np.array(visual_keys.shape[1])
 output['content_query_tokens']=np.array(len(content))
 for temperature in [.25,1.,4.]:
  st=combine_evidence(positive,negative,temperature)
  if temperature==1:
   error=(exact-st['normalized_plausibility']).abs().max().item()
   assert error<1e-5,('Attention plausibility reconstruction failed',error)
   output['plausibility_reconstruction_error']=np.array(error)
  for name,positions,token_pool in [('prediction',[len(indices)-1],'mean'),('question_end',[len(question)-1],'mean'),('question_mean',list(range(len(question))),'mean'),('content_mean',content,'mean'),('content_max',content,'max')]:
   def pool(x):
    x=x[:,positions]
    return x.max(1).values if token_pool=='max' else x.mean(1)
   raw=pool(st['cf']);score=pool(st['cf_logscore']);ig=pool(st['ignorance'])
   vals={'mean_cf':raw.mean(),'mean_logcf':score.mean(),'top4_logcf':score.topk(4).values.mean(),'max_logcf':score.max(),'ig':ig.mean(),'negative_log_ig':pool(-st['log_ignorance']).mean()}
   for key,val in vals.items():
    assert torch.isfinite(val)
    output[f'{name}_T{temperature:g}_{key}']=np.array(val.item())
attn.wo.register_forward_pre_hook(split_hook)
config={'hypothesis_frame':'visual key/patch positions as categorical attention targets','query_features':'actual post-RoPE query coordinates','coefficient':'key coordinates centered across visual keys, divided by sqrt(head_dim)','attention_softmax_reconstruction_checked':True,'layer':0,'sources':['image context tokens','all other text/context tokens'],'queries':['prediction','question_end','question_mean'],'input':'original image/question','normalized_variant':'per-head conditional context divided by attention mass','actual_sum_reconstruction_checked':False,'no_output_head_score':True}
(TASK/'results/extraction_config.json').write_text(json.dumps(config,indent=2))
with torch.inference_mode():
 for split in a.splits.split(','):
  rows=json.loads((SPLITS/f'{split}.json').read_text());rows=rows[:a.limit] if a.limit else rows
  dest=TASK/f'results/features/{split}';dest.mkdir(parents=True,exist_ok=True)
  for n,row in enumerate(rows):
   out=dest/f'{row["row_id"]}.npz'
   if out.exists():continue
   output.clear();current_question[0]=row['question']
   pixels=existing.helpers.load_image(str(resolve_record_path(row['image_path'])),max_num=1).to(dtype=torch.bfloat16,device='cuda')
   response=model.chat(tokenizer,pixels,row['question'],{'max_new_tokens':1,'do_sample':False})
   assert all(np.isfinite(v).all() for v in output.values())
   np.savez_compressed(out,**output)
   if (n+1)%25==0:print(split,n+1,'/',len(rows),flush=True)
   del response
  print('DONE',split,len(rows),flush=True)
