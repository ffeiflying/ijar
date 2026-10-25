"""Frozen-model detectors. No package/model changes; uses only image+question."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

import sys
import time
ROOT=TASK4 / "results"
BASE=EXPERIMENT
os.environ['HF_MODULES_CACHE']=str(CACHE/'hf_modules')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(TASK2/'code'))
import torch
from PIL import Image
import run_internvl as existing

PROMPTS={
'plain':lambda q:q,
'verify':lambda q:f'Examine the image carefully. Does the following question contain a false premise that contradicts the image, such as a nonexistent object, an incorrect attribute, or an incorrect relationship?\nQuestion: {q}\nA: Yes, the question contains a false visual premise.\nB: No, the question is consistent with the image.\nRespond with only A or B.',
'grounded':lambda q:f'Check whether the image supports the assumptions made by this question. Do not answer the question itself.\nQuestion: {q}\nA: The question cannot be answered as stated because an assumption is contradicted by the image.\nB: The question can be answered as stated using the image.\nAnswer with only A or B.',
'reverse':lambda q:f'Inspect the image and check the visual premises in the question.\nQuestion: {q}\nA: All premises of the question are consistent with the image.\nB: The question makes a false assumption about objects, attributes, or relationships visible in the image.\nAnswer with only A or B.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',default='dev',choices=['dev','validation','final']);p.add_argument('--modes',default='plain,verify,grounded,reverse,verify4,blind');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 rows=json.loads((SPLITS/f'{a.split}.json').read_text());rows=rows[:a.limit] if a.limit else rows
 modes=a.modes.split(',');out=ROOT/f'{a.split}_candidates.jsonl'
 done=set()
 if out.exists():
  for line in out.read_text().splitlines():
   r=json.loads(line);done.add((r['row_id'],r['mode']))
 if all((r['row_id'],m) in done for r in rows for m in modes):
  print('All requested records already completed:',out);return
 raise SystemExit('Historical v1 comparison is frozen. Use tasks/07_attention_acf/code with evidence_v2 for new ACF experiments.')
 torch.manual_seed(42)
 model=existing.AutoModel.from_pretrained(existing.MODEL_PATH,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True,local_files_only=True).eval().cuda()
 tokenizer=existing.AutoTokenizer.from_pretrained(existing.MODEL_PATH,trust_remote_code=True,use_fast=False,local_files_only=True)
 ids={c:tokenizer.encode(c,add_special_tokens=False) for c in ['A','B']}
 assert all(len(v)==1 for v in ids.values()),ids
 config={'model':existing.MODEL_PATH,'split':a.split,'split_sha256':hashlib.sha256((SPLITS/f'{a.split}.json').read_bytes()).hexdigest(),'prompts':{k:v('{QUESTION}') for k,v in PROMPTS.items()},'A_B_ids':ids,'max_new_tokens':1,'do_sample':False,'features':'plain all-layer original/elementwise evidence; prompted A-B decision margin; verify4 max4tiles; blind no image','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 cfg=ROOT/f'{a.split}_candidates.config.json'
 if cfg.exists():assert json.loads(cfg.read_text())==config,'Configuration differs'
 else:cfg.write_text(json.dumps(config,indent=2))
 logits=[];acts={};collect=[False]
 def output_hook(module,inp,out):logits.append(out[0,-1].detach().float())
 model.language_model.output.register_forward_hook(output_hook)
 def capture(i):
  def hook(module,inp):
   if collect[0]:acts[i]=inp[0].detach().reshape(-1,inp[0].shape[-1])[-1].clone()
  return hook
 for i,layer in enumerate(model.language_model.model.layers):layer.feed_forward.w2.register_forward_pre_hook(capture(i))
 def features():
  f={}
  for i,phi in acts.items():
   weight=model.language_model.model.layers[i].feed_forward.w2.weight
   # Match original BF16 projection then reduce repeated identical columns algebraically.
   centered=weight-weight.mean(1,keepdim=True)
   z=(centered@phi[:,None])/weight.shape[0]
   pos=z.double().relu().sum();neg=(-z.double()).relu().sum();K=weight.shape[1]
   log_ep=-torch.log1p(K*torch.expm1(pos))
   log_en=-torch.log(-torch.expm1(K*torch.log1p(-torch.exp(-neg))))
   cf=K*torch.exp(log_ep)*torch.expm1(pos)*(1-torch.exp(log_en-neg))
   f[f'cf_L{i}']=cf.item()
   # Alternative: preserve per-feature signed contributions instead of copying projection.
   contributions=centered.float()*phi.float()[None,:]/weight.shape[0]
   positive=contributions.relu().sum(1).double();negative=(-contributions).relu().sum(1).double()
   log_positive=torch.where(positive>0,positive+torch.log(-torch.expm1(-positive)),torch.full_like(positive,-torch.inf))
   lep=-torch.logsumexp(torch.cat([log_positive,positive.new_zeros(1)]),0)
   leneg=-torch.log(-torch.expm1(torch.log1p(-torch.exp(-negative)).sum()))
   kappa=(torch.exp(lep+log_positive)*(-torch.expm1(leneg-negative))).sum()
   f[f'elementwise_cf_L{i}']=kappa.item()
   f[f'activation_norm_L{i}']=phi.float().norm().item()
  assert all(__import__('math').isfinite(v) for v in f.values()),'Nonfinite features'
  return f
 with out.open('a',buffering=1) as handle,torch.inference_mode():
  for row in rows:
   for mode in modes:
    if (row['row_id'],mode) in done:continue
    start=time.time();logits.clear();acts.clear();collect[0]=mode=='plain'
    prompt=PROMPTS['verify' if mode in ['verify4','blind'] else mode](row['question'])
    pixels=None if mode=='blind' else existing.helpers.load_image(str(resolve_record_path(row['image_path'])),max_num=4 if mode=='verify4' else 1).to(dtype=torch.bfloat16,device='cuda')
    # Local chat assumes image unless history is provided; an empty history supports text-only controls.
    returned=model.chat(tokenizer,pixels,prompt,{'max_new_tokens':1,'do_sample':False},history=[] if mode=='blind' else None)
    assert len(logits)==1,len(logits)
    l=logits[0];prob=l.softmax(-1);margin=(l[ids['A'][0]]-l[ids['B'][0]]).item()
    if mode=='reverse':margin=-margin
    feat=features() if mode=='plain' else {}
    result={'row_id':row['row_id'],'image_id':row['image_id'],'label':row['label'],'conflict_type':row['conflict_type'],'mode':mode,'decision_margin':margin,'entropy':float(-(prob*prob.clamp_min(1e-30).log()).sum()),'negative_max_prob':-prob.max().item(),'features':feat,'response':returned[0] if isinstance(returned,tuple) else returned,'seconds':time.time()-start}
    handle.write(json.dumps(result)+'\n')
    print(json.dumps({k:result[k] for k in ['row_id','mode','decision_margin','seconds']}),flush=True)
    del returned
 print('DONE',a.split,flush=True)
if __name__=='__main__':main()
