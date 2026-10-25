"""Capture frozen A/B verifier classifier features, without any environment changes."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

ROOT=TASK5 / "results"
IMPROVE=SPLITS
BASE=EXPERIMENT
os.environ['HF_MODULES_CACHE']=str(CACHE/'hf_modules')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(TASK2/'code'));sys.path.insert(0,str(TASK4/'code'))
import numpy as np
import torch
import run_internvl as existing
from run_candidates import PROMPTS
p=argparse.ArgumentParser();p.add_argument('--splits',default='dev,validation');a=p.parse_args()
model=existing.AutoModel.from_pretrained(existing.MODEL_PATH,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True,local_files_only=True).eval().cuda()
tokenizer=existing.AutoTokenizer.from_pretrained(existing.MODEL_PATH,trust_remote_code=True,use_fast=False,local_files_only=True)
ids=[tokenizer.encode(c,add_special_tokens=False) for c in ['A','B']];assert all(len(v)==1 for v in ids)
weight=model.language_model.output.weight[[v[0] for v in ids]].detach().float().cpu().numpy()
wp=ROOT/'ab_weight.npy'
if wp.exists():assert np.array_equal(np.load(wp),weight)
else:np.save(wp,weight)
cap=[]
layercap={}
for index in [0,31]:
 def layer_hook(module,inp,index=index):layercap[index]=inp[0][0,-1].detach().float().cpu().numpy()
 model.language_model.model.layers[index].feed_forward.w2.register_forward_pre_hook(layer_hook)
def hook(module,inp):cap.append(inp[0][0,-1].detach().float().cpu().numpy())
model.language_model.output.register_forward_pre_hook(hook)
with torch.inference_mode():
 for split in a.splits.split(','):
  rows=json.loads((IMPROVE/f'{split}.json').read_text());out=ROOT/f'{split}_head.npz';xs=[];rowids=[];phis0=[];phis31=[]
  if out.exists():
   prev=np.load(out);xs=list(prev['phi']);rowids=list(prev['row_id']);phis0=list(prev['phi0']);phis31=list(prev['phi31'])
  done=set(rowids)
  for row in rows:
   if row['row_id'] in done:continue
   cap.clear();pixels=existing.helpers.load_image(str(resolve_record_path(row['image_path'])),max_num=1).to(dtype=torch.bfloat16,device='cuda')
   model.chat(tokenizer,pixels,row['question'],{'max_new_tokens':1,'do_sample':False})
   phis0.append(layercap[0].copy());phis31.append(layercap[31].copy());cap.clear()
   model.chat(tokenizer,pixels,PROMPTS['verify'](row['question']),{'max_new_tokens':1,'do_sample':False})
   assert len(cap)==1
   xs.append(cap[0]);rowids.append(row['row_id'])
   if len(xs)%50==0:
    np.savez_compressed(out,phi=np.array(xs),phi0=np.array(phis0),phi31=np.array(phis31),row_id=np.array(rowids));print(split,len(xs),flush=True)
  np.savez_compressed(out,phi=np.array(xs),phi0=np.array(phis0),phi31=np.array(phis31),row_id=np.array(rowids));print('DONE',split,len(xs),flush=True)
