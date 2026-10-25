"""Core ACF only: raw image/question, post-attention FFN projection inputs.
No LM/A-B score, no verification prompts, no updated model parameters.
"""
import argparse,json,os,sys,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import DATA,SPLITS,CACHE,TASK2,TASK5,EXPERIMENT,resolve_record_path
TASK=EXPERIMENT/'tasks/07_attention_acf'
os.environ['HF_MODULES_CACHE']=str(CACHE/'hf_modules');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(TASK2/'code'));sys.path.insert(0,str(TASK5/'code'))
import numpy as np
import torch
import run_internvl as existing
from evidence_v2 import EvidenceModel
p=argparse.ArgumentParser();p.add_argument('--splits',default='dev,validation');p.add_argument('--layers',default='all');a=p.parse_args()
torch.manual_seed(42)
model=existing.AutoModel.from_pretrained(existing.MODEL_PATH,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True,local_files_only=True).eval().cuda()
tokenizer=existing.AutoTokenizer.from_pretrained(existing.MODEL_PATH,trust_remote_code=True,use_fast=False,local_files_only=True)
selected_layers=set(range(len(model.language_model.model.layers))) if a.layers=='all' else set(map(int,a.layers.split(',')))
features={}
for index,layer in enumerate(model.language_model.model.layers):
 if index not in selected_layers:continue
 def capture(module,inp,index=index):features[index]=inp[0][0,-1].detach().float().cpu().numpy().astype(np.float16)
 layer.feed_forward.w2.register_forward_pre_hook(capture)
config={'evidence_version':EvidenceModel.VERSION,'site':'post-attention FFN feed_forward.w2 input; manuscript feature location','layers':len(model.language_model.model.layers),'input':'original question and image unchanged','token':'prefill last position predicting first response token','model':existing.MODEL_PATH,'max_tiles':1,'max_new_tokens':1,'do_sample':False,'feature_dtype':'float16 storage of BF16 activations (exact for representable finite range)','purpose':'core ACF calibration and layer/temperature comparisons; no output-head scores'}
if a.layers!='all':config['selected_layers']=sorted(selected_layers)
cp=TASK/('results/extraction_config.json' if a.layers=='all' else f'results/extraction_subset_{a.layers}.config.json')
if cp.exists():assert json.loads(cp.read_text())==config
else:cp.write_text(json.dumps(config,indent=2))
with torch.inference_mode():
 for split in a.splits.split(','):
  rows=json.loads((SPLITS/f'{split}.json').read_text());dest=TASK/'results/features'/split;dest.mkdir(parents=True,exist_ok=True)
  for n,row in enumerate(rows):
   out=dest/f'{row["row_id"]}.npz'
   if out.exists():continue
   features.clear()
   image=existing.helpers.load_image(str(resolve_record_path(row['image_path'])),max_num=1).to(dtype=torch.bfloat16,device='cuda')
   returned=model.chat(tokenizer,image,row['question'],{'max_new_tokens':1,'do_sample':False})
   assert len(features)==len(selected_layers) and all(np.isfinite(x).all() for x in features.values())
   tmp=out.with_suffix('.partial.npz');np.savez_compressed(tmp,**{f'L{i}':v for i,v in features.items()});tmp.replace(out)
   if (n+1)%25==0:print(split,n+1,'/',len(rows),flush=True)
   del returned
  print('DONE',split,len(rows),flush=True)
