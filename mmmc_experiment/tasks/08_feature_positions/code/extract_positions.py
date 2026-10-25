"""Internal attention/FFN sites only; raw image-question, frozen model."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import SPLITS,CACHE,TASK2,EXPERIMENT,resolve_record_path
TASK=EXPERIMENT/'tasks/08_feature_positions'
os.environ['HF_MODULES_CACHE']=str(CACHE/'hf_modules');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(TASK2/'code'))
import numpy as np
import torch
import run_internvl as existing
p=argparse.ArgumentParser();p.add_argument('--splits',default='dev,validation');p.add_argument('--limit',type=int,default=0);p.add_argument('--layers',default='0');a=p.parse_args()
layers=list(map(int,a.layers.split(',')));sites=['attn_wo','ffn_w1','ffn_w3']
model=existing.AutoModel.from_pretrained(existing.MODEL_PATH,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True,local_files_only=True).eval().cuda()
tokenizer=existing.AutoTokenizer.from_pretrained(existing.MODEL_PATH,trust_remote_code=True,use_fast=False,local_files_only=True)
features={};weightdir=TASK/'results/weights';weightdir.mkdir(exist_ok=True)
for i in layers:
 block=model.language_model.model.layers[i]
 modules={'attn_wo':block.attention.wo,'ffn_w1':block.feed_forward.w1,'ffn_w3':block.feed_forward.w3}
 for site,module in modules.items():
  key=f'L{i}_{site}';weightfile=weightdir/f'{key}.pt'
  if not weightfile.exists():torch.save(module.weight.detach().cpu(),weightfile)
  def capture(mod,inp,key=key):features[key]=inp[0][0,-1].detach().float().cpu().numpy().astype(np.float16)
  module.register_forward_pre_hook(capture)
config={'sites':sites,'layers':layers,'query':'last prefill position predicting first output token','model':existing.MODEL_PATH,'input':'unchanged original question and image','max_tiles':1,'max_new_tokens':1,'no_output_head_scores':True}
(TASK/'results/extraction_config.json').write_text(json.dumps(config,indent=2))
with torch.inference_mode():
 for split in a.splits.split(','):
  rows=json.loads((SPLITS/f'{split}.json').read_text());rows=rows[:a.limit] if a.limit else rows
  dest=TASK/f'results/features/{split}';dest.mkdir(parents=True,exist_ok=True)
  for n,row in enumerate(rows):
   out=dest/f'{row["row_id"]}.npz'
   if out.exists():continue
   features.clear();pixels=existing.helpers.load_image(str(resolve_record_path(row['image_path'])),max_num=1).to(dtype=torch.bfloat16,device='cuda')
   returned=model.chat(tokenizer,pixels,row['question'],{'max_new_tokens':1,'do_sample':False})
   assert len(features)==len(layers)*len(sites) and all(np.isfinite(v).all() for v in features.values())
   tmp=out.with_suffix('.partial.npz');np.savez_compressed(tmp,**features);tmp.replace(out)
   if (n+1)%25==0:print(split,n+1,'/',len(rows),flush=True)
   del returned
  print('DONE',split,len(rows),flush=True)
