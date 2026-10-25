"""Label-free offline background distribution of intermediate visual tokens."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import CACHE, SPLITS, TASK2, resolve_record_path
os.environ['HF_MODULES_CACHE']=str(CACHE/'hf_modules')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(TASK2/'code'))
import run_internvl as existing
import numpy as np
import torch
TASK=Path(__file__).resolve().parent.parent
LAYERS=[0,8,16,20,24,27,31]

class Captured(Exception):
    pass

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(42)
    model=existing.AutoModel.from_pretrained(existing.MODEL_PATH,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True,local_files_only=True).eval().cuda()
    tok=existing.AutoTokenizer.from_pretrained(existing.MODEL_PATH,trust_remote_code=True,use_fast=False,local_files_only=True)
    lm=model.language_model
    state={}
    old_generate=model.generate
    def generate(*a,**kw):
        state['visual']=kw['input_ids'][0]==model.img_context_token_id
        return old_generate(*a,**kw)
    model.generate=generate
    def install(layer):
        def capture(module,inputs):
            h=inputs[0][0,state['visual']]
            p=lm.output(lm.model.norm(h)).float().softmax(-1)
            state[layer]=p.double().mean(0).cpu().numpy()
            if layer==max(LAYERS):
                raise Captured()
        lm.model.layers[layer].ffn_norm.register_forward_pre_hook(capture)
    for layer in LAYERS:
        install(layer)
    rows=json.loads((SPLITS/'dev.json').read_text())
    unique={}
    for row in rows:
        unique.setdefault(row['image_id'],row)
    dest=TASK/'results/reference_images'
    dest.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for n,row in enumerate(unique.values()):
            path=dest/f'{row["image_id"]}.npz'
            if path.exists():
                continue
            state.clear()
            pixels=existing.helpers.load_image(str(resolve_record_path(row['image_path'])),max_num=1).to(dtype=torch.bfloat16,device='cuda')
            try:
                model.chat(tok,pixels,row['question'],{'max_new_tokens':1,'do_sample':False})
                raise RuntimeError('Early exit absent')
            except Captured:
                pass
            tmp=path.with_suffix('.partial.npz')
            np.savez_compressed(tmp,**{f'L{l}':state[l] for l in LAYERS})
            tmp.replace(path)
            if (n+1)%10==0:
                print('reference',n+1,'/',len(unique),flush=True)
    means={}
    for layer in LAYERS:
        values=[]
        for imageid in unique:
            with np.load(dest/f'{imageid}.npz') as z:
                values.append(z[f'L{layer}'])
        mean=np.mean(values,axis=0)
        assert abs(mean.sum()-1)<1e-5
        means[f'L{layer}']=mean
    np.savez_compressed(TASK/'results/background.npz',**means)
    (TASK/'results/config.json').write_text(json.dumps({'layers':LAYERS,'reference_images':list(unique),'n_images':len(unique),'estimator':'mean over all visual patches and unique dev images','labels_used':False,'answers_used':False,'query_independence':'causal visual prefix precedes user question; task14 position fixed post-attention residual','model':existing.MODEL_PATH,'inference':'one prefill, no generated answers'},indent=2))
    print('DONE background',len(unique),flush=True)

if __name__=='__main__':
    main()
