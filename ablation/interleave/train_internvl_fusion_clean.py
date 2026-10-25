"""Fresh bounded InternVL+DINO projector with freeform supervision.

Excludes ALL evaluated true object classes from ImageNet training. Fixed 512
images, two epochs, no test-set selection. Half the images receive synthetic
wrong-class text. Official chat template; assistant-only loss; frozen backbones.
The optional MCQ objective uses balanced option order; default is freeform.
"""
import argparse
import json
import os
from pathlib import Path
import random
import re
import runpy
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(HERE/"internvl"))
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from run_e1_e2_internvl import build_transform, MODEL_ID, MODEL_CACHE
from modeling_internvl_chat_dino import InternVLChatModelDINO, get_conv_template
from transformers import AutoTokenizer


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--size",type=int,default=512)
    ap.add_argument("--epochs",type=int,default=2)
    ap.add_argument("--task",choices=['mcq','freeform'],default='freeform')
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/"projector.pth").exists(): raise FileExistsError(out)
    random.seed(20261016); np.random.seed(20261016); torch.manual_seed(20261016)
    torch.set_num_threads(8)
    rng=random.Random(20261016)
    names=runpy.run_path('/data/liuxiaofei/uncertainty/Defense-Prefix-main/datasets/configs/imagenet_config.py')['classes']
    excluded=set()
    for file in [ROOT/'datasets/STA/sta_mini.csv',ROOT/'typo-parameter/clean/data.csv']:
        for question in pd.read_csv(file).question:
            m=re.search(r'\(a\)\s*(.*?)\s*\(b\)',question)
            if not m: raise ValueError(question)
            excluded.add(m.group(1).strip())
    dataset=ImageFolder('/data/public/datasets/ImageNet-val/val')
    eligible=[(p,names[y]) for p,y in dataset.samples if names[y] not in excluded]
    rng.shuffle(eligible)
    assert len(eligible)>=args.size
    entries=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',24)
    for i,(path,label) in enumerate(eligible[:args.size]):
        distractor=rng.choice([n for n in names if n!=label])
        swap=i%2
        attack=(i//2)%2
        a,b=(distractor,label) if swap else (label,distractor)
        question=("this photo is a ' '? Only simply output the corresponding letter of the option "
                  "(such as a or b or c or d) do not output the specific content of the option   "
                  f"(a) {a} (b) {b}")
        entries.append(dict(path=path,label=label,distractor=distractor,swapped=swap,
                            attack=attack,
                            question='What is the main object in this image? Answer with a concise noun phrase.' if args.task=='freeform' else question,
                            answer=label if args.task=='freeform' else 'b' if swap else 'a'))
    (out/'train_manifest.json').write_text(json.dumps(entries,indent=2))
    (out/'protocol.json').write_text(json.dumps(dict(seed=20261016,size=args.size,epochs=args.epochs,
        excluded_classes=sorted(excluded),eligible_images=len(eligible),lr=1e-4,batch_size=2,
        grad_accum=4,max_rms_ratio=.25,task=args.task,objective='assistant-only '+args.task+', 50% synthetic text',
        note='supervised robustness intervention; not training-free ACF'),indent=2))
    model=InternVLChatModelDINO.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16,use_flash_attn=True,
        trust_remote_code=True,cache_dir=MODEL_CACHE,local_files_only=True,ignore_mismatched_sizes=True).to('cuda:0').eval()
    tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True,use_fast=False,cache_dir=MODEL_CACHE,local_files_only=True)
    model.img_context_token_id=tok.convert_tokens_to_ids('<IMG_CONTEXT>')
    model._init_dino('cuda:0'); model.reset_dino_projector()
    model.dino_gate.data.fill_(1.)
    model.dino_max_rms_ratio=.25
    model.set_dino_mode('fused')
    for name,param in model.named_parameters(): param.requires_grad=('dino_projector' in name)
    # Raw DINO patch features are taken before projection, with identical
    # resizing/pooling as the inference helper; collect with a pre-hook.
    cache=[]
    features=[]
    handle=model.dino_projector.register_forward_pre_hook(lambda m,a: features.append(a[0].detach()))
    transform=build_transform(448)
    t0=time.time()
    with torch.inference_mode():
        for i,e in enumerate(entries):
            image=Image.open(e['path']).convert('RGB').resize((448,448))
            if e['attack']:
                draw=ImageDraw.Draw(image)
                text=e['distractor']
                box=draw.textbbox((0,0),text,font=font,stroke_width=2)
                draw.text(((448-box[2])/2,210),text,font=font,fill='white',stroke_width=2,stroke_fill='black')
            pv=transform(image).unsqueeze(0).to('cuda:0',torch.bfloat16)
            native=model.extract_feature(pv)
            features.clear(); model.extract_dino_residual(pv)
            cache.append((native.cpu(),features[0].cpu()))
            if (i+1)%32==0: print(f'[cache] {i+1}/{len(entries)} {time.time()-t0:.0f}s',flush=True)
    handle.remove()
    # Clone outside inference_mode so autograd can save frozen tower features.
    cache=[(a.clone(),b.clone()) for a,b in cache]
    def sequence(e):
        conv=get_conv_template(model.template); conv.system_message=model.system_message
        conv.append_message(conv.roles[0],'<image>\n'+e['question'])
        conv.append_message(conv.roles[1],None)
        prefix=conv.get_prompt().replace('<image>','<img>'+'<IMG_CONTEXT>'*model.num_image_token+'</img>')
        ids=tok(prefix,return_tensors='pt')['input_ids'][0]
        answer=tok(e['answer']+conv.sep,add_special_tokens=False,return_tensors='pt')['input_ids'][0]
        full=torch.cat([ids,answer]); labels=full.clone(); labels[:len(ids)]=-100
        return full,labels
    tokenized=[sequence(e) for e in entries]
    optimizer=torch.optim.AdamW(model.dino_projector.parameters(),lr=1e-4,weight_decay=.01)
    losses=[]
    for epoch in range(args.epochs):
        order=list(range(len(entries))); rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        for step,start in enumerate(range(0,len(order),2)):
            idx=order[start:start+2]
            ids=[tokenized[i][0] for i in idx]; labs=[tokenized[i][1] for i in idx]
            size=max(map(len,ids))
            input_ids=torch.full((len(idx),size),tok.pad_token_id or 0,dtype=torch.long,device='cuda:0')
            labels=torch.full_like(input_ids,-100); mask=torch.zeros_like(input_ids)
            for j,(x,y) in enumerate(zip(ids,labs)):
                input_ids[j,:len(x)]=x.to('cuda:0'); labels[j,:len(y)]=y.to('cuda:0'); mask[j,:len(x)]=1
            native=torch.cat([cache[i][0] for i in idx]).to('cuda:0')
            dino=torch.cat([cache[i][1] for i in idx]).to('cuda:0')
            fused=model.fuse_dino_residual(native,model.dino_projector(dino))
            embed=model.language_model.get_input_embeddings()(input_ids).clone()
            selected=input_ids==model.img_context_token_id
            assert int(selected.sum())==len(idx)*model.num_image_token
            embed[selected]=fused.reshape(-1,embed.shape[-1])
            output=model.language_model(inputs_embeds=embed,attention_mask=mask,use_cache=False,return_dict=True)
            while not hasattr(output,'logits'): output=output[0]
            logits=output.logits
            loss=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1))
            if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
            (loss/4).backward()
            if (step+1)%4==0 or start+2>=len(order):
                grad=float(torch.nn.utils.clip_grad_norm_(model.dino_projector.parameters(),1.))
                assert np.isfinite(grad)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(dict(epoch=epoch+1,step=step+1,loss=float(loss.detach())))
            if (step+1)%16==0: print(f'[train] epoch={epoch+1} step={step+1} loss={loss.item():.4f} elapsed={time.time()-t0:.0f}s',flush=True)
        torch.save(dict(dino_projector={k:v.detach().cpu() for k,v in model.dino_projector.state_dict().items()},
                        dino_gate=1.,fusion_config=dict(max_rms_ratio=.25),epoch=epoch+1,
                        protocol=json.loads((out/'protocol.json').read_text())),out/f'epoch{epoch+1}.pth')
        assert all(p.grad is None for name,p in model.named_parameters() if 'dino_projector' not in name)
    torch.save(torch.load(out/f'epoch{args.epochs}.pth',map_location='cpu'),out/'projector.pth')
    pd.DataFrame(losses).to_csv(out/'loss.csv',index=False)
    print('[done]',out/'projector.pth',flush=True)


if __name__=='__main__': main()
