"""Paired CF localization experiment; no training or environment edits.

InternVL native, fixed layers, known synthetic text masks. Distinguishes
image-token CF from GradCAM of decision-position CF. The former is causal-
decoder prefix-only and does not see the subsequent question.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE.parent/'interleave/internvl'))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from modeling_internvl_chat_dino import InternVLChatModelDINO, get_conv_template
from transformers import AutoTokenizer
from gradcam import gradcam_saliency

MODEL='OpenGVLab/InternVL2_5-8B'
CACHE='/data/public/models/InternVL/InternVL2_5-8B'
QUESTION='What is the main object in this image? Answer with a concise noun phrase.'
CONDITIONS=['clean','correct_text','wrong_text','irrelevant_text','occlusion']


def cf(states, centered):
    """Closed form of EvidenceModel, with the same row centering/axes."""
    j,k=centered.shape
    v=((centered @ states.float().T)/j).double()
    p=v.clamp_min(0).sum(0)
    n=(-v).clamp_min(0).sum(0)
    eta_p=1/(k*torch.expm1(p)+1)
    eta_n=-1/torch.expm1(k*torch.log1p(-torch.exp(-n)))
    return k*eta_p*torch.expm1(p)*(1-eta_n*torch.exp(-n))


def build_prompt(model,tok):
    conv=get_conv_template(model.template)
    conv.system_message=model.system_message
    conv.append_message(conv.roles[0],'<image>\n'+QUESTION)
    conv.append_message(conv.roles[1],None)
    prompt=conv.get_prompt().replace('<image>','<img>'+'<IMG_CONTEXT>'*model.num_image_token+'</img>')
    return tok(prompt,return_tensors='pt').to('cuda:0')


def prepare(out,n,seed):
    rng=random.Random(seed)
    labels=runpy.run_path('/data/liuxiaofei/uncertainty/Defense-Prefix-main/datasets/configs/imagenet_config.py')['classes']
    data=ImageFolder('/data/public/datasets/ImageNet-val/val')
    indices=rng.sample(range(len(data.samples)),n)
    entries=[]
    for pair_id,index in enumerate(indices):
        source,label_id=data.samples[index]
        correct=str(labels[label_id]).split(',')[0].strip()
        candidates=[str(x).split(',')[0].strip() for x in labels
                    if x!=labels[label_id] and abs(len(str(x).split(',')[0].strip())-len(correct))<=3]
        wrong=rng.choice(candidates)
        texts=dict(correct_text=correct,wrong_text=wrong,irrelevant_text='please look')
        fontsize=30
        while True:
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',fontsize)
            draw=ImageDraw.Draw(Image.new('RGB',(448,448)))
            sizes=[draw.textbbox((0,0),s,font=font,stroke_width=1,anchor='lt') for s in texts.values()]
            width=max(b[2] for b in sizes)+8
            height=max(b[3] for b in sizes)+8
            if width<=390 or fontsize<=12:break
            fontsize-=1
        positions=[((448-width)//2,(448-height)//2),(24,24),(448-width-24,448-height-24),(24,448-height-24)]
        x,y=positions[pair_id%4]
        bbox=[x,y,x+width,y+height]
        clean=Image.open(source).convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
        boxmask=Image.new('L',(448,448),0)
        ImageDraw.Draw(boxmask).rectangle((x,y,x+width-1,y+height-1),fill=255)
        glyph=Image.new('L',(448,448),0)
        ImageDraw.Draw(glyph).text((x+4,y+4),wrong,font=font,fill=255,stroke_width=1,stroke_fill=255,anchor='lt')
        pairdir=out/'images'/f'{pair_id:03d}'
        pairdir.mkdir(parents=True)
        boxmask.save(pairdir/'box_mask.png');glyph.save(pairdir/'wrong_glyph_mask.png')
        for condition in CONDITIONS:
            image=clean.copy()
            if condition in texts:
                overlay=image.copy()
                ImageDraw.Draw(overlay).text((x+4,y+4),texts[condition],font=font,fill='white',stroke_width=1,stroke_fill='black',anchor='lt')
                image=Image.blend(image,overlay,.9)
            elif condition=='occlusion':
                ImageDraw.Draw(image).rectangle((x,y,x+width-1,y+height-1),fill=(128,128,128))
            path=pairdir/f'{condition}.png';image.save(path)
            entries.append(dict(pair_id=pair_id,source=source,source_index=index,condition=condition,
                correct_class=correct,wrong_class=wrong,text=texts.get(condition,''),bbox=bbox,
                font_size=fontsize,image=str(path),box_mask=str(pairdir/'box_mask.png'),
                glyph_mask=str(pairdir/'wrong_glyph_mask.png')))
    (out/'manifest.json').write_text(json.dumps(entries,indent=2))
    return entries


def region_stats(value,mask):
    value=np.asarray(value,dtype=float)
    a=float(mask.sum());b=float((1-mask).sum())
    inside=float((value*mask).sum()/a)
    outside=float((value*(1-mask)).sum()/b)
    mass=float((value*mask).sum()/value.sum()) if value.sum()>0 else float('nan')
    peak=np.unravel_index(int(value.argmax()),value.shape)
    return dict(inside=inside,outside=outside,ratio=inside/outside if outside>0 else float('nan'),
                enrichment=mass/float(mask.mean()) if np.isfinite(mass) else float('nan'),
                peak_mask_fraction=float(mask[peak]),zero_map=int(np.max(np.abs(value))==0))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--n',type=int,default=16)
    ap.add_argument('--layers',default='0,22')
    ap.add_argument('--seed',type=int,default=20261017)
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'npz').mkdir()
    torch.manual_seed(args.seed);np.random.seed(args.seed);torch.set_num_threads(8)
    entries=prepare(out,args.n,args.seed)
    layers=[int(x) for x in args.layers.split(',')]
    metadata=dict(args=vars(args),model=MODEL,question=QUESTION,layers=layers,
        gradient_target='post-native-projector visual embeddings, 16x16',
        decision_target='prefill last question/assistant-prefix position CF',
        local_target='CF of each image token in prefill',
        caveats=['GradCAM is sensitivity attribution, not pixel CF or attention probability',
                 'local image tokens precede the question in the causal decoder',
                 'synthetic rectangle/glyph masks are known; nontext is not an annotated object mask',
                 'fixed layers, no selection by localization results'],torch=torch.__version__)
    tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,use_fast=False,cache_dir=CACHE,local_files_only=True)
    model=InternVLChatModelDINO.from_pretrained(MODEL,torch_dtype=torch.bfloat16,
        use_flash_attn=True,trust_remote_code=True,cache_dir=CACHE,local_files_only=True,
        ignore_mismatched_sizes=True).to('cuda:0').eval()
    model.requires_grad_(False);model.set_dino_mode('native_only')
    model.img_context_token_id=tok.convert_tokens_to_ids('<IMG_CONTEXT>')
    inputs=build_prompt(model,tok)
    selected=inputs.input_ids==model.img_context_token_id
    assert int(selected.sum())==256
    weights={}
    for layer in layers:
        b=model.language_model.model.layers[layer].feed_forward.w2.weight.detach().float()
        weights[layer]=b-b.mean(1,keepdim=True)
    tf=T.Compose([T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    all_rows=[]
    t0=time.time();max_error=0.
    for step,item in enumerate(entries):
        image=Image.open(item['image']).convert('RGB')
        pv=tf(image).unsqueeze(0).to('cuda:0',torch.bfloat16)
        with torch.no_grad():native=model.extract_feature(pv)
        features=native.detach().requires_grad_(True)
        # Freeze towers and backprop only through the decoder to the visual
        # feature map. This is exactly the selected GradCAM target.
        captured={};handles=[]
        for layer in layers:
            def hook(module,arguments,layer=layer):captured[layer]=arguments[0]
            handles.append(model.language_model.model.layers[layer].feed_forward.w2.register_forward_pre_hook(hook))
        embeds=model.language_model.get_input_embeddings()(inputs.input_ids).detach().clone()
        embeds[selected]=features.reshape(-1,features.shape[-1])
        output=model.language_model(inputs_embeds=embeds,attention_mask=inputs.attention_mask,
                                    use_cache=False,return_dict=True)
        for h in handles:h.remove()
        masks={}
        for name,path in [('box',item['box_mask']),('glyph',item['glyph_mask'])]:
            full=torch.tensor(np.array(Image.open(path))/255.,dtype=torch.float32)[None,None]
            masks[name]=F.avg_pool2d(full,28,28)[0,0].numpy()
        maps={}
        scalars={}
        for layer in layers:
            states=captured[layer][0]
            local=cf(states[selected[0]],weights[layer])
            decision=cf(states[-1:],weights[layer])[0]
            if step==0:
                from envidence import EvidenceModel
                b=model.language_model.model.layers[layer].feed_forward.w2.weight.detach().float()
                reference=EvidenceModel(b)
                reference.get_evidence_weights(states[-1:].detach().float().T)
                error=abs(float(decision)-float(reference.get_evidence_conflict()))
                max_error=max(max_error,error)
                assert error<2e-6,(layer,error)
                # Finite differences through the CF function, independently of
                # the multimodal backward path, using one real hidden state.
                x=states[-1:].detach().float().requires_grad_()
                d=torch.randn_like(x);d=d/d.norm()
                grad=torch.autograd.grad(cf(x,weights[layer]).sum(),x)[0]
                eps=.05
                finite=(cf(x.detach()+eps*d,weights[layer])-cf(x.detach()-eps*d,weights[layer]))/(2*eps)
                analytic=(grad*d).sum()
                assert torch.isfinite(grad).all()
                assert torch.allclose(finite[0],analytic.double(),rtol=.08,atol=2e-5),(finite,analytic)
                print('[check] layer',layer,'reference_error',error,'directional_grad',float(analytic),'finite',float(finite[0]),flush=True)
            grad_dec=torch.autograd.grad(decision,features,retain_graph=True)[0]
            grad_img=torch.autograd.grad(local.mean(),features,retain_graph=True)[0]
            assert torch.isfinite(grad_dec).all() and torch.isfinite(grad_img).all()
            cam_dec=gradcam_saliency(features.detach(),grad_dec).detach().cpu().numpy().reshape(16,16)
            cam_img=gradcam_saliency(features.detach(),grad_img).detach().cpu().numpy().reshape(16,16)
            signed=(features.detach().float()*grad_dec.float()).sum(-1).detach().cpu().numpy().reshape(16,16)
            kmap=local.detach().cpu().numpy().reshape(16,16)
            values=dict(token_cf=kmap,decision_cf_cam=cam_dec,image_cf_cam=cam_img,
                        decision_grad_times_activation_abs=np.abs(signed))
            maps.update({f'L{layer}_{name}':value for name,value in values.items()})
            maps[f'L{layer}_decision_grad_times_activation_signed']=signed
            scalars[layer]=dict(decision_cf=float(decision.detach()),image_cf=float(local.mean().detach()))
            for method,value in values.items():
                assert np.isfinite(value).all(),(item,layer,method)
                for mask_name,mask in masks.items():
                    all_rows.append(dict(pair_id=item['pair_id'],condition=item['condition'],layer=layer,
                        method=method,mask=mask_name,decision_cf=scalars[layer]['decision_cf'],
                        image_cf=scalars[layer]['image_cf'],**region_stats(value,mask)))
        del output,embeds,captured,states,local,decision,features,grad_dec,grad_img
        with torch.no_grad():
            response,*_=model.chat(tok,pv,QUESTION,dict(max_new_tokens=32,do_sample=False))
        np.savez_compressed(out/'npz'/f"{item['pair_id']:03d}_{item['condition']}.npz",
            **maps,box_mask=masks['box'],glyph_mask=masks['glyph'],response=response,
            image=item['image'],correct_class=item['correct_class'],wrong_class=item['wrong_class'],
            scalar_json=json.dumps(scalars))
        item['response']=response;item['scalars']=scalars
        with (out/'region_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
        print(f"[sample] {step+1}/{len(entries)} pair={item['pair_id']} {item['condition']} response={response!r} elapsed={time.time()-t0:.0f}s",flush=True)
        del maps,values,signed,kmap,cam_dec,cam_img
    metadata['acf_reference_max_abs_error']=max_error
    metadata['elapsed_seconds']=time.time()-t0
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'manifest.json').write_text(json.dumps(entries,indent=2))
    print('[done]',out,flush=True)


if __name__=='__main__':main()
