"""Summarize paired localization without selecting layers or examples."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CONDITIONS=['clean','correct_text','wrong_text','irrelevant_text','occlusion']


def estimate(values,rng):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if len(a)==0:return dict(n_valid=0,mean=None,median=None,ci_low=None,ci_high=None)
    boot=a[rng.integers(len(a),size=(4000,len(a)))].mean(1)
    return dict(n_valid=len(a),mean=float(a.mean()),median=float(np.median(a)),
                ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--fig-pairs',type=int,default=4)
    args=ap.parse_args();out=Path(args.out)
    data=pd.read_csv(out/'region_metrics.csv')
    metadata=json.loads((out/'metadata.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    rng=np.random.default_rng(20261017)
    summary=[]
    for (layer,method,mask,condition),g in data.groupby(['layer','method','mask','condition']):
        summary.append(dict(layer=int(layer),method=method,mask=mask,condition=condition,n=len(g),
            n_zero=int(g.zero_map.sum()),enrichment=estimate(g.enrichment,rng),
            inside_minus_outside=estimate(g.inside-g.outside,rng),ratio=estimate(g.ratio,rng)))
    contrasts=[]
    for (layer,method,mask),g in data.groupby(['layer','method','mask']):
        pivot=g.pivot(index='pair_id',columns='condition',values='enrichment')
        for other in ['clean','correct_text','irrelevant_text','occlusion']:
            contrasts.append(dict(layer=int(layer),method=method,mask=mask,contrast='wrong_text - '+other,
                enrichment_difference=estimate(pivot.wrong_text-pivot[other],rng)))
    scalar=data.drop_duplicates(['pair_id','condition','layer'])
    scalar_contrasts=[]
    for layer,g in scalar.groupby('layer'):
        for score in ['decision_cf','image_cf']:
            p=g.pivot(index='pair_id',columns='condition',values=score)
            for other in ['clean','correct_text','irrelevant_text','occlusion']:
                scalar_contrasts.append(dict(layer=int(layer),score=score,contrast='wrong_text - '+other,
                    difference=estimate(p.wrong_text-p[other],rng)))
    result=dict(n_pairs=int(data.pair_id.nunique()),fixed_layers=metadata['layers'],
        summary=summary,paired_contrasts=contrasts,scalar_contrasts=scalar_contrasts,
        notes=['Exploratory paired bootstrap across images, not across patches.',
               'Uniform nonnegative map has enrichment 1; zero CAM maps remain undefined.',
               'GradCAM is sensitivity attribution; token CF is a distinct local measurement.',
               'No semantic correctness labels are inferred from freeform string matching.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'figs').mkdir(exist_ok=True)
    keys=['token_cf','decision_cf_cam','image_cf_cam','decision_grad_times_activation_abs']
    titles=['Image-token CF value','Decision-CF GradCAM (+)',
            'Mean image-CF GradCAM (+)','Decision |gradient x activation|']
    file_links=[]
    for pair in range(min(args.fig_pairs,result['n_pairs'])):
        entries={r['condition']:r for r in manifest if r['pair_id']==pair}
        archives={c:np.load(out/'npz'/f'{pair:03d}_{c}.npz') for c in CONDITIONS}
        for layer in metadata['layers']:
            fig,axes=plt.subplots(5,5,figsize=(15,14))
            # Each metric shares a scale across the five conditions of a pair.
            limits={}
            for key in keys:
                arrays=[archives[c][f'L{layer}_{key}'] for c in CONDITIONS]
                lo=min(float(x.min()) for x in arrays) if key=='token_cf' else 0.
                hi=max(float(x.max()) for x in arrays)
                limits[key]=(lo,max(hi,lo+1e-12))
            for i,c in enumerate(CONDITIONS):
                entry=entries[c]
                image=Image.open(entry['image'])
                response=str(archives[c]['response'])
                axes[i,0].imshow(image)
                axes[i,0].set_title(c+'\n'+response[:54],fontsize=9)
                for j,(key,title) in enumerate(zip(keys,titles),start=1):
                    ax=axes[i,j];ax.imshow(image)
                    array=archives[c][f'L{layer}_{key}']
                    vmin,vmax=limits[key]
                    if np.max(np.abs(array))>0:
                        obj=ax.imshow(array,extent=(0,448,448,0),cmap='magma',alpha=.58,
                            interpolation='nearest',vmin=vmin,vmax=vmax)
                        fig.colorbar(obj,ax=ax,fraction=.046,pad=.025)
                    else:ax.text(224,224,'ZERO POSITIVE CAM',ha='center',color='white',bbox=dict(facecolor='black',alpha=.7))
                    record=data[(data.pair_id==pair)&(data.condition==c)&(data.layer==layer)&(data.method==key)&(data['mask']=='box')].iloc[0]
                    enrichment=f'{record.enrichment:.2f}' if np.isfinite(record.enrichment) else 'undefined'
                    ax.set_title(title+'\ntext enrichment='+enrichment,fontsize=8)
                    x0,y0,x1,y1=entry['bbox']
                    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='cyan',lw=1))
                for ax in axes[i]:ax.axis('off')
            fig.suptitle(f"InternVL native L{layer}; pair {pair}; object: {entries['clean']['correct_class']}; wrong text: {entries['clean']['wrong_class']}\nFixed synthetic text rectangle, shared per-column scale; no correctness assigned",fontsize=12)
            fig.tight_layout(rect=(0,0,1,.96))
            path=out/'figs'/f'pair_{pair:03d}_L{layer}.png'
            fig.savefig(path,dpi=140);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)
            file_links.append(path.name)
    # Compact aggregate plot: text-box localization, including undefined counts.
    fig,axes=plt.subplots(len(metadata['layers']),2,figsize=(13,4*len(metadata['layers'])),squeeze=False)
    for row,layer in enumerate(metadata['layers']):
        for col,method in enumerate(['token_cf','decision_cf_cam']):
            ax=axes[row,col]
            for i,c in enumerate(CONDITIONS):
                s=next(s for s in summary if s['layer']==layer and s['method']==method and s['mask']=='box' and s['condition']==c)
                e=s['enrichment']
                if e['mean'] is not None:
                    ax.errorbar(i,e['mean'],yerr=[[max(0,e['mean']-e['ci_low'])],[max(0,e['ci_high']-e['mean'])]],fmt='o',capsize=4)
                ax.text(i,.03,f"valid {e['n_valid']}/{s['n']}",transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
            ax.axhline(1,color='gray',linestyle='--');ax.set_xticks(range(5),CONDITIONS,rotation=20,ha='right')
            ax.set_ylabel('Text-box mass / area fraction');ax.set_title(f'L{layer} {method}; paired bootstrap 95% CI')
    fig.tight_layout();fig.savefig(out/'figs/aggregate.png',dpi=160);fig.savefig(out/'figs/aggregate.pdf');plt.close(fig)
    page=['<!doctype html><meta charset="utf-8"><title>CF text localization</title><style>body{font:16px sans-serif;margin:24px}img{max-width:100%}</style>',
          '<h1>CF text localization — paired pilot</h1><p>16 images x 5 conditions. CF-value and CF attribution are different quantities. Zero positive CAM maps are preserved.</p>',
          '<p><a href="summary.json">Statistics</a> | <a href="region_metrics.csv">Per-image metrics</a> | <a href="manifest.json">Inputs and full responses</a></p>',
          '<img src="figs/aggregate.png">']
    page+=['<h2>'+html.escape(x)+'</h2><img loading="lazy" src="figs/'+html.escape(x)+'">' for x in file_links]
    (out/'index.html').write_text('\n'.join(page))
    for s in summary:
        if s['mask']=='box' and s['condition']=='wrong_text':
            print(s['layer'],s['method'],'enrichment',s['enrichment'],'zero',s['n_zero'])
    print('[saved]',out/'index.html')


if __name__=='__main__':main()
