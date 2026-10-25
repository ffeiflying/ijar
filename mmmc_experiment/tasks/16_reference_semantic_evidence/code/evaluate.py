"""Dev-frozen background contrast, simple-support CF and ignorance."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp
from sklearn.metrics import roc_auc_score
TASK=Path(__file__).resolve().parent.parent
SOURCE=TASK.parent/'14_multilayer_semantic_evidence'
os.environ['HF_MODULES_CACHE']=str(TASK.parents[1]/'shared/cache/hf_modules')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoTokenizer
sys.path.insert(0,str(SOURCE/'code'))
_spec=importlib.util.spec_from_file_location('source_layer_evaluation',SOURCE/'code/evaluate.py')
_source=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_source)
load,metrics=_source.load,_source.metrics
from extract import lexical
MODEL='/data/public/models/InternVL/InternVL2_5-8B/models--OpenGVLab--InternVL2_5-8B/snapshots/e9e4c0dc1db56bfab10458671519b7fa3dd29463'
LAYERS=[0,8,16,20,24,27,31]

def make_scores(rows,features,tok,background):
    query_ids=[]
    for row in rows:
        ids=tok.encode(row['question'],add_special_tokens=False)
        valid=[i for i in ids if lexical(tok,i)]
        query_ids.append(valid if valid else ids)
    scores={}
    masses={}
    for layer in LAYERS:
        logprior=np.log(np.maximum(background[f'L{layer}'],1e-300))
        for alpha in [0.,.25,.5,.75,1.]:
            lognull=alpha*logprior-logsumexp(alpha*logprior)
            offset=np.array([lognull[ids].mean() for ids in query_ids])
            for readout in ['attention','top3','allpatch']:
                contrast=features[f'L{layer}_attn_nll_{readout}_mean']+offset
                # ell=-contrast. Simple support on opposite premise only when ell<0.
                cf_log=np.maximum(contrast,0)
                cf=-np.expm1(-cf_log)
                ig=np.exp(-np.abs(contrast))
                p=-np.expm1(np.minimum(contrast,0))
                assert np.allclose(cf+ig+p,1,atol=1e-10)
                base=f'L{layer}/{readout}/A{alpha:g}'
                scores[base+'/contrast']=contrast
                scores[base+'/cf']=cf_log
                scores[base+'/betp']=cf+.5*ig
                masses[base]={'raw_cf':cf,'ignorance':ig,'support_premise':p,'log_bayes_factor':-contrast}
    return scores,masses

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['freeze','validation'],required=True)
    args=parser.parse_args()
    dest=TASK/'results'
    bgpath=dest/'background.npz'
    tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,use_fast=False,local_files_only=True)
    with np.load(bgpath) as z:
        bg={k:z[k] for k in z.files}
    frozenpath=dest/'frozen_dev_selection.json'
    if args.mode=='freeze':
        rows,features=load('dev')
        scores,masses=make_scores(rows,features,tok,bg)
        y=np.array([r['label'] for r in rows])
        auc={k:float(roc_auc_score(y,v)) for k,v in scores.items()}
        selected={}
        for layer in LAYERS:
            for statistic in ['cf','contrast','betp']:
                candidates=[k for k in auc if k.startswith(f'L{layer}/') and k.endswith('/'+statistic)]
                key=max(candidates,key=lambda k:(auc[k],k))
                selected[f'L{layer}/{statistic}']={'key':key,'dev_auroc':auc[key]}
        best_cf=max((k for k in auc if k.endswith('/cf')),key=lambda k:(auc[k],k))
        best_contrast=max((k for k in auc if k.endswith('/contrast')),key=lambda k:(auc[k],k))
        frozen={'best_cf':best_cf,'best_contrast':best_contrast,'selections':selected,'n_dev_configurations':len(auc),'background_sha256':hashlib.sha256(bgpath.read_bytes()).hexdigest(),'reference_images':100,'learned_class_labels':False,'hyperparameter_selection_labels':'dev only','limitations':'iterative validation, lexical likelihood approximation, not calibrated object absence'}
        assert not frozenpath.exists(),'Do not overwrite frozen selection'
        frozenpath.write_text(json.dumps(frozen,indent=2))
        (dest/'dev_all_metrics.json').write_text(json.dumps(auc,indent=2))
        print(json.dumps({'best_cf':best_cf,'dev_cf_auroc':auc[best_cf],'best_contrast':best_contrast,'dev_contrast_auroc':auc[best_contrast]},indent=2))
        return
    frozen=json.loads(frozenpath.read_text())
    assert frozen['background_sha256']==hashlib.sha256(bgpath.read_bytes()).hexdigest()
    rows,features=load('validation')
    scores,masses=make_scores(rows,features,tok,bg)
    results={}
    for name,selection in frozen['selections'].items():
        key=selection['key']
        result={'key':key,'dev_auroc':selection['dev_auroc'],**metrics(rows,scores[key],bootstrap=key in [frozen['best_cf'],frozen['best_contrast']])}
        base=key.rsplit('/',1)[0]
        result['mean_raw_cf']=float(masses[base]['raw_cf'].mean())
        result['mean_ignorance']=float(masses[base]['ignorance'].mean())
        result['cf_zero_fraction']=float(np.mean(masses[base]['raw_cf']==0))
        results[name]=result
    report={'validation_rows':len(rows),'frozen':frozen,'methods':results,'critical_note':'CF clipped contrast differs from matching baseline. BetP is monotone in contrast; no added ranking from mass remapping. Background null is approximate lexical reference. Independent final holdout still needed.'}
    (dest/'validation_metrics.json').write_text(json.dumps(report,indent=2))
    lines=['# 背景参照的语义证据','', '使用100张开发图片构建无标签视觉词汇背景，比较当前图片对问题词的支持与背景支持。二元简单支持质量保留Theta，再计算它与文字前提的冲突。超参数只在开发集选；本轮验证属于迭代检查。','', '| block | CF验证AUROC | 对比值验证AUROC |','|---|---|---|']
    for layer in LAYERS:
        lines.append(f'| {layer} | {results[f"L{layer}/cf"]["auroc"]:.4f} | {results[f"L{layer}/contrast"]["auroc"]:.4f} |')
    lines+=['',f'开发集选出的CF配置：`{frozen["best_cf"]}`。','', 'CF使用稳定−log(1−CF)排序；反对证据不足时CF=0，导致并列。BetP与未截断对比值单调等价，不能声称只是转成质量函数就提高了AUROC。背景分布并非对象不存在的真概率，尚需语义校准和独立确认。']
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'best_cf':next(v for v in results.values() if v['key']==frozen['best_cf']),'table':{k:round(v['auroc'],4) for k,v in results.items()}},indent=2))

if __name__=='__main__':
    main()
