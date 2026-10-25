"""Offline supervised DS calibration, frozen before validation scoring."""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

TASK=Path(__file__).resolve().parent.parent
SOURCE=TASK.parent/'14_multilayer_semantic_evidence'
sys.path.insert(0,str(SOURCE/'code'))
from evaluate import load, metrics, controls
LAYERS=[0,8,16,20,24,27,31]
FAMILIES={'matching_only':['matching_baseline'],'sources_only':['source_DS','source_discounted'],'semantic_all':['matching_baseline','source_DS','feature_DS','lexical_grounding','head_conflict']}

def fuse(logratios,reliability,scale):
    e=logratios/scale
    committed=(-np.expm1(-np.abs(e)))*np.asarray(reliability)[None,:]
    effective=-np.log1p(-np.minimum(committed,1-1e-15))
    ep=np.where(e>=0,effective,0).sum(-1)
    en=np.where(e<0,effective,0).sum(-1)
    base=np.logaddexp(-ep,-en)
    lognc=base+np.log1p(-np.exp(-ep-en-base))
    mass_c=(-np.expm1(-ep))*np.exp(-en-lognc)
    mass_n=(-np.expm1(-en))*np.exp(-ep-lognc)
    ignorance=np.exp(-ep-en-lognc)
    conflict=(-np.expm1(-ep))*(-np.expm1(-en))
    assert np.allclose(mass_c+mass_n+ignorance,1,atol=1e-9)
    return {'belief':mass_c,'betp':mass_c+.5*ignorance,'cf':conflict,'logcf':-lognc,'ignorance':ignorance,'normal_belief':mass_n}

def check():
    rng=np.random.default_rng(42)
    error=0.
    for _ in range(30):
        e=rng.normal(size=(1,5))
        r=rng.uniform(size=5)
        mass=-np.expm1(-np.abs(e[0]))*r
        exact={0:0.,1:0.,2:0.,3:0.}
        for choices in itertools.product([False,True],repeat=5):
            intersection=3
            weight=1.
            for i,choice in enumerate(choices):
                intersection&=(2 if e[0,i]>0 else 1) if choice else 3
                weight*=mass[i] if choice else 1-mass[i]
            exact[intersection]+=weight
        result=fuse(e,r,1)
        error=max(error,abs(result['cf'][0]-exact[0]),abs(result['belief'][0]-exact[2]/(1-exact[0])),abs(result['ignorance'][0]-exact[3]/(1-exact[0])))
    assert error<1e-12
    return {'cases':30,'independent_mass_enumeration_max_error':error}

def fit_linear(X,y,kind='lda',C=1.):
    scaler=StandardScaler().fit(X)
    transformed=scaler.transform(X)
    model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto') if kind=='lda' else LogisticRegression(C=C,max_iter=3000,random_state=42)
    model.fit(transformed,y)
    coef=model.coef_[0]/scaler.scale_
    intercept=float(model.intercept_[0]-np.dot(model.coef_[0],scaler.mean_/scaler.scale_))
    assert np.allclose(X@coef+intercept,model.decision_function(transformed),atol=1e-8)
    return {'coef':coef.tolist(),'intercept':intercept}

def predict(model,X):
    return X@np.asarray(model['coef'])+model['intercept']

def data(rows,scores,keys):
    return np.column_stack([scores[k] for k in keys])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['train','validation'],required=True)
    args=parser.parse_args()
    dest=TASK/'results'
    dest.mkdir(parents=True,exist_ok=True)
    frozen_path=dest/'frozen_calibrators.json'
    if args.mode=='train':
        assert not frozen_path.exists(),'Do not overwrite existing frozen model'
        math_check=check()
        selection=json.loads((SOURCE/'results/frozen_dev_selection.json').read_text())
        rows,scores=load('dev')
        y=np.array([r['label'] for r in rows])
        groups=np.array([r['image_id'] for r in rows])
        unique=np.unique(groups)
        splits=[]
        for train,test in KFold(5,shuffle=True,random_state=42).split(unique):
            splits.append((np.flatnonzero(np.isin(groups,unique[train])),np.flatnonzero(np.isin(groups,unique[test]))))
        calibrators={}
        oof_ratios={}
        oof_metrics={}
        for group,names in FAMILIES.items():
            models=[]
            columns=[]
            allkeys=[]
            for layer in LAYERS:
                keys=[selection['selections'][f'L{layer}/{name}']['key'] for name in names]
                allkeys+=keys
                X=data(rows,scores,keys)
                oof=np.empty(len(y))
                for train,test in splits:
                    m=fit_linear(X[train],y[train])
                    oof[test]=predict(m,X[test])
                models.append({'layer':layer,'keys':keys,'linear':fit_linear(X,y)})
                columns.append(oof)
            oof=np.column_stack(columns)
            skill=np.clip([1-brier_score_loss(y,expit(oof[:,j]))/.25 for j in range(len(LAYERS))],0,.95)
            discounts={'none':np.ones(len(LAYERS)),'brier_skill':skill,'sqrt_brier_skill':np.sqrt(skill)}
            calibrators[group]={'experts':models,'reliability':{k:v.tolist() for k,v in discounts.items()}}
            oof_ratios[group]=oof
            for discount,r in discounts.items():
                for scale in [1.,2.,4.,16.]:
                    values=fuse(oof,r,scale)
                    for statistic in ['belief','betp','cf','logcf']:
                        name=f'{group}/{discount}/T{scale:g}/{statistic}'
                        oof_metrics[name]=float(roc_auc_score(y,values[statistic]))
            # Same-input non-evidential calibration baseline.
            X=data(rows,scores,allkeys)
            baseline={}
            for C in [.01,.1,1.,10.]:
                oof=np.empty(len(y))
                for train,test in splits:
                    m=fit_linear(X[train],y[train],kind='lr',C=C)
                    oof[test]=predict(m,X[test])
                baseline[str(C)]={'dev_oof_auroc':float(roc_auc_score(y,oof)),'linear':fit_linear(X,y,kind='lr',C=C)}
            best=max(baseline,key=lambda k:baseline[k]['dev_oof_auroc'])
            calibrators[group]['lr_baseline']={'keys':allkeys,'C':float(best),**baseline[best]}
        best=max(oof_metrics,key=lambda k:(oof_metrics[k],k))
        bestcf=max((k for k in oof_metrics if k.endswith('/cf') or k.endswith('/logcf')),key=lambda k:(oof_metrics[k],k))
        bestbelief=max((k for k in oof_metrics if k.endswith('/belief') or k.endswith('/betp')),key=lambda k:(oof_metrics[k],k))
        bestlr=max(calibrators,key=lambda k:calibrators[k]['lr_baseline']['dev_oof_auroc'])
        frozen={'supervised':True,'frame':['normal','modality_conflict'],'dev_rows':len(rows),'selected':best,'best_cf':bestcf,'best_belief':bestbelief,'best_lr_group':bestlr,'dev_oof_auroc':oof_metrics[best],'dev_cf_oof_auroc':oof_metrics[bestcf],'oof_note':'grouped by image; input feature configurations were preselected on full dev, so OOF is for tuning, not unbiased generalization','calibrators':calibrators,'math_check':math_check,'source_selection_sha256':hashlib.sha256((SOURCE/'results/frozen_dev_selection.json').read_bytes()).hexdigest(),'validation_seen':False,'method_limitations':'Gaussian likelihood assumptions; correlated layer experts; known DS simple-support construction; no standalone novelty claim'}
        frozen_path.write_text(json.dumps(frozen,indent=2))
        (dest/'dev_oof_metrics.json').write_text(json.dumps(oof_metrics,indent=2))
        print(json.dumps({k:frozen[k] for k in ['selected','best_cf','best_belief','best_lr_group','dev_oof_auroc','dev_cf_oof_auroc','math_check']},indent=2))
        return
    frozen=json.loads(frozen_path.read_text())
    assert frozen['source_selection_sha256']==hashlib.sha256((SOURCE/'results/frozen_dev_selection.json').read_bytes()).hexdigest()
    rows,scores=load('validation')
    cache={}
    results={}
    for key in set([frozen['selected'],frozen['best_cf'],frozen['best_belief']]):
        group,discount,temp,statistic=key.split('/')
        model=frozen['calibrators'][group]
        if group not in cache:
            cache[group]=np.column_stack([predict(e['linear'],data(rows,scores,e['keys'])) for e in model['experts']])
        values=fuse(cache[group],model['reliability'][discount],float(temp[1:]))
        results[key]={'statistic':statistic,**metrics(rows,values[statistic],bootstrap=True),'mean_mass_conflict_class':float(values['belief'].mean()),'mean_ignorance':float(values['ignorance'].mean()),'mean_internal_conflict':float(values['cf'].mean())}
    lr={}
    for group,model in frozen['calibrators'].items():
        baseline=model['lr_baseline']
        score=predict(baseline['linear'],data(rows,scores,baseline['keys']))
        lr[group]={'C':baseline['C'],'dev_oof_auroc':baseline['dev_oof_auroc'],**metrics(rows,score,bootstrap=group==frozen['best_lr_group'])}
    report={'validation_rows':len(rows),'supervised':True,'selected_on_dev':frozen['selected'],'best_cf_selected_on_dev':frozen['best_cf'],'evidence':results,'lr_baselines':lr,'question_only_controls':{k:metrics(rows,v) for k,v in controls(rows).items()},'main_limitations':['supervised calibration, not zero-shot','CF is disagreement, belief is class support; reported separately','same-feature LR must be compared before attributing value to evidence theory','validation is exploratory and has been used by earlier tasks; final fresh confirmation required']}
    (dest/'validation_metrics.json').write_text(json.dumps(report,indent=2))
    lines=['# 有监督内部证据校准','', '输入来自一次InternVL前向的内部读出。开发200条、验证400条，按图片互斥。层级高斯/LDA专家离线拟合，转成单焦元+Theta质量后进行DS融合。此方法属于有监督校准，不能作为零样本方法。','', '| 开发集冻结的输出 | 验证 AUROC | 95%图片bootstrap区间 |','|---|---|---|']
    for key,result in results.items():
        lo,hi=result['image_bootstrap_auroc95ci']
        lines.append(f'| {key} | {result["auroc"]:.4f} | [{lo:.4f}, {hi:.4f}] |')
    lines+=['', '## 相同输入的LR基线','', '| 输入组 | 验证AUROC |','|---|---|']
    for name,result in lr.items():
        lines.append(f'| {name} | {result["auroc"]:.4f} |')
    lines+=['', 'Bel/BetP不是CF。二元任务假设空间、似然到简单支持质量、DS融合都是已有基础构造，当前结果本身不构成新颖性证明。若与LR相近，不能声称证据理论提高了分类能力。旧工作结果未改动。']
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
