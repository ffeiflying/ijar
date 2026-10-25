"""Dev-only selection and frozen validation evaluation, never select on val."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TASK=Path(__file__).resolve().parent.parent
EXPERIMENT=TASK.parents[1]
SPLITS=EXPERIMENT/'shared/data/splits'
LAYERS=[0,8,16,20,24,27,31]

def family(key):
    if '_source_' in key and '_cross_cf_' in key:
        return 'source_DS'
    if '_source_' in key and '_discounted_cf_' in key:
        return 'source_discounted'
    if '_acf_' in key and 'image_only' not in key:
        return 'feature_DS'
    if '_grounding_cf_' in key:
        return 'lexical_grounding'
    if '_interhead_cf_' in key:
        return 'head_conflict'
    if '_nll_' in key:
        return 'matching_baseline'
    return None

def load(split):
    rows=json.loads((SPLITS/f'{split}.json').read_text())
    files=[TASK/f'results/features_v2/{split}/{r["row_id"]}.npz' for r in rows]
    missing=[str(p) for p in files if not p.exists()]
    assert not missing,f'{split}: {len(missing)} missing; first {missing[:2]}'
    loaded=[]
    for path in files:
        with np.load(path) as z:
            loaded.append({k:float(z[k]) for k in z.files})
    keys=set(loaded[0])
    assert all(set(z)==keys for z in loaded),'Inconsistent feature schema'
    scores={k:np.array([z[k] for z in loaded]) for k in sorted(keys)}
    assert all(np.isfinite(v).all() for v in scores.values())
    return rows,scores

def metrics(rows,score,bootstrap=False):
    y=np.array([r['label'] for r in rows])
    result={'auroc':float(roc_auc_score(y,score)),'auprc':float(average_precision_score(y,score))}
    pairs=collections.defaultdict(list)
    for i,r in enumerate(rows):
        pairs[r['image_id']].append(i)
    differences=[]
    for indices in pairs.values():
        normal=[i for i in indices if y[i]==0]
        conflict=[i for i in indices if y[i]==1]
        if normal and conflict:
            differences.append(float(score[conflict].mean()-score[normal].mean()))
    result['paired_direction_fraction']=float(np.mean(np.array(differences)>0)) if differences else None
    result['mean_paired_delta']=float(np.mean(differences)) if differences else None
    if bootstrap:
        groups=list(pairs.values())
        rng=np.random.default_rng(42)
        auc=[]
        for _ in range(1000):
            index=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
            auc.append(roc_auc_score(y[index],score[index]))
        result['image_bootstrap_auroc95ci']=np.quantile(auc,[.025,.975]).tolist()
    return result

def controls(rows):
    dev=json.loads((SPLITS/'dev.json').read_text())
    def words(q):
        return re.findall('[a-z]{3,}',q.lower())
    freq=collections.Counter(w for r in dev for w in words(r['question']))
    return {
        'question_characters':np.array([len(r['question']) for r in rows]),
        'question_word_rarity':np.array([np.mean([-np.log((freq[w]+1)/(sum(freq.values())+len(freq))) for w in words(r['question'])]) for r in rows]),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['freeze','validation'],required=True)
    a=p.parse_args()
    frozen_path=TASK/'results/frozen_dev_selection.json'
    if a.mode=='freeze':
        rows,scores=load('dev')
        y=np.array([r['label'] for r in rows])
        auc={k:float(roc_auc_score(y,v)) for k,v in scores.items() if family(k)}
        selections={}
        for layer in LAYERS:
            for name in ['source_DS','source_discounted','feature_DS','lexical_grounding','head_conflict','matching_baseline']:
                keys=[k for k in auc if k.startswith(f'L{layer}_') and family(k)==name]
                key=max(keys,key=lambda k:(auc[k],k))
                selections[f'L{layer}/{name}']={'key':key,'dev_auroc':auc[key]}
        evidence=[k for k in auc if family(k)!='matching_baseline']
        best=max(evidence,key=lambda k:(auc[k],k))
        baseline=max((k for k in auc if family(k)=='matching_baseline'),key=lambda k:(auc[k],k))
        frozen={'selection':'dev-only AUROC, fixed high-is-conflict orientation; no sign inversion','dev_rows':len(rows),'dev_sha256':hashlib.sha256((SPLITS/'dev.json').read_bytes()).hexdigest(),'validation_sha256':hashlib.sha256((SPLITS/'validation.json').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((TASK/'results/config_v2.json').read_bytes()).hexdigest(),'best_evidence':best,'best_matching_baseline':baseline,'selections':selections,'n_dev_configurations':len(auc),'limitations':'exploratory many configurations; lexical frame and entropy discounts uncalibrated; no final-test claim'}
        if frozen_path.exists():
            assert json.loads(frozen_path.read_text())==frozen,'Existing frozen selection differs; do not overwrite'
        else:
            frozen_path.write_text(json.dumps(frozen,indent=2))
        (TASK/'results/dev_all_metrics.json').write_text(json.dumps(auc,indent=2))
        top=sorted(auc,key=auc.get,reverse=True)[:12]
        print(json.dumps({'best_evidence':best,'best_evidence_auc':auc[best],'best_baseline':baseline,'baseline_auc':auc[baseline],'top12':{k:auc[k] for k in top}},indent=2))
        return
    frozen=json.loads(frozen_path.read_text())
    assert hashlib.sha256((TASK/'results/config_v2.json').read_bytes()).hexdigest()==frozen['config_sha256']
    rows,scores=load('validation')
    devrows,devscores=load('dev')
    results={}
    for name,selection in frozen['selections'].items():
        key=selection['key']
        results[name]={'key':key,'dev_auroc':selection['dev_auroc'],**metrics(rows,scores[key],bootstrap=key in [frozen['best_evidence'],frozen['best_matching_baseline']])}
    control={k:metrics(rows,v) for k,v in controls(rows).items()}
    control['frame_coverage']=metrics(rows,scores['frame_coverage'])
    control['content_count']=metrics(rows,scores['content_count'])
    control['image_only_by_layer']={str(l):metrics(rows,scores[f'L{l}_attn_T1_image_only_acf']) for l in LAYERS}
    norm=lambda q:' '.join(q.lower().split())
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):
        groups[norm(r['question'])].append(i)
    matched=[ix for ix in groups.values() if len({rows[i]['label'] for i in ix})==2]
    matched_scores={}
    for key in [frozen['best_evidence'],frozen['best_matching_baseline']]:
        matched_scores[key]=[metrics([rows[i] for i in ix],scores[key][ix])['auroc'] for ix in matched]
    report={'n_validation_rows':len(rows),'n_validation_images':len({r['image_id'] for r in rows}),'frozen':frozen,'methods':results,'controls':control,'same_question_groups':len(matched),'same_question_rows':sum(map(len,matched)),'same_question_mean_auroc':{k:float(np.mean(v)) if v else None for k,v in matched_scores.items()},'mean_all_layers_extraction_seconds':float(scores['seconds_all_layers'].mean()),'timing_note':'all7 layers and all candidate readouts measured together; not single-detector latency'}
    (TASK/'results/validation_frozen_metrics.json').write_text(json.dumps(report,indent=2))
    names=['source_DS','source_discounted','feature_DS','lexical_grounding','head_conflict','matching_baseline']
    lines=['# 中后层与第0层：单次前向内部证据对照','',f'开发集{len(devrows)}条；按图片互斥的验证集{len(rows)}条。所有配置在开发集选择后冻结。共探索{frozen["n_dev_configurations"]}个配置，结果属于探索性验证，不能作为最终泛化结论。','', '| block（从0编号） | 来源DS | 来源折扣CF | 特征DS | 词汇匹配CF | 跨head CF | 纯匹配基线 |','|---|---|---|---|---|---|---|']
    for l in LAYERS:
        vals=[f"{results[f'L{l}/{name}']['auroc']:.4f}" for name in names]
        lines.append('| '+str(l)+' | '+' | '.join(vals)+' |')
    best=frozen['best_evidence']
    best_result=next(v for v in results.values() if v['key']==best)
    lines+=['',f'开发集选出的证据候选：`{best}`；开发AUROC {best_result["dev_auroc"]:.4f}，验证AUROC {best_result["auroc"]:.4f}。', '', '## 解释边界','', '- DS特征分解和跨来源融合使用新版 evidence_v2；logscore是稳定的−log(1−CF)，不是原始CF。','- 假设空间目前是开发问题中的词汇token标签；未建立严格的语义等价类。同义词、子词和多对象共存仍是限制。','- 折扣质量由归一化熵启发式给出，未证明是校准的语义可靠性。','- head或图文来源可能相关，精确计算冲突公式不代表独立性假设成立。','- 匹配CF表示内部语义锚定不一致，不自动证明图像中对象不存在。','- 不生成回答、不做sampling；全部实验只写入本任务目录，旧结果未修改。','', '## 对照','', '```json',json.dumps(control,ensure_ascii=False,indent=2),'```']
    (TASK/'results/RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'best_evidence':best_result,'table':{name:round(v['auroc'],4) for name,v in results.items()},'question_controls':{k:v for k,v in control.items() if k!='image_only_by_layer'}},indent=2))

if __name__=='__main__':
    main()
