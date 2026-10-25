"""Label-separated diagnostic audit; lexical overlap is not semantic accuracy."""
import collections
import json
import re
import statistics
from pathlib import Path

TASK = Path(__file__).resolve().parent.parent
EXPERIMENT = TASK.parents[1]
rows = {r['row_id']:r for r in json.loads((EXPERIMENT/'shared/data/splits/dev.json').read_text())}

def normalize(text):
    return re.sub('[^a-z]','',text.lower())

summary = {}
examples = []
for variant in ['diagnostics','cosine_diagnostics','centered_diagnostics']:
    records = [json.loads(f.read_text()) for f in sorted((TASK/f'results/{variant}/dev').glob('*.json'))]
    if not records:
        continue
    positions = {}
    for position in records[0]['positions']:
        hit = {0:[],1:[]}
        top1 = collections.Counter()
        entropies = []
        for d in records:
            for patch in d['positions'][position].values():
                top1[patch['top_tokens'][0]] += 1
                if 'normalized_entropy' in patch:
                    entropies.append(patch['normalized_entropy'])
            for query in d['content_tokens']:
                target = normalize(query['token'])
                if len(target)<3:
                    continue
                candidates = {normalize(word) for patch in query['patches'] for word in d['positions'][position][str(patch)]['top_tokens']}
                hit[rows[d['row_id']]['label']].append(int(target in candidates))
        positions[position] = {'top1_frequency':top1.most_common(12),'exact_subword_hit_at_10_over_top3_patches':{str(label):{'hits':sum(vals),'queries':len(vals),'rate':statistics.mean(vals) if vals else None} for label,vals in hit.items()},'mean_normalized_entropy':statistics.mean(entropies) if entropies else None}
    query_patches = [p for d in records for q in d['content_tokens'] for p in q['patches']]
    side = round(records[0]['visual_token_count']**.5)
    summary[variant] = {'n_rows':len(records),'n_images':len({d['image_id'] for d in records}),'mean_seconds_excluding_first':statistics.mean(d['seconds'] for d in records[1:]),'bottom_row_fraction_of_query_top3':sum(p>=side*(side-1) for p in query_patches)/len(query_patches),'positions':positions}
    for d in records:
        if d['row_id'] not in [1591,1590,957,956]:
            continue
        examples.append({'variant':variant,'row_id':d['row_id'],'question':d['question'],'official_label':rows[d['row_id']]['label'],'image_path':rows[d['row_id']]['image_path'],'queries':d['content_tokens'],'readout':d['positions']})
(TASK/'results/diagnostic_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(TASK/'results/review_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
lines = ['# 第0层语义读出诊断','', '这些数字是词片段精确重合率，不是语义准确率、CF 或冲突检测 AUROC。没有使用标签训练或选词。问题中的动词、属性词和子词也可能计入，因此这里只作读出失败筛查。','', '| 读出 | 位置 | 正常问题词片段命中 | 冲突问题词片段命中 |','|---|---|---|---|']
for variant,s in summary.items():
    for position,p in s['positions'].items():
        groups = p['exact_subword_hit_at_10_over_top3_patches']
        display = [f"{groups[str(label)]['hits']}/{groups[str(label)]['queries']}" for label in [0,1]]
        lines.append(f'| {variant} | {position} | {display[0]} | {display[1]} |')
lines += ['', '## 解释边界','', '- 正常问题中的对象也未能可靠读出时，冲突问题的低匹配不能解释为冲突。','- 不能使用 1−词语匹配分数 作为反对质量；也不能仅凭这些输出的低熵认定视觉证据可靠。','- attention 选中区域的语义支持不足，与图像中对象不存在是不同命题。','- 当前只是固定第0层可行性检查；尚无新的语义 ACF 性能结论。','', '## 样例','']
for example in examples:
    if example['variant']!='diagnostics':
        continue
    lines.append(f"- 问题：{example['question']}；官方标签：{example['official_label']}。")
    for query in example['queries']:
        decoded = [example['readout']['post_attention'][str(p)]['top_tokens'][:3] for p in query['patches']]
        lines.append(f"  - query `{query['token']}` → patches {query['patches']} → {decoded}")
(TASK/'results/RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:{'rows':v['n_rows'],'bottom_row_fraction':v['bottom_row_fraction_of_query_top3'],'positions':{p:x['exact_subword_hit_at_10_over_top3_patches'] for p,x in v['positions'].items()}} for k,v in summary.items()},ensure_ascii=False,indent=2))
