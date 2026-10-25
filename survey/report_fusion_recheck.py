"""Build a reviewable report from the 2026-10-16 fusion pilot outputs."""
import csv
import html
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fusion_report_20261016'


def read(name):
    return list(csv.DictReader((ROOT/'results'/name/'samples.csv').open()))


def main():
    OUT.mkdir(exist_ok=True)
    old=read('fusion_recheck_20261016_old_freeform')
    fresh=read('fusion_recheck_20261016_fresh_freeform')
    validation={}
    for name,rows in [('old',old),('fresh',fresh)]:
        assert len(rows)==800, (name,len(rows))
        keys=[(r['condition'],r['sample_id'],r['mode']) for r in rows]
        assert len(set(keys))==len(keys)
        index={(r['condition'],r['sample_id'],r['mode']):r for r in rows}
        native=[r for r in rows if r['mode']=='native']
        assert all(r['response']==index[(r['condition'],r['sample_id'],'zero')]['response'] for r in native)
        validation[name]=dict(n_images=len(native),n_records=len(rows),zero_equal_native=True,
                             all_labels_unadjudicated=all(r['correct']=='-1' for r in rows),
                             max_forward_calls=max(int(r['n_forward']) for r in rows))
    old_native={(r['condition'],r['sample_id']):r['response'] for r in old if r['mode']=='native'}
    new_native={(r['condition'],r['sample_id']):r['response'] for r in fresh if r['mode']=='native'}
    assert old_native==new_native
    validation['native_equal_across_runs']=True
    for name in ['old','fresh']:
        meta=json.loads((ROOT/'results'/f'fusion_recheck_20261016_{name}_freeform/metadata.json').read_text())
        validation[name]['metadata']=meta
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    merged={}
    for rows,mode,label in [(old,'native','native'),(old,'fused','old_fused'),
                            (fresh,'fused','fresh_fused'),(old,'wrong_image','old_wrong_image'),
                            (fresh,'wrong_image','fresh_wrong_image')]:
        for r in rows:
            if r['mode']!=mode:continue
            key=(r['condition'],r['sample_id'])
            item=merged.setdefault(key,dict(condition=r['condition'],sample_id=r['sample_id'],
                reference_target=r['reference_target'],reference_distractor=r['reference_distractor']))
            item[label]=r['response']
    fields=['condition','sample_id','reference_target','reference_distractor','native','old_fused',
            'fresh_fused','old_wrong_image','fresh_wrong_image']
    with (OUT/'freeform_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(merged.values())
    rng=random.Random(20261016)
    review=[]
    key=[]
    for i,item in enumerate(merged.values()):
        responses=[(m,item[m]) for m in fields[4:]];rng.shuffle(responses)
        for j,(mode,response) in enumerate(responses):
            code=f'{i:03d}-{j}'
            review.append(dict(review_id=code,condition=item['condition'],sample_id=item['sample_id'],
                reference_target=item['reference_target'],reference_distractor=item['reference_distractor'],
                response=response,semantic_correct='',attack_influenced='',notes=''))
            key.append(dict(review_id=code,mode=mode))
    for filename,rows in [('blind_review.csv',review),('blind_review_key.csv',key)]:
        with (OUT/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    esc=html.escape
    page=['<!doctype html><meta charset="utf-8"><title>InternVL fusion freeform pilot</title>',
          '<style>body{font:15px sans-serif;margin:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}img{width:200px}th{position:sticky;top:0;background:#eee}</style>',
          '<h1>InternVL fusion freeform pilot</h1><p>100 attack + 100 clean. Complete answers; semantic correctness requires adjudication. Keyword matches are not accuracy.</p>',
          '<table><tr><th>Image / references</th>'+''.join('<th>'+esc(k)+'</th>' for k in fields[4:])+'</tr>']
    for item in merged.values():
        folder='datasets/STA/images' if item['condition']=='attack' else 'typo-parameter/clean/images'
        src='../../'+folder+'/'+item['sample_id']
        page.append('<tr><td><img loading="lazy" src="'+esc(src)+'"><br>'+esc(item['condition']+' '+item['sample_id'])+
            '<br>Target: '+esc(item['reference_target'])+'<br>Distractor: '+esc(item['reference_distractor'])+'</td>'+
            ''.join('<td>'+esc(item[k])+'</td>' for k in fields[4:])+'</tr>')
    page.append('</table>')
    (OUT/'freeform_review.html').write_text('\n'.join(page))
    print(json.dumps({k:v for k,v in validation.items() if k=='native_equal_across_runs'},indent=2))
    print('Report:',OUT)


if __name__=='__main__':main()
