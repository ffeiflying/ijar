"""Build visually checked small balanced image/question crossed controls.

These are synthetic mechanism checks on development images, not fresh test data.
"""
import json
from pathlib import Path
TASK=Path(__file__).resolve().parent.parent
dest=TASK/'results/controls'
rows=json.loads((dest/'source_images.json').read_text())
images={r['image_id']:r for r in rows}
concepts=[('cow',2404322),('horse',2403249),('zebra',150269),('flowers',2408126),('beds',2393859),('lamp',2402855),('bicycle',2410477),('cat',2406839)]
records=[]
for i,(concept,imageid) in enumerate(concepts):
    plural=concept in ['flowers','beds']
    question=f'What color {"are" if plural else "is"} the {concept}?'
    for label,source in [(0,imageid),(1,concepts[(i+4)%len(concepts)][1])]:
        row=images[source]
        records.append({'row_id':900000+2*i+label,'image_id':source,'question':question,'image_path':row['image_path'],'label':label,'question_group':i,'concept':concept,'annotation':'visual inspection of image_review.jpg: referenced object present/absent; synthetic false-premise control, not official MMMC label'})
assert all(sum(r['image_id']==imageid and r['label']==label for r in records)==1 for imageid in images if imageid in [x[1] for x in concepts] for label in [0,1])
(dest/'balanced_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
(dest/'PROTOCOL.md').write_text('''# Small balanced mechanism control

8 concepts,16 rows. For each exact question, one visually supported image and one image where the referenced object is absent. Each of8 images appears once in each label, removing pure image-identity and pure question-wording discrimination. Negative images use a fixed4-position shift, not label tuning. Image review montage visually inspected before labels assigned.

This is a small synthetic mechanism check using development images; not an independent accuracy benchmark. Official labels were not transferred after swapping images. Labels refer to manually checked presence/absence of the named object and false premise of the color question. Methods/configurations remain frozen before this control. Each input is independently scored with one forward, never aggregate multiple sampled answers.
''')
print('wrote',len(records),'balanced controls')
