"""Verify official train shard and exclude every test image by ID and byte hash."""
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
import pyarrow as pa
TASK=Path(__file__).resolve().parent.parent
EXPERIMENT=TASK.parents[1]
source=TASK/'data/train_source'
partial=source/'data-00000-of-00012.arrow.partial'
path=source/'data-00000-of-00012.arrow'
expected='7f7e435324898dffb86f082b13c1ac4f316d1e5e9bb451c41302f1acb9feab8e'
if not path.exists():
    assert partial.stat().st_size==485823504
    assert hashlib.sha256(partial.read_bytes()).hexdigest()==expected
    partial.replace(path)
else:
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
table=pa.ipc.open_stream(str(path)).read_all()
rows=table.select([n for n in table.column_names if n!='image']).to_pylist()
test_ids=set()
test_hashes=set()
for file in sorted((EXPERIMENT/'shared/data/MMMC/test').glob('*.arrow')):
    t=pa.ipc.open_stream(str(file)).read_all()
    for i,imageid in enumerate(t['image_id'].to_pylist()):
        if imageid in test_ids:
            continue
        test_ids.add(imageid)
        test_hashes.add(hashlib.sha256(t['image'][i].as_py()['bytes']).hexdigest())
groups=collections.defaultdict(list)
for i,row in enumerate(rows):
    row['source_index']=i
    groups[row['image_id']].append(row)
eligible=[]
overlap_id=overlap_hash=0
for imageid,group in groups.items():
    if imageid in test_ids:
        overlap_id+=1
        continue
    if not any(r['conflict_type'] is None for r in group) or not any(r['conflict_type'] is not None for r in group):
        continue
    digest=hashlib.sha256(table['image'][group[0]['source_index']].as_py()['bytes']).hexdigest()
    if digest in test_hashes:
        overlap_hash+=1
        continue
    eligible.append(imageid)
random.Random(20261026).shuffle(eligible)
assert len(eligible)>=1200,('Not enough disjoint image pairs',len(eligible))
splits={'train':eligible[:1000],'calibration':eligible[1000:1200]}
outputs={}
image_dir=TASK/'data/train_images'
image_dir.mkdir(parents=True,exist_ok=True)
for name,imageids in splits.items():
    records=[]
    for imageid in imageids:
        group=groups[imageid]
        selected=[next(r for r in group if r['conflict_type'] is None),next(r for r in group if r['conflict_type'] is not None)]
        payload=table['image'][selected[0]['source_index']].as_py()['bytes']
        assert payload==table['image'][selected[1]['source_index']].as_py()['bytes']
        target=image_dir/f'{imageid}.img'
        if target.exists():
            assert target.read_bytes()==payload
        else:
            target.write_bytes(payload)
        for row in selected:
            records.append({**row,'row_id':10_000_000+row['source_index'],'image_path':str(target.resolve()),'label':int(row['conflict_type'] is not None),'official_split':'train','source_shard':path.name})
    records_path=TASK/f'data/{name}.json'
    payload=json.dumps(records,ensure_ascii=False,indent=2)
    if records_path.exists():
        assert records_path.read_text()==payload
    else:
        records_path.write_text(payload)
    outputs[name]={'images':len(imageids),'rows':len(records),'sha256':hashlib.sha256(records_path.read_bytes()).hexdigest(),'conflict_types':dict(collections.Counter(str(r['conflict_type']) for r in records))}
report={'revision':'0bdbffbb35bf1a2569468fd939a5f70290442e55','shard_sha256':expected,'shard_rows':len(rows),'unique_images':len(groups),'excluded_all_test_image_ids':len(test_ids),'excluded_all_test_content_hashes':len(test_hashes),'train_test_id_overlap_removed':overlap_id,'train_test_hash_overlap_removed':overlap_hash,'eligible_pairs':len(eligible),'outputs':outputs,'seed':20261026,'calibration_role':'model selection only; images disjoint from train and entire official test','environment_changed':False}
(TASK/'data/train_protocol.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
