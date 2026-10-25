"""Freeze a fresh image-disjoint confirmation set, no model scores consulted."""
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
import pyarrow as pa
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import DATA,SPLITS,TASK1
TASK=Path(__file__).resolve().parent.parent
dest=TASK/'results/confirmation'
dest.mkdir(parents=True,exist_ok=True)
assert not (dest/'protocol.json').exists(),'Do not replace existing confirmation protocol'
excluded=set()
excluded_sources={}
for path in list(SPLITS.glob('*.json'))+[DATA/'pilot.json']:
    obj=json.loads(path.read_text())
    if not isinstance(obj,list):
        continue
    ids={r['image_id'] for r in obj if isinstance(r,dict) and 'image_id' in r}
    excluded.update(ids)
    excluded_sources[str(path)]={'images':len(ids),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
manifest=json.loads((TASK1/'results/test_manifest.json').read_text())
tables=[]
sources=[]
for entry in manifest:
    if not entry['path'].endswith('.arrow'):
        continue
    path=DATA/'MMMC'/entry['path']
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.stat().st_size==entry['size'] and digest==entry['lfs']['oid']
    tables.append(pa.ipc.open_stream(str(path)).read_all())
    sources.append({'path':str(path),'sha256':digest})
table=pa.concat_tables(tables)
metadata=table.select([name for name in table.column_names if name!='image']).to_pylist()
groups=collections.defaultdict(list)
for i,row in enumerate(metadata):
    row['row_id']=i
    groups[row['image_id']].append(row)
eligible=sorted(imageid for imageid,rows in groups.items() if imageid not in excluded and any(r['conflict_type'] is None for r in rows) and any(r['conflict_type'] is not None for r in rows))
random.Random(20261025).shuffle(eligible)
selected=eligible[:300]
assert len(selected)==300 and not (set(selected)&excluded)
images=dest/'images'
images.mkdir(exist_ok=True)
records=[]
for imageid in selected:
    rows=groups[imageid]
    normal=next(r for r in rows if r['conflict_type'] is None)
    conflict=next(r for r in rows if r['conflict_type'] is not None)
    payload=table['image'][normal['row_id']].as_py()['bytes']
    assert payload==table['image'][conflict['row_id']].as_py()['bytes']
    path=images/f'{imageid}.img'
    assert not path.exists(),'Fresh image path unexpectedly exists'
    path.write_bytes(payload)
    for row in [normal,conflict]:
        records.append({**row,'image_path':str(path.resolve()),'label':int(row['conflict_type'] is not None)})
records_path=dest/'confirmation_records.json'
records_path.write_text(json.dumps(records,ensure_ascii=False,indent=2))
frozen_path=TASK/'results/frozen_dev_selection.json'
protocol={'seed':20261025,'images':300,'rows':len(records),'selected_image_ids':selected,'excluded_previous_image_count':len(excluded),'eligible_unused_images':len(eligible),'excluded_sources':excluded_sources,'records_sha256':hashlib.sha256(records_path.read_bytes()).hexdigest(),'frozen_selection_sha256':hashlib.sha256(frozen_path.read_bytes()).hexdigest(),'background_sha256':hashlib.sha256((TASK/'results/background.npz').read_bytes()).hexdigest(),'primary':'frozen dev winner L24/attention/A1/cf','baseline':'same readout, reference exponent and mass assignment at block0','selection_rule':'uniform shuffle remaining image pairs; no score/answer filtering; all official conflict types retained','confirmation_rule':'evaluate these two configurations only; no test-set layer/threshold/hyperparameter tuning','sources':sources,'conflict_types':dict(collections.Counter(str(r['conflict_type']) for r in records))}
(dest/'protocol.json').write_text(json.dumps(protocol,indent=2))
print(json.dumps({k:protocol[k] for k in ['images','rows','excluded_previous_image_count','eligible_unused_images','conflict_types']},indent=2))
