"""Fresh300-image confirmation, frozen train/calibration choice and no score filters."""
import collections,hashlib,json,random,sys
from pathlib import Path
import pyarrow as pa
TASK=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(TASK.parents[1]/'shared'))
from layout import DATA,SPLITS,TASK1
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
dest=TASK/'results/confirmation';dest.mkdir(parents=True,exist_ok=True)
assert not (dest/'protocol.json').exists()
frozen=TASK/'results/supervised_mil/frozen.json'
choices=json.loads(frozen.read_text())['winners']
primary=max((k for k in choices if k.endswith('evidence')),key=lambda k:choices[k]['calibration_auroc'])
excluded=set();sources={}
paths=list(SPLITS.glob('*.json'))+[DATA/'pilot.json',TASK.parent/'16_reference_semantic_evidence/results/confirmation/confirmation_records.json']
for p in paths:
    obj=json.loads(p.read_text())
    if not isinstance(obj,list): continue
    ids={r['image_id'] for r in obj if isinstance(r,dict) and 'image_id' in r}
    excluded.update(ids);sources[str(p)]={'images':len(ids),'sha256':sha(p)}
tables=[];arrows=[]
for entry in json.loads((TASK1/'results/test_manifest.json').read_text()):
    if not entry['path'].endswith('.arrow'):continue
    p=DATA/'MMMC'/entry['path'];assert p.stat().st_size==entry['size'] and sha(p)==entry['lfs']['oid']
    tables.append(pa.ipc.open_stream(str(p)).read_all());arrows.append({'path':str(p),'sha256':sha(p)})
table=pa.concat_tables(tables);groups=collections.defaultdict(list)
for i,r in enumerate(table.select([k for k in table.column_names if k!='image']).to_pylist()):
    groups[r['image_id']].append({**r,'row_id':i})
eligible=sorted(set(groups)-excluded);random.Random(20261027).shuffle(eligible)
selected=eligible[:300];assert len(selected)==300 and not (set(selected)&excluded)
images=dest/'images';images.mkdir(exist_ok=True);records=[]
for image_id in selected:
    group=groups[image_id];normal=next(r for r in group if r['conflict_type'] is None);conflict=next(r for r in group if r['conflict_type'] is not None)
    payload=table['image'][normal['row_id']].as_py()['bytes'];assert payload==table['image'][conflict['row_id']].as_py()['bytes']
    path=images/f'{image_id}.img';assert not path.exists();path.write_bytes(payload)
    for r in [normal,conflict]:records.append({**r,'image_path':str(path.resolve()),'label':int(r['conflict_type'] is not None)})
rp=dest/'confirmation_records.json';rp.write_text(json.dumps(records,ensure_ascii=False,indent=2))
protocol={'seed':20261027,'images':300,'rows':600,'excluded_previous_images':len(excluded),'eligible_unused_images':len(eligible),'selected_image_ids':selected,'excluded_sources':sources,'official_sources':arrows,'records_sha256':sha(rp),'frozen_selection_sha256':sha(frozen),'primary':primary,'primary_config':choices[primary],'fixed_controls':list(choices),'selection_rule':'uniform image shuffle, no scores/answers/label-type filters','evaluation_rule':'report frozen primary and all fixed controls; do not select on confirmation','prior_confirmation':'task16 never read for feature or model selection; IDs used only to exclude images','conflict_types':dict(collections.Counter(str(r['conflict_type']) for r in records))}
(dest/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps({k:v for k,v in protocol.items() if k not in ['selected_image_ids','excluded_sources','official_sources']},indent=2))
