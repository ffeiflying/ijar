"""Reserve fresh image-disjoint ACF test after prior development/validation/control sets."""
import hashlib,json,random,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import DATA,SPLITS,TASK1
import pyarrow as pa
rows=[]
for e in json.loads((TASK1/'results/test_manifest.json').read_text()):
 if not e['path'].endswith('.arrow'):continue
 p=DATA/'MMMC'/e['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==e['lfs']['oid']
 rows.extend(pa.ipc.open_stream(str(p)).read_all().to_pylist())
groups=defaultdict(list)
for i,r in enumerate(rows):r['row_id']=i;groups[r['image_id']].append(r)
pairs=[]
for image_id,rs in groups.items():
 clean=[r for r in rs if r['conflict_type'] is None];bad=[r for r in rs if r['conflict_type'] is not None]
 if clean and bad:pairs.append((image_id,clean[0],bad[0]))
random.Random(42).shuffle(pairs)
seen=set()
for split in ['dev','validation','final']:seen.update(r['image_id'] for r in json.loads((SPLITS/f'{split}.json').read_text()))
chosen=pairs[600:900];assert len(chosen)==300 and not({p[0] for p in chosen}&seen)
records=[]
for image_id,clean,bad in chosen:
 assert clean['image']['bytes']==bad['image']['bytes']
 image=DATA/'images'/f'{image_id}.img';image.write_bytes(clean['image']['bytes'])
 for r in [clean,bad]:records.append({**{k:v for k,v in r.items() if k!='image'},'image_path':str(image),'label':int(r['conflict_type'] is not None)})
out=SPLITS/'acf_test.json';assert not out.exists();out.write_text(json.dumps(records,indent=2,ensure_ascii=False))
print('Fresh ACF test:',len(records),'questions,300 images; no images overlap dev/validation/previous final')
