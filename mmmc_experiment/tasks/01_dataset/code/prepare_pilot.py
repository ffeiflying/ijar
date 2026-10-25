"""Build deterministic paired pilot exclusively from checksum-verified test shards."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

import pyarrow as pa
ROOT = EXPERIMENT
manifest = json.loads((TASK1 / 'results/test_manifest.json').read_text())
rows = []
provenance = []
for entry in manifest:
    if not entry['path'].endswith('.arrow'):
        continue
    p = DATA / 'MMMC' / entry['path']
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert p.stat().st_size == entry['size'] and digest == entry['lfs']['oid'], p
    provenance.append({'path':str(p), 'sha256':digest})
    rows.extend(pa.ipc.open_stream(str(p)).read_all().to_pylist())
assert len(rows) == 4000, len(rows)
groups = defaultdict(list)
for i, row in enumerate(rows):
    row['row_id'] = i
    groups[row['image_id']].append(row)
pairs = []
for image_id, group in groups.items():
    clean = [r for r in group if r['conflict_type'] is None]
    conflict = [r for r in group if r['conflict_type'] is not None]
    if clean and conflict:
        pairs.append((image_id, clean[0], conflict[0]))
random.Random(42).shuffle(pairs)
selected = pairs[:100]
image_dir = DATA / 'pilot_images'
image_dir.mkdir(exist_ok=True)
records = []
for image_id, clean, conflict in selected:
    # Ensure paired questions refer to exactly the same image bytes.
    assert clean['image']['bytes'] == conflict['image']['bytes']
    path = image_dir / f'{image_id}.img'
    path.write_bytes(clean['image']['bytes'])
    for row in [clean, conflict]:
        records.append({**{k:v for k,v in row.items() if k != 'image'}, 'image_path':str(path), 'label':int(row['conflict_type'] is not None)})
(DATA / 'pilot.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
report = {'revision':'0bdbffbb35bf1a2569468fd939a5f70290442e55', 'test_rows':len(rows), 'test_labels':dict(Counter(str(r['conflict_type']) for r in rows)), 'eligible_image_pairs':len(pairs), 'seed':42, 'pilot_rows':len(records), 'pilot_conflict_types':dict(Counter(str(r['conflict_type']) for r in records)), 'selection':'shuffle image groups with seed42; first100; first clean/conflict per image', 'verified_shards':provenance}
(DATA / 'pilot_manifest.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
