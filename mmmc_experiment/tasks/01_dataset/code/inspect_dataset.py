"""Inspect Arrow without requiring load_from_disk metadata or decoding images."""
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

import pyarrow as pa
root = TASK1 / "results"
files = sorted((DATA / 'MMMC/test').glob('*.arrow'))
if not files:
    raise RuntimeError('No verified Arrow shards available')
tables = [pa.ipc.open_stream(str(p)).read_all() for p in files]
table = pa.concat_tables(tables)
columns = [c for c in table.column_names if c != 'image']
rows = table.select(columns).to_pylist()
report = {'n_rows':len(rows), 'columns':str(table.schema), 'conflict_type_counts':dict(Counter(r['conflict_type'] for r in rows)), 'examples':rows[:12]}
(root / 'dataset_inspection.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report, indent=2, ensure_ascii=False))
