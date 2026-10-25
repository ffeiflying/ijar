"""Read-only snapshot of v1 code/configuration/results with SHA256 manifest."""
import hashlib
import json
import zipfile
from pathlib import Path
TASK=Path(__file__).resolve().parent.parent
EXPERIMENT=TASK.parents[1]
PROJECT=EXPERIMENT.parent
SOURCE=TASK.parent/'16_reference_semantic_evidence'
dest=TASK/'results/v1_archive'
dest.mkdir(parents=True,exist_ok=True)
archive=dest/'method_v1.zip'
assert not archive.exists(),'Do not overwrite frozen archive'
files=list((SOURCE/'code').glob('*.py'))
files += [SOURCE/'README.md',SOURCE/'results/background.npz',SOURCE/'results/config.json',SOURCE/'results/frozen_dev_selection.json',SOURCE/'results/validation_metrics.json',SOURCE/'results/deployment_verification.json']
files += list((SOURCE/'results/confirmation').glob('*.json'))
files += list((SOURCE/'results/confirmation').glob('*.jsonl'))
files += [SOURCE/'results/confirmation/RESULTS.md',SOURCE/'results/confirmation/roc.png']
files += [TASK.parent/'14_multilayer_semantic_evidence/code/extract.py',TASK.parent/'02_layer0/code/run_internvl.py',EXPERIMENT/'shared/layout.py',PROJECT/'layerauroc/run_internvl.py']
files += list((SOURCE/'notes').glob('*.md'))
manifest=[]
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in sorted(set(files)):
        payload=path.read_bytes()
        relative=str(path.relative_to(PROJECT))
        z.writestr(relative,payload)
        manifest.append({'path':relative,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
report={'source':'v1 block24 background-referenced semanticCF, frozen after first fresh confirmation','source_files':manifest,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,'checkpoint':'InternVL2_5-8B snapshot e9e4c0dc1db56bfab10458671519b7fa3dd29463, not copied','environment':'/data/liuxiaofei/miniconda3/envs/internvl, unchanged','test_images':'original immutable source data referenced by confirmation_records; payloads not duplicated','old_results_mutated':False}
(dest/'manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'files':len(manifest),'archive_bytes':report['archive_bytes'],'sha256':report['archive_sha256']},indent=2))
