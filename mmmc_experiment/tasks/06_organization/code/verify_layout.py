import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[3]
entries=json.loads((R/'shared/relocation_manifest.json').read_text());checked=0
for item in entries:
 p=R/item['new'];assert p.exists(),p
 if item['sha256'] and p.suffix in ['.jsonl','.npz','.npy']:
  assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
  checked+=1
for task in (R/'tasks').iterdir():
 assert all((task/sub).is_dir() for sub in ['code','results','logs','notes']),task
assert not list(R.glob('*.py')) and not list(R.glob('*.jsonl'))
report={'task_folders':len(list((R/'tasks').iterdir())),'all_tasks_have_code_results_logs_notes':True,'raw_artifacts_verified_byte_identical':checked,'root_has_no_loose_scripts_or_results':True}
(R/'tasks/06_organization/results/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
