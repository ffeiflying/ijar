"""Download official immutable MMMC snapshot without installing packages."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import EXPERIMENT, PROJECT, DATA, SPLITS, CACHE, TASK1, TASK2, TASK3, TASK4, TASK5, resolve_record_path, result_path

import subprocess

REV = '0bdbffbb35bf1a2569468fd939a5f70290442e55'
parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/data/datasets/MMMC')
parser.add_argument('--endpoint', default='https://hf-mirror.com')
parser.add_argument('--test-only', action='store_true')
ARGS = parser.parse_args()
ROOT = Path(ARGS.root)
def request(url):
    return subprocess.check_output(['curl', '-fL', '--retry', '3', '--max-time', '120', '-sS', url])
def download(entry):
    path = ROOT / entry['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get('lfs', {}).get('oid')
    if path.exists() and path.stat().st_size == entry['size']:
        if not expected or hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            print('verified', entry['path'], flush=True)
            return
    url = f'{ARGS.endpoint}/datasets/ustc-zhangzm/MMMC/resolve/{REV}/{entry["path"]}'
    tmp = path.with_suffix(path.suffix + '.partial')
    for attempt in range(3):
        try:
            subprocess.run(['curl', '-fL', '--http1.1', '--retry', '10', '--retry-all-errors', '--retry-delay', '2', '--connect-timeout', '30', '--max-time', '1800', '-C', '-', '-sS', '-o', str(tmp), url], check=True)
            assert tmp.stat().st_size == entry['size'], path
            if expected:
                assert hashlib.sha256(tmp.read_bytes()).hexdigest() == expected, path
            tmp.replace(path)
            print('downloaded', entry['path'], path.stat().st_size, flush=True)
            return
        except Exception as exc:
            print('retry', entry['path'], repr(exc), flush=True)
            if attempt == 2:
                raise
if __name__ == '__main__':
    entries = []
    subs = ['', '/test'] if ARGS.test_only else ['', '/test', '/train']
    for sub in subs:
        cached = TASK1 / 'results' / (('test' if sub == '/test' else 'root') + '_manifest.json')
        if sub in ['', '/test'] and cached.exists():
            listing = json.loads(cached.read_text())
        else:
            listing = json.loads(request(f'{ARGS.endpoint}/api/datasets/ustc-zhangzm/MMMC/tree/{REV}{sub}'))
        entries.extend(e for e in listing if e['type'] == 'file')
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'download_manifest.json').write_text(json.dumps({'repo':'ustc-zhangzm/MMMC','revision':REV,'files':entries}, indent=2))
    # Fetch test first to enable pilot while training data downloads.
    for split in ['test/', 'other', 'train/']:
        selected = [e for e in entries if (e['path'].startswith(split) if split != 'other' else '/' not in e['path'])]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(download, selected))
