"""Build a source-only publication copy without touching experiment artifacts."""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    '.git', '.agents', '.aws', '.codex', '.planning', '.superpowers',
    '__pycache__', '.pytest_cache', '.ipynb_checkpoints', '.venv', 'venv',
    'hf_cache', 'cache', 'vendor', 'site-packages', 'data', 'datasets',
    'results', 'logs', 'weights', 'checkpoints', 'images', 'github_export',
}
EXTENSIONS = {'.py', '.sh', '.md', '.rst', '.txt', '.toml', '.yaml', '.yml', '.json', '.ini', '.cfg', '.tex', '.bib'}
NAMES = {'LICENSE', 'NOTICE', 'COPYING', 'Makefile', '.gitignore', '.gitattributes'}
SECRET = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}')

def candidates():
    import os
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not d.startswith('.') and not (Path(base)/d).is_symlink())
        for name in sorted(files):
            path = Path(base)/name
            if path.is_symlink() or name.startswith('.env') or name.startswith('log_') or name.endswith('_metrics.txt'):
                continue
            if path.suffix.lower() not in EXTENSIONS and name not in NAMES:
                continue
            # Attack image directories also contain generated data.csv files;
            # source-only export deliberately excludes every CSV/JSONL/array.
            yield path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'github_export/IJAR')
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT or ROOT.is_relative_to(out):
        parser.error('output must not be the source directory or its ancestor')
    if out.exists():
        parser.error('output already exists; choose a fresh directory')
    selected = []
    problems = []
    for path in candidates():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > 2 * 1024 * 1024:
            problems.append(f'oversized source file: {relative}')
            continue
        try:
            content = path.read_text(encoding='utf-8')
        except (UnicodeError, OSError):
            problems.append(f'unreadable/non-text: {relative}')
            continue
        if SECRET.search(content):
            problems.append(f'possible credential: {relative}')
        selected.append((path, relative))
    if problems:
        raise SystemExit('\n'.join(problems))
    out.mkdir(parents=True)
    for path, relative in selected:
        target = out/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    manifest = {'files': len(selected), 'bytes': sum(p.stat().st_size for p, _ in selected),
                'excluded_directory_names': sorted(EXCLUDED),
                'note': 'Source/documentation only; no model, dataset, generated results or environment. Credential pattern check is heuristic.'}
    (out/'SOURCE_EXPORT.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'output': str(out), **manifest}, indent=2))

if __name__ == '__main__':
    main()
