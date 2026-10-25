"""Single source for task paths and immutable historical artifact relocation."""
import json
from pathlib import Path
EXPERIMENT=Path(__file__).resolve().parent.parent
PROJECT=EXPERIMENT.parent
DATA=EXPERIMENT/'shared/data'
SPLITS=DATA/'splits'
CACHE=EXPERIMENT/'shared/cache'
TASK1=EXPERIMENT/'tasks/01_dataset'
TASK2=EXPERIMENT/'tasks/02_layer0'
TASK3=EXPERIMENT/'tasks/03_last_layer'
TASK4=EXPERIMENT/'tasks/04_auroc_improvement'
TASK5=EXPERIMENT/'tasks/05_cf_correction'
def resolve_record_path(value):
    """Historical JSON stays byte-identical; resolve its old absolute paths at runtime."""
    path=Path(value)
    try:relative=path.relative_to(EXPERIMENT) if path.is_absolute() else path
    except ValueError:return path
    entries=json.loads((EXPERIMENT/'shared/relocation_manifest.json').read_text())
    for item in sorted(entries,key=lambda x:len(x['old']),reverse=True):
        try:tail=relative.relative_to(item['old'])
        except ValueError:continue
        return EXPERIMENT/item['new']/tail
    return EXPERIMENT/relative

def result_path(name):
    p=Path(name)
    if p.is_absolute():return resolve_record_path(p)
    matches=[t/'results'/p for t in [TASK2,TASK3,TASK4,TASK5,TASK1] if (t/'results'/p).exists()]
    if len(matches)==1:return matches[0]
    if len(matches)>1:raise ValueError(f'Ambiguous result name: {name}')
    return TASK2/'results'/p
