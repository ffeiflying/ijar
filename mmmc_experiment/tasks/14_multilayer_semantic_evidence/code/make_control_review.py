"""Contact sheet for visual verification of a small same-question control."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from layout import SPLITS,resolve_record_path
task=Path(__file__).resolve().parent.parent
dest=task/'results/controls'
dest.mkdir(parents=True,exist_ok=True)
rows=json.loads((SPLITS/'dev.json').read_text())[:34:2]
canvas=Image.new('RGB',(5*256,4*280),'white')
draw=ImageDraw.Draw(canvas)
for i,row in enumerate(rows):
    picture=Image.open(resolve_record_path(row['image_path'])).convert('RGB')
    picture=ImageOps.contain(picture,(250,240))
    x=(i%5)*256
    y=(i//5)*280
    canvas.paste(picture,(x,y))
    draw.text((x+3,y+241),str(row['image_id']),fill='black')
    draw.text((x+3,y+256),row['question'][:36],fill='black')
canvas.save(dest/'image_review.jpg',quality=90)
(dest/'source_images.json').write_text(json.dumps(rows,indent=2))
print(dest/'image_review.jpg')
