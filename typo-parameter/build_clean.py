"""
Build clean reference dataset (no typographic overlay)
-------------------------------------------------------
Uses STA original images directly (same source as typo datasets).
No text overlay. Correct label-image pairing guaranteed.

Output: IJAR/typo-parameter/clean/
"""

import sys
import os
import csv
from PIL import Image
from tqdm import tqdm

IJAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STA_CSV = os.path.join(IJAR_DIR, "datasets", "STA", "sta_mini.csv")
STA_IMG_DIR = os.path.join(IJAR_DIR, "datasets", "STA", "images")
OUTPUT_DIR = os.path.join(IJAR_DIR, "typo-parameter", "clean")

import re

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

entries = []
with open(STA_CSV) as f:
    for row in csv.DictReader(f):
        idx = row["index"]
        question = row["question"]
        m_a = re.search(r'\(a\)\s*(.+?)\s*\(b\)', question)
        m_b = re.search(r'\(b\)\s*(.+?)$', question)
        correct_class = m_a.group(1).strip() if m_a else ""
        wrong_class = m_b.group(1).strip() if m_b else ""
        entries.append((idx, correct_class, wrong_class))

print(f"Loaded {len(entries)} entries from STA")

with open(os.path.join(OUTPUT_DIR, "data.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "question", "answer"])
    for idx, correct_class, wrong_class in tqdm(entries, desc="Building clean"):
        img_path = os.path.join(STA_IMG_DIR, idx)
        img = Image.open(img_path).convert("RGB")
        img.save(os.path.join(OUTPUT_DIR, "images", idx), quality=95)

        question = (
            f"this photo is a ' '? Only simply output the corresponding "
            f"letter of the option (such as a or b or c or d) do not output "
            f"the specific content of the option   "
            f"(a) {correct_class} (b) {wrong_class}"
        )
        writer.writerow([idx, question, "a"])

print(f"Done: {len(entries)} images -> {OUTPUT_DIR}/images")
print(f"CSV: {OUTPUT_DIR}/data.csv")
