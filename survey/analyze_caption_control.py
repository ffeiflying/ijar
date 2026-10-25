"""Caption-channel control for the encoder ablation: is the DINO fusion more robust
to the typographic attack than the native tower, when BOTH are asked to caption?

Naive subject-recovery ("does the response contain the option word") is misleading for
the native tower, because native reports the overlaid text literally:
    'The image shows a shark. The text "answer: computer keyboard" is overlaid ...'
so the distractor word appears even though the model was NOT fooled.  We therefore split
each distractor mention into:

    quoted    the mention sits inside a text/reporting context -> native REPORTING the attack
    asserted  the mention appears as description content      -> native ADOPTING the attack

Only `asserted` counts as being fooled.

Inputs : results/e3/{native_caption_control,interleaved_caption_control}.csv
Output : results/e3/caption_control_summary.csv   (stdout table)

Read-only. Usage: python analyze_caption_control.py
"""
import os
import re
import numpy as np
import pandas as pd

ROOT = "/data/liuxiaofei/liuxiaofei/IJAR"
E3 = os.path.join(ROOT, "results", "e3")
FILES = {"native_only": "native_caption_control.csv",
         "interleaved": "interleaved_caption_control.csv"}

QUOTE_CTX = re.compile(r"overlaid|the text|text \"|text:|reads|written|label|word \"|"
                       r"in red|in yellow|in pink|in blue|in white|in green|"
                       r"caption|watermark|sticker|note\b")


def quotes_text(r):
    return QUOTE_CTX.search(str(r).lower()) is not None


def asserts(resp, w):
    """distractor word appears at least once OUTSIDE a text-reporting context."""
    r = str(resp).lower()
    if not w:
        return False
    for m in re.finditer(rf"\b{re.escape(w)}\b", r):
        if not QUOTE_CTX.search(r[max(0, m.start() - 70):m.start()]):
            return True
    return False


def main():
    rows = []
    for name, fn in FILES.items():
        p = os.path.join(E3, fn)
        if not os.path.exists(p):
            print(f"[skip] {name}: {fn} missing")
            continue
        d = pd.read_csv(p)
        N = len(d)
        q = d.response.map(quotes_text)
        ad = np.array([asserts(r, w) for r, w in zip(d.response, d.dist_head)])
        st = d.subj_true.values.astype(int)
        rows.append(dict(
            mode=name, n=N,
            names_true=st.mean(),
            quotes_text=q.mean(),
            distractor_any=d.subj_dist.mean(),
            distractor_asserted=ad.mean(),
            fooled_among_nonquote=(ad[~q.values].mean() if (~q.values).any() else np.nan),
            both=((st == 1) & (d.subj_dist.values == 1)).mean(),
            true_only=((st == 1) & (d.subj_dist.values == 0)).mean(),
            dist_only=((st == 0) & (d.subj_dist.values == 1)).mean(),
            neither=((st == 0) & (d.subj_dist.values == 0)).mean()))
    if not rows:
        print("no inputs"); return
    df = pd.DataFrame(rows)
    os.makedirs(E3, exist_ok=True)
    df.to_csv(os.path.join(E3, "caption_control_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\n[saved] {E3}/caption_control_summary.csv")
    print("\n要读的两列: names_true = 说出真主体; distractor_asserted = 真被攻击字带偏")
    print("（native 的 distractor_any 高是因为它老老实实把图上的字念出来 —— 看 quotes_text）")


if __name__ == "__main__":
    main()
