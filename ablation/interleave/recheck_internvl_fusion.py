"""Independent paired audit of InternVL native + DINO residual fusion.

No old results are overwritten. Run in the internvl environment, offline.
Caches both towers once; uses the normal chat/generate path with cached features.
Controls: zero gate, wrong-image DINO, spatially averaged DINO, swapped options.
ACF uses live L22 weights and reports prefill and whole-trajectory separately.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "internvl"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import binomtest

from run_e1_e2_internvl import load_image, resolve, MODEL_ID, MODEL_CACHE
from modeling_internvl_chat_dino import InternVLChatModelDINO
from transformers import AutoTokenizer


def strict_choice(response):
    m = re.fullmatch(r"\s*\(?([abAB])\)?[.,:;)\]]?\s*", response)
    return m.group(1).lower() if m else ""


def swap_question(question):
    m = re.search(r"\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*$", question)
    if not m:
        raise ValueError(question)
    return question[:m.start()] + f"(a) {m.group(2)} (b) {m.group(1)}"


def reference_match(response, question):
    m = re.search(r"\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*$", question)
    target, distractor = m.group(1), m.group(2)
    def mentioned(name):
        return any(re.search(r"(?<!\w)"+re.escape(alias.strip().lower())+r"(?!\w)",response.lower())
                   for alias in name.split(',') if alias.strip())
    a,b = mentioned(target),mentioned(distractor)
    return target,distractor,('both' if a and b else 'target_only' if a else 'distractor_only' if b else 'neither')


def kappa_setup(weight):
    # Closed-form reduction of envidence.EvidenceModel, independently checked
    # against the original implementation on real hidden states at runtime.
    w = weight.detach().float()
    return w - w.mean(dim=1, keepdim=True)


def kappa(weight, states):
    # EvidenceModel expands (J,1) to (J,K), then sums over J. All K
    # hypotheses therefore receive the same positive and negative evidence.
    v = ((weight @ states.float().T) / weight.shape[0]).double()
    pos = v.clamp_min(0).sum(0)
    neg = (-v).clamp_min(0).sum(0)
    count = weight.shape[1]
    support = 1 / (count * torch.exp(pos) - count + 1)
    opposition = 1 / (1 - (1 - torch.exp(-neg)).pow(count))
    return (count * support * torch.expm1(pos) * (1-opposition*torch.exp(-neg))).cpu().numpy()


def metrics(frame):
    y = 1 - frame.correct.to_numpy()
    out = dict(n=len(frame), n_error=int(y.sum()), acc=float(1-y.mean()),
               valid_rate=float(frame.valid.mean()))
    for col in ("acf_decision", "acf_mean", "entropy"):
        s = frame[col].to_numpy()
        order = np.argsort(s, kind="stable")
        risk = np.cumsum(y[order]) / np.arange(1, len(y)+1)
        out[col] = dict(auroc=float(roc_auc_score(y, s)) if len(set(y))>1 else None,
                        ap=float(average_precision_score(y, s)) if y.sum() else None,
                        aurc=float(risk.mean()), risk80=float(risk[max(0, round(.8*len(y))-1)]))
    return out


def summarize(outdir):
    df = pd.read_csv(outdir / "samples.csv", keep_default_na=False)
    if 'reference_category' in df and (df.reference_category!='').any():
        rows=[]
        for (mode,condition),f in df.groupby(['mode','condition']):
            rows.append(dict(mode=mode,condition=condition,n=len(f),
                reference_category_counts=f.reference_category.value_counts().to_dict(),
                note='lexical screening only, NOT adjudicated semantic accuracy; no error-AUROC until human review'))
        (outdir/'summary.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(rows,indent=2),flush=True)
        return
    summary = {}
    for keys, f in df.groupby(["mode", "condition", "swapped"]):
        summary["/".join(map(str, keys))] = metrics(f)
    comparisons = []
    rng = np.random.default_rng(20261016)
    for (condition, swapped), f in df.groupby(["condition", "swapped"]):
        pivot = f.pivot(index="sample_id", columns="mode", values="correct")
        for mode in pivot.columns:
            if mode == "native":
                continue
            pair = pivot[["native", mode]].dropna()
            a, b = pair.native.to_numpy(), pair[mode].to_numpy()
            d = b-a
            boot = d[rng.integers(len(d), size=(2000, len(d)))].mean(1)
            wins, losses = int(((a==0)&(b==1)).sum()), int(((a==1)&(b==0)).sum())
            comparisons.append(dict(condition=condition, swapped=int(swapped), mode=mode,
                                    n=len(d), delta_acc=float(d.mean()), ci=np.quantile(boot,[.025,.975]).tolist(),
                                    corrected=wins, broken=losses,
                                    mcnemar_exact_p=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.))
    result = dict(metrics=summary, paired=comparisons)
    (outdir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--modes", default="native,fused,zero,wrong_image,spatial_mean")
    ap.add_argument("--checkpoint", default=str(ROOT / "results/interleave/internvl_dino_projector_fused.pth"))
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--task", choices=['mcq','freeform'],default='mcq')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(out); return
    if (out / "samples.csv").exists():
        raise FileExistsError("Use a fresh output directory")
    torch.manual_seed(42); np.random.seed(42)
    torch.set_num_threads(8)
    modes = args.modes.split(",")
    metadata = dict(args=vars(args), torch=torch.__version__, seed=42, layer=22,
                    sign=1, max_new_tokens=32 if args.task=='freeform' else 16, selection="all rows, no test-set tuning",
                    checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False, cache_dir=MODEL_CACHE, local_files_only=True)
    model = InternVLChatModelDINO.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, use_flash_attn=True,
        trust_remote_code=True, cache_dir=MODEL_CACHE, local_files_only=True,
        ignore_mismatched_sizes=True).to("cuda:0").eval()
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    model._init_dino("cuda:0")
    ck = torch.load(args.checkpoint, map_location="cpu")
    # Crucial: allocate FP32 BEFORE loading the FP32 checkpoint.
    model.dino_projector.float().load_state_dict(ck["dino_projector"], strict=True)
    with torch.no_grad():
        model.dino_gate.data = model.dino_gate.data.float()
        model.dino_gate.fill_(float(ck["dino_gate"]))
    model.dino_max_rms_ratio = ck.get("fusion_config", {}).get("max_rms_ratio")
    metadata["gate"] = float(model.dino_gate.item())
    print("[loaded] gate=",metadata["gate"], flush=True)
    sta = pd.read_csv(ROOT / "datasets/STA/sta_mini.csv").iloc[:args.limit]
    clean = pd.read_csv(ROOT / "typo-parameter/clean/data.csv").sample(n=500, random_state=42).iloc[:args.limit]
    items = []
    for condition, frame, folder in [("attack", sta, ROOT/"datasets/STA/images"),
                                     ("clean", clean, ROOT/"typo-parameter/clean/images")]:
        for _, row in frame.iterrows():
            items.append(dict(condition=condition, sample_id=str(row["index"]),
                              question=str(row.question).strip(), gt=str(row.answer).strip().lower(),
                              path=resolve(str(folder), str(row["index"]))))
    (out / "manifest.json").write_text(json.dumps(items, indent=2))
    cache = []
    t0 = time.time()
    with torch.inference_mode():
        for i,item in enumerate(items):
            pv = load_image(item["path"]).to("cuda:0",torch.bfloat16)
            native = model.extract_feature(pv)
            residual = model.extract_dino_residual(pv)
            cache.append((native.cpu(), residual.cpu()))
            if (i+1)%50==0:
                print(f"[features] {i+1}/{len(items)} {time.time()-t0:.1f}s",flush=True)
    # Assert cache bypass preserves exactly the image features used by generate.
    original_extract = model.extract_feature
    original_residual = model.extract_dino_residual
    current = {}
    model.extract_feature = types.MethodType(lambda self,pv: current["native"], model)
    model.extract_dino_residual = types.MethodType(lambda self,pv: current["residual"], model)
    # An actual image forward and the cached-feature path must agree exactly.
    current["native"], current["residual"] = [x.to("cuda:0") for x in cache[0]]
    parity = {}
    for mode in ("native_only", "fused"):
        model.set_dino_mode(mode)
        pv = load_image(items[0]["path"]).to("cuda:0", torch.bfloat16)
        with torch.inference_mode():
            cached = model.chat(tok,pv,items[0]["question"],dict(max_new_tokens=16,do_sample=False))
            model.extract_feature, model.extract_dino_residual = original_extract, original_residual
            direct = model.chat(tok,pv,items[0]["question"],dict(max_new_tokens=16,do_sample=False))
        assert cached[0] == direct[0], (mode,cached[0],direct[0])
        difference = float((cached[1][0][0]-direct[1][0][0]).abs().max())
        assert difference == 0, (mode,difference)
        parity[mode] = dict(response=direct[0], max_probability_difference=difference)
        model.extract_feature = types.MethodType(lambda self,pv: current["native"], model)
        model.extract_dino_residual = types.MethodType(lambda self,pv: current["residual"], model)
    metadata["cached_direct_parity"] = parity
    states = []
    down = model.language_model.model.layers[22].feed_forward.w2
    weight = kappa_setup(down.weight)
    handle = down.register_forward_pre_hook(lambda m,a: states.append(a[0][0,-1].detach()))
    dummy = torch.zeros(1,3,448,448,device="cuda:0",dtype=torch.bfloat16)
    gate = metadata["gate"]
    cols = ["sample_id","condition","swapped","mode","gt","response","choice","valid","correct",
            "n_forward","acf_decision","acf_mean","entropy","max_prob","residual_native_rms_ratio","elapsed_ms",
            "reference_target","reference_distractor","reference_category"]
    with open(out / "samples.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols); writer.writeheader()
        for i,item in enumerate(items):
            nat,res = [x.to("cuda:0") for x in cache[i]]
            # Different image within the same condition; cyclic permutation.
            offset = args.limit if item["condition"]=="clean" else 0
            wrong_index = offset + ((i-offset+1)%len(sta if offset==0 else clean))
            wrong = cache[wrong_index][1].to("cuda:0")
            current["native"] = nat
            for swapped in ((0,) if args.task=='freeform' else (0,1)):
                q = ('What is the main object in this image? Answer with a concise noun phrase.' if args.task=='freeform'
                     else swap_question(item["question"]) if swapped else item["question"])
                gt = ({"a":"b","b":"a"}[item["gt"]] if swapped else item["gt"])
                for mode in modes:
                    model.set_dino_mode("native_only" if mode=="native" else "fused")
                    current["residual"] = (wrong if mode=="wrong_image" else
                                           res.mean(1,keepdim=True).expand_as(res) if mode=="spatial_mean" else res)
                    with torch.no_grad(): model.dino_gate.fill_(0 if mode=="zero" else gate)
                    states.clear(); start=time.time()
                    with torch.inference_mode():
                        response, probs, _, _ = model.chat(tok,dummy,q,dict(max_new_tokens=metadata['max_new_tokens'],do_sample=False))
                        values = kappa(weight,torch.stack(states))
                    if i==0 and swapped==0 and mode=="native":
                        from envidence import EvidenceModel
                        evidence = EvidenceModel(down.weight.detach().float())
                        original=[]
                        for state in states:
                            evidence.get_evidence_weights(state.reshape(-1,1))
                            original.append(float(evidence.get_evidence_conflict().item()))
                        err=float(np.max(np.abs(values-original)))
                        metadata["acf_closed_form_max_error"]=err
                        print("[ACF parity]",err,flush=True)
                        assert err<1e-6, (values,original)
                    p = probs[0][0].float()
                    p = p / p.sum()
                    choice = strict_choice(response)
                    target,distractor,category=reference_match(response,item['question']) if args.task=='freeform' else ('','','')
                    with torch.no_grad():
                        delta = model.fuse_dino_residual(nat,current['residual']).float()-nat.float()
                        ratio = float(delta.square().mean().sqrt()/nat.float().square().mean().sqrt()) if mode!='native' else 0.
                    writer.writerow(dict(sample_id=item["sample_id"],condition=item["condition"],swapped=swapped,mode=mode,
                        gt=gt,response=response,choice=choice,valid=int(bool(choice)),correct=-1 if args.task=='freeform' else int(choice==gt),n_forward=len(states),
                        acf_decision=float(values[0]),acf_mean=float(values.mean()),
                        entropy=float(-(p*p.clamp_min(1e-30).log()).sum()),max_prob=float(p.max()),
                        residual_native_rms_ratio=ratio,elapsed_ms=1000*(time.time()-start),
                        reference_target=target,reference_distractor=distractor,reference_category=category))
                f.flush()
            if (i+1)%10==0:
                print(f"[eval] {i+1}/{len(items)} {time.time()-t0:.0f}s",flush=True)
    handle.remove()
    metadata["elapsed_seconds"]=time.time()-t0
    (out / "metadata.json").write_text(json.dumps(metadata,indent=2))
    summarize(out)


if __name__ == "__main__":
    main()
