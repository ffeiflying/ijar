# MMMC modality-conflict detection (isolated experiment)

Current priority: InternVL2.5-8B in the existing `internvl` environment. Do not install, upgrade, remove packages, or modify old experiment outputs.

Official dataset: `ustc-zhangzm/MMMC`, revision `0bdbffbb35bf1a2569468fd939a5f70290442e55`. `data/MMMC/test` contains 4000 records with verified official LFS SHA256 checksums. The requested `/data/datasets` location is pending administrator filesystem permission; data is temporarily staged here.

MMMC has genuine paired normal/conflicting questions per image. Null `conflict_type` is normal; object/attribute/relation are conflict. `prepare_pilot.py` verifies both complete shards and exports 100 image pairs (200 questions) selected by deterministic seed42 group shuffle. See `pilot_manifest.json` for provenance and label counts.

`run_internvl.py` uses only local model files. It captures input to decoder block index0 down projection (`language_model.model.layers.0.feed_forward.w2`) with a pre-hook. It verifies the existing evidence matrix agrees with model weights and uses existing `envidence.EvidenceModel`. Scores average per generation forward step (prefill last position predicts first response token), with deterministic decoding and max64 new tokens, matching the earlier single448px image convention. This tests binary input conflict detection; it does not grade answer correctness.

Commands (GPU execution needs access outside sandbox):

```sh
/data/liuxiaofei/miniconda3/envs/internvl/bin/python -s mmmc_experiment/prepare_pilot.py
/data/liuxiaofei/miniconda3/envs/internvl/bin/python -s mmmc_experiment/run_internvl.py --limit 2 --output internvl_smoke.jsonl
/data/liuxiaofei/miniconda3/envs/internvl/bin/python -s mmmc_experiment/run_internvl.py
/data/liuxiaofei/miniconda3/envs/internvl/bin/python -s mmmc_experiment/summarize.py
```

The summary fixes high ACF as the positive direction, and reports AUROC, AUPRC, TPR at FPR<=5%, image-paired differences, per-conflict-type AUROC against matched controls, and 1000 image-pair bootstrap confidence intervals. No threshold is fitted to the pilot.

Qwen/MoF/DeepSeek are still pending after the user's instruction to prioritize InternVL. Full train download is also pending; evaluation uses official test only.
