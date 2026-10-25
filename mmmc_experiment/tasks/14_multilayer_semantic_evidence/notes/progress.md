# Progress

2026-10-25: new task created after user broadened scope. No earlier results edited, moved, removed or overwritten. No CLIP calibration training started. Existing internvl environment and local InternVL model retained.

Completed v1 4-row smoke and v2 4-row smoke, preserving separate directories. Independent30-case checks: matrix identity error8.88e-16, normalized plausibility error3.33e-16, singleton/Theta conflict2.78e-16, pair-averaging2.78e-16. These check math only.

Completed formal dev200, seven layers0/8/16/20/24/27/31, using two shards; GPU1 slow shard was interrupted by exact matching own PID3398942, completed files retained, remaining resumed on GPU0. No other processes interrupted. A temporary shell inspection command had SyntaxError from an unmatched bracket; no files changed by that command.

Frozen336 dev configurations before validation inference. Best evidence candidate L31_attn_T1_grounding_cf_attention_mean devAUROC0.69985; best plain matching baseline L31_attn_nll_attention_mean dev0.7535. Exact sourceDS best per layer reaches dev0.6168 atL16. This suggests semantic support is useful but current evidence mass assignment may lose discrimination; must check validation before claiming improvement.

Validation400 running on GPU0 after freeze. Same environment, all new artifacts inside task14.

Validation400 completed: dev-frozen best groundingCF0.5685125 (CI0.52694–0.60540); best matching baseline0.732725. AtL16 sourceDS0.593875. Image-only controls~0.5; word rarity0.51219 and question length0.526625. Do not describe dev0.700 as validated.

Balanced synthetic mechanism check completed on8 visually inspected development images/8 questions,16 crossed rows. Each image and question appears once per label, so image-only and question-only labels cannot classify. All16 audited exactly one language-model forward. Frozen groundingCF gives higher score for absent object in8/8 same-question pairs, AUROC0.96875; pure matching1.0. Small development-image mechanism check, NOT a new generalization benchmark. SourceDS atblock20 shows8/8 directional changes but only0.8594 pooledAUC. This supports image dependence of internal semantic scores but not full MMMC effectiveness.
