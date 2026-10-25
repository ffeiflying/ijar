# Single-forward internal evidential modality-conflict detection

2026-10-25 user explicitly permits method/layer/frame/mass-assignment exploration. Evidence theory remains core; model-internal detection, one forward, no answer sampling. Preserve all other work's results; new artifacts ONLY inside this task. Existing environment unchanged. Prior block0-only constraint superseded for this task.

1. complete: implement multi-layer frozen semantic readout and evidence decomposition; compare0,8,16,20,24,27,31 in one prefill. Fixed image-disjoint dev200 and validation400.
2. complete: smoke numerical reconstruction and exact small-frame mass verification; inspect semantic recovery.
3. complete: evaluate all prespecified configurations on dev only; freeze one winner per evidence family and layer before accessing validation scores.
4. complete: evaluate frozen winners on validation; paired image bootstrap CIs and image-only/question-only controls. Report evidence methods separately from plain matching baselines.
5. complete: current entropy-discounted groundingCF failed generalization (dev0.700/val0.569); task15 tested supervised calibration, task16 tested background-referenced simple-support evidence and a fresh confirmation. No novelty claim or fabricated derivation.

Candidate frame: fixed lexical token hypotheses from development questions without labels/answers/key_component; these are mutually exclusive alternative readout labels, not claims that world objects are globally exclusive. Limited lexical approximation, not established semantic equivalence classes. All hypotheses and evidence scales frozen from dev. Main exact evidence decomposition uses normalized visual residual features and fixed lexical unembedding rows. Attention selects/weights patches; patch mixtures do not assume independence. Text–visual singleton/ignorance conflict is separately identified as a grounding proxy, not proof of object absence.
