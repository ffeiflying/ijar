# Training-free single-prefill modality conflict research

Constraints: frozen InternVL environment/model; no fitted heads, supervised calibration, fine-tuning or sampling; evidence theory central. Preserve prior tasks. Target~0.85 remains unmet. Supervised task17 scores are excluded. Do not inspect fresh task17 confirmation scores/labels for selection.

1. in_progress: expand primary-source research: training-free internal grounding, attention disagreement, semantic alternatives, random-set/consonant masses, dependence-aware fusion.
2. pending: document practical candidates and exact single-pass/semantic assumptions, reject methods requiring reruns or training.
3. pending: implement inexpensive candidates on existing dev features; freeze before working validation. Report evidence-vs-same-feature baselines honestly.
4. pending: record results and next actionable direction; no novelty or performance fabrication.

Update: expanded primary-source review complete (including closest training-free DST CAD, multiple modality branches). Spatial candidate rejected by frozen validation0.5484. Raw AV semantic source candidate dev0.6316, validation abandoned before scoring. Typed patch semantic-sets candidate devCF0.76085 frozen; validation extraction complete and scoring running. Added fourth candidate: prequential local visual counterevidence, reading the predecessor of each input word (avoids reading a word from a state that already contains it). Passive image/nonimage AV decomposition with no additional transformer call. Evaluate whether visual contribution actually suppresses the claimed input token instead of declaring low absolute likelihood a contradiction. No guarantee this local contrast equals a full causal modality ablation.
