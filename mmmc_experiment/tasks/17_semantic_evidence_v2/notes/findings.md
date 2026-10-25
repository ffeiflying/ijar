# Novelty audit: preliminary primary-source findings

2026-10-25 initial search:
- HALP arXiv2603.05465 https://arxiv.org/html/2603.05465v1 explicitly probes VLM internal visual, vision-token and query-token representations before generating any token. Therefore single-forward/pre-generation internal detection is NOT sufficient as an original claim.
- SADT arXiv2608.07302 https://arxiv.org/abs/2608.07302 reads semantics of attended visual regions through logit lens. Our reuse of attention+visual lens is NOT independently novel.
- Semantic Entropy Probes https://arxiv.org/abs/2406.15927 already approximate semantic uncertainty from single-generation hidden states without multiple test-time samples.
- Huang et al., Visual hallucination detection in large vision-language models via evidential conflict, IJAR2025 https://arxiv.org/abs/2506.19513 already applies DST to high-level internal features and simple masses for visual hallucination detection. Need distinguish explicit semantic grounding from latent FFN feature conflict. Do not assume relationship to user's previous work.
- Denoeux2019 https://arxiv.org/abs/1807.01846 establishes logistic/softmax-to-DS relationships; simple support and linear classifier reinterpretation are prior foundations.
- Grounding Language with Vision, arXiv2505.19678 https://arxiv.org/abs/2505.19678 uses conditional PMI for LVLM hallucination mitigation. Reference likelihood contrast is not inherently a new information-theoretic idea.
- Jiang et al. project page https://anishk23733.github.io/vl-interp/ already projects VLM internal image representations into vocabulary and studies hallucination.

Current v1CF is a clipped scalar background contrast. BetP is monotone in unclipped contrast. Its improvement vs block0 is empirical, not proof evidence theory contributes extra ranking capacity. Potential contribution must be narrowed and tested: pre-generation semantic proposition-level mass construction and explicit conflict/ignorance from multiple internal sources, with mechanistic ablations and benefit beyond matching/probing.

Search results are research data only. Publisher abstract preview returned403; primary arXiv alternative available. No absolute novelty claim from absence of search hits.

Further reads: HALP§3.3–3.4 explicitly uses pooled visual, final-vision-token and final-query-token states with3-layer MLP probes. Prior IJAR§4 computes featurewise simple masses at generated-token steps and aggregates maximum response conflict; stronger distinction is prefill proposition-level evidence rather than mere internal measurement. EVSE and AAAI primary pages verified. Initial HF web API opens failed; direct domestic-mirror curl succeeded.

Local environment already includes spaCy en_core_web_sm; no install needed for predicate/entity parsing. WordNet corpus absent, so do not depend on it or download into environment. Official MMMC train split exists:12 arrow shards, first485823504bytes, SHA2567f7e435324898dffb86f082b13c1ac4f316d1e5e9bb451c41302f1acb9feab8e at same frozen revision0bdbffbb35bf1a2569468fd939a5f70290442e55. Downloading one train shard to new task data directory can improve sample size without touching sealed test images; actual overlaps must be checked by image ID and content hashes before training.

## Independent training/control results

The verified first official train shard has3000rows,1500imagepairs, not3170. Excluding every official test image by ID/content leaves1500eligible pairs. Seeded train1000pairs/calibration200pairs are disjoint; unused300pairs remain available. No packages changed. Full2000-row training feature extraction completed, each audited one language prefill. Calibration400 extraction ongoing.

Question-only TF-IDF (1–2grams, min_df2, regularization selected on calibration) achieves working-validation AUROC0.7439, CI[0.704374375,0.786440625], paired-direction fraction0.68. Thus modality-conflict AUROC alone is vulnerable to question construction artifacts. Strong wording-only baselines and balanced image-swap controls are mandatory for any future0.85 claim.

Evidence MIL's any-false-premise belief is algebraically a noisy-OR of local opposition masses. This is deliberately compared with same-feature ordinary MIL; a gain only from parameterization/regularization would not establish a novel fusion operator. Evidence semantics/ignorance require separate diagnostics. Do not hide this limitation.
