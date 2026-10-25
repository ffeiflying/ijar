# Calibrated internal evidence fusion

User allows methods/frames/mass changes, evidence-theoretic basis, single-forward internal features, no sampling, preserve old results. This is a supervised calibration candidate, clearly distinct from unsupervised raw CF.

1. complete: use task14 frozen dev-selected internal features as read-only input; fit layer-level Gaussian/LDA evidence likelihood ratios with image-grouped dev cross-validation.
2. complete: define binary frame {normal, modality conflict}; convert per-layer log likelihood ratios into simple-support BPAs, optionally reliability-discount using dev OOF Brier score. Fuse exactly with DS; retain Bel(conflict), BetP(conflict), ignorance and conflict separately.
3. complete: freeze calibrators, feature lists and detection statistic using dev only before reading any validation metrics. Compare plain LR on exactly the same inputs.
4. complete: evaluate image-disjoint validation400; selectedBetP0.686, selectedCF0.488. Keep as negative/calibration baseline, not final method or novel contribution.

Rationale: task14 dev matching0.754 exceeds heuristic groundingCF0.700, suggesting mass calibration needs testing. This task is an empirical calibration baseline for the research direction, not itself proof of an ICCV contribution. The binary frame is task status, not token outputs or an A/B generation prompt. Calibration is offline; inference reads features from one InternVL prefill and applies small linear models.
