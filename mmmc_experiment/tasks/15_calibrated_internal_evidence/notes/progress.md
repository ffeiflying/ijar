# Progress

2026-10-25: task created while task14 validation extraction is running, before any validation performance has been evaluated. Offline supervised evidence calibration and same-feature LR baseline will use only dev200. No extra model inference, no environment changes, no old results modified.

Frozen dev OOF selection: semantic_all/brier_skill/T1/betp0.7891; bestCF source-only0.50035. Independent30-case DS source enumeration error2.22e-16. Validation: selectedBetP0.6859875 (CI0.64648–0.72353), selectedCF0.4882375 (CI0.43650–0.54423). Same-input LR semantic_all0.67636, matching-onlyLR0.73899. No evidence-theoretic performance gain established, supervised calibration not a good final method yet. Keep CF and belief scores separate.
