# Block0 semantic evidence feasibility

User resumed goal on 2026-10-25. Keep block0, existing internvl environment, old code/results unchanged. Use frozen unembedding ONLY to interpret intermediate visual states, not final answer uncertainty. No answer sampling.

1. in_progress: implement one-prefill semantic lens diagnostic and inspect dev examples.
2. pending: measure whether attended visual states recover interpretable concepts at block0; compare input, post-attention residual, and block0 output as position controls (all block0).
3. pending: if semantic recovery works, define shared mutually exclusive hypotheses and calibrated masses, preserving ignorance; verify DS calculations and report ACF against lens-only baseline on held-out validation.
4. pending: if uncalibrated lens fails, document failure and evaluate a lightweight offline semantic calibration within block0; do not silently switch layers or invent negative evidence.

No gold answers, labels, conflict type or key_component enter feature extraction or hypothesis construction. Metadata may be used only after extraction for evaluation. No test-set tuning. Different patches/heads are correlated sources, not independent semantic hypotheses.
