# Background-referenced semantic evidence

2026-10-25: task14 entropy-discounted grounding CF dev0.700/val0.569 versus internal matching baseline val0.733. Task15 supervised DS belief val0.686, CF0.488. Need revise evidence rather than present these as success. This next method uses a label-free development image reference model instead of entropy reliability.

1. complete: estimate per-layer background lexical distribution from100 unique development images, no labels/answers. Frozen normalized intermediate visual lens, same layers/position as task14. No answer generation.
2. complete: for each question compute mean log likelihood ratio log p_visual(query token)−log p_reference(query token), using task14 meanNLL features and frozen background. This is a relative lexical-grounding statistic, not proof of absence.
3. complete: establish binary approximate grounding frame; convert signed log ratio to simple-support mass, Theta=exp(−abs(ell)). Text premise puts mass on its asserted premise. Exact conflict equals visual opposite mass. Compare unclipped contrast and BetP, no gain claimed from monotone remapping. Fully committed Bayesian sigmoid is ranking-equivalent and need not be separately tuned.
4. complete: dev-frozen layer24/attention/A1 CF validation0.7216. Single-forward detector reproduces400 offline scores, max error5.33e-15. Same-configuration layer0 baseline validation0.5797.
5. complete: freeze and score600 fresh confirmation rows/300 images excluding all prior900 images, no further tuning. CF block24=0.73996 (CI0.70680–0.77484), block0=0.59103. Plain same-layer matching0.71265, unclipped contrast0.76746. Forward count1 verified for all inputs.
6. next research question (not required for this completed experimental round): retain semantic structure beyond word tokens; separate object/attribute/relation evidence; improve conflict versus ignorance without losing signal by clipping. Establish value beyond plain contrast and check related work before novelty claims. Do not tune on the fresh confirmation set.

All outputs isolated in this task; existing results read-only. Offline100-image reference collection has cost; deployed input requires one forward and a lookup, no sampling. A fresh final holdout will be needed after selecting a good method.
