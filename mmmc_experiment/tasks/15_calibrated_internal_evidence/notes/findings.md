# Theory and limits

Fit equal-prior class-conditional multivariate Gaussian experts with common shrunk covariance at each selected layer. Their discriminant estimates a log likelihood ratio ell=log p(x|conflict)/p(x|normal) under the fitted Gaussian model. This is a modeling assumption, not exact true-data likelihood.

Positive ell gives simple support on {conflict}, negative ell on {normal}, singleton mass1−exp(−abs(ell)), Theta mass exp(−abs(ell)). Before discount, plausibility likelihood ratio equals exp(ell). Optional reliability r from image-grouped dev OOF Brier skill discounts committed mass, returning the remainder to Theta. Correlated layers violate naive independence; explicit discount and baseline comparison required.

Fuse BPAs on the same binary frame exactly. Aggregate positive and negative simple-support weights ep/en. Unnormalized empty-set mass=(1−exp(−ep))(1−exp(−en)). Normalized singleton masses and Theta computed stably. Bel/BetP and raw CF are different statistics and must not be conflated. In general CF measures disagreement, not automatically task abnormality.

Supervised calibration must be labeled as such; do not compare it as a zero-shot method. Basic Gaussian/LR-to-DS constructions are established, not a standalone novelty claim. All artifact reads from task14 are read-only.
