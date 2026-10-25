# Rationale and limits

Raw token likelihood contains a lexical-frequency component; compare it with the same model's marginal probability over development images. The reference averages over image patches/images, not query-conditioned attention under alternative images, so this is an approximate background model. It is a lexical grounding Bayes contrast under the chosen two models, not a calibrated object-presence probability.

Given ell=log L_supported/L_background, a simple-support BPA has m(P)=1−exp(−ell),m(Theta)=exp(−ell) when ell>=0; for ell<0, m(not P)=1−exp(ell),m(Theta)=exp(ell). Its singleton versus full-frame plausibility ratio reproduces exp(ell). A text source asserting P conflicts by kappa=m_visual(not P). Length-normalized mean log likelihood is used to reduce query length effects; it is a power likelihood/modeling choice.

When scoring CF this clips the negative log contrast at0, introducing ties; the unclipped contrast baseline must be reported separately. A sigmoid CF with fully committed visual Bayesian masses is only a monotone transformation, not an evidence-theoretic performance gain.

Fresh600 confirmation: CF24=0.7399556 versus same-configurationCF0=0.5910333. RawCF and stablelogCF yield exactly equal AUROC, no saturatedCF=1 in either set. Same-layer plain matching0.71265, with paired95% AUC-difference CI−0.00029 to0.05585, so do not claim established superiority to matching. Unclipped background contrast0.76746 shows CF clipping loses information; visual-opposite BetP=CF+0.5Theta is monotone in the contrast and does not add ranking by itself.

Object conflict is strongest (199pairs,0.78961), attributes43pairs0.68983, relations58pairs0.61623. Current lexical selection removes several spatial/function words, so relation semantics is incompletely represented; query attention remains contextual, but this is not a full predicate hypothesis frame. Future work should retain structured predicates and independent support/opposition signals rather than just explain existing scalar scores. Confirmation set is now frozen for reporting, not future tuning.
