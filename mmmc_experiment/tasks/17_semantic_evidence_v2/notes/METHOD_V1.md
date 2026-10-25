# Frozen method v1: background-referenced internal semantic conflict

Status2026-10-25: empirical baseline, not a verified novel contribution. Immutable original implementation/results: tasks/16_reference_semantic_evidence. Existing internvl environment, local InternVL2.5-8B snapshot e9e4c0dc1db56bfab10458671519b7fa3dd29463.

## Inputs and internal readout

Original image+question, one448px tile. One vision encoding and one language prefill, no answer generation. Frozen configuration block24 (zero-based), post-attention visual residuals before FFN. For each retained English question subword t, mean actual causal attention over heads to image patches is conditioned on visual-token mass, yielding a_tj. Read p_j(v)=softmax(W_U RMSNorm(h_j))[v] at image token j, and p_img(t)=sum_j a_tj p_j(t). Uses frozen unembedding as internal semantic lens, not final-answer uncertainty.

Reference q_l(v)=average p_j(v) across patches and100 unique dev images. No class labels/answers in reference estimation. Reference and model hashes frozen before validation/confirmation. Lexical selection in task14extract excludes short/nonalphabetic/stop words; several relation words excluded, a known weakness.

## Evidence model

ell=mean_t[log p_img(t)−log q_24(t)]. Length-normalized likelihood-style contrast, approximate word-grounding model, NOT proven object-presence likelihood. Theta={P,notP}, where P is interpreted operationally as current-image semantic support relative to background.

If ell>=0: m_V(P)=1−exp(−ell), m_V(Theta)=exp(−ell), m_V(notP)=0.
If ell<0: m_V(notP)=1−exp(ell), m_V(Theta)=exp(ell), m_V(P)=0.
Text premise asserts P: m_T(P)=1. Exact unnormalized conjunctive conflict kappa=m_V(notP). Stable rank score−log(1−kappa)=max(−ell,0). Ignorance=exp(−abs(ell)). The mass components sum to1. No semantic negation is established simply by low target-token likelihood.

Opposite BetP=kappa+0.5*m_V(Theta) is monotone in−ell. Thus a BetP gain over rawCF is not new information from mass conversion. V1 has only one scalar degree of freedom, cannot independently represent equally net-supported cases with different support/opposition strengths.

## Frozen results

Dev200 questions/100images; iterative validation400/200images. Dev-selected block24CF validation0.7215875. Same-config block0 validation0.579725. Deployment400 score parity max error5.33e-15.

Fresh confirmation seed20261025:300 previously unused images/600 questions, excludes900 prior images. Main configuration chosen BEFORE confirmation, no test tuning. Block24CF AUROC0.7399556 [0.70680,0.77484]; block0CF0.5910333 [0.55976,0.62630]. Same-layer plain matching0.71265; paired CI for CF−matching [−0.00029,0.05585], superiority not established. Unclipped contrast0.7674611. Object199pairsCF0.78961, attribute43pairs0.68983, relation58pairs0.61623.

All600 inputs audited one vision/language forward and zero answer tokens. RawCF/logCF AUROC equal with no saturatedCF=1. Current-hardware median latency0.0963s block24 vs0.0617s block0, excludes loading and offline reference building; no sampling speedup comparison.

## Scientific position

Already established: useful image-conditioned signal and improvement vs block0 on fresh images. Not established: advantage of DST beyond contrast, full semantic contradiction modeling, external-dataset generalization, or ICCV-level novelty. Next methods must test these, not retrofit a claim.
