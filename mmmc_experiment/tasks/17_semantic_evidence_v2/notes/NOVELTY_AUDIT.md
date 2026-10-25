# Novelty evidence and falsifiable claims

Checked2026-10-25. This is a bounded primary-source review, not proof no similar paper exists. Current v1 alone does not justify a broad first-ever claim.

| Prior work | Already covered | Difference worth testing |
|---|---|---|
| [HALP](https://arxiv.org/html/2603.05465v1) | internal VLM representations before generating any token, learned probes | explicit semantic mass construction and conflict/ignorance, beyond same-feature probe |
| [SADT](https://arxiv.org/html/2608.07302v1) | logit lens of attended visual regions, checks generated object grounding | input-premise detection during prefill; no generated answer required |
| [Jiang et al.](https://anishk23733.github.io/vl-interp/) | visual-token vocabulary projection and object confidence | proposition/typed relations, multiple-source evidential behavior |
| [Evidential Conflict, IJAR2025](https://arxiv.org/html/2506.19513v1) | internal FFN features, simple masses, conflict across generated response tokens | interpretable proposition hypotheses during prefill rather than latent-channel conflict over a response |
| [Semantic Entropy Probes](https://arxiv.org/abs/2406.15927) | single-generation hidden-state uncertainty without test-time multiple samples | direct internal semantic evidence instead of only predicting a scalar teacher score |
| [EVSE](https://aclanthology.org/2026.eacl-long.334/) | evidence mass over semantics, ignorance for unobserved answers | internal views instead of sampled-answer clustering; cannot reuse unobserved-output ignorance interpretation |
| [Belief Divergence/Conflict](https://ojs.aaai.org/index.php/AAAI/article/view/39727) | evidence-based ambiguity/conflict decomposition | shared proposition frames from one pass, instead of response perturbation ensemble |
| [C-PMI grounding](https://arxiv.org/abs/2505.19678) | mutual-information-based grounding calibration | reference-normalized internal evidence must demonstrate more than a contrast score |
| [Denoeux](https://arxiv.org/abs/1807.01846) | simple supports and classifier-to-DS interpretation | foundational tool, never claim as new |

## What v1 actually proves

On a fresh300-image/600-question confirmation, frozen block24CF0.740 versus block0CF0.591. A small balanced image/question control shows image-conditioned response. No evidence that a monotone mass conversion improves ranking: unclipped contrast0.767; BetP monotone-equivalent. No independent proof of full semantic contradiction or novel theory.

## Proposed narrower research claim (hypothesis, not established)

During one prefill, assign multiple internal semantic views to common proposition-specific frames and preserve independently varying support, opposition and ignorance. Quantify grounded semantic disagreement with dependence-aware fusion, before any answer exists. Test on modality conflicts including attributes and relations.

## Evidence required for a credible contribution

1. Same expected compatibility/net log ratio but different support+opposition must yield distinguishable conflict/ignorance (not a scalar reparameterization).
2. Remove DST fusion while keeping identical features and training data; compare PMI, entropy, head disagreement, LR/MLP probes and evidential scores. Gains or additional calibrated diagnostic value must survive.
3. Keep exact wording fixed while changing image support; balance image-only/wording-only controls. Different objects or correlated heads must not automatically count as logical contradiction.
4. Use explicit object/attribute/relation hypothesis definitions, document unsupported relations as unknown. Evaluate per-type AUROC and mass behavior.
5. Audit one VLM forward, no answer generation, and end-to-end latency including parser/lens/fusion. Count offline reference or supervised training costs separately.
6. Freeze method before new holdout;0.85 is a target, never manufacture it by test selection. Cross-model/dataset confirmation needed before a broad paper claim.

## Current technical weakness to fix

V1 first averages signed log evidence then assigns one simple support; opposite head/atom evidence can cancel into apparent ignorance. A multi-source version can retain two nonnegative accumulators and exact conflict, while keeping the same normalized plausibility. This mathematical distinction is testable and potentially useful; whether it is novel or improves detection remains empirical.
