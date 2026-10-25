# Training-free candidates and exact eligibility

All candidates use a frozen model and passive readouts in one prefill. Extra projections/aggregation count as inference overhead. No sampling, learned detector or calibration coefficients. Choosing a formula/layer on dev is exploratory method selection, not fitting, and still risks selection bias; validation and eventual fresh confirmation are separate.

| Direction | Primary foundation | Eligible adaptation | Important boundary |
|---|---|---|---|
| Semantic grounding of attended regions | [SADT](https://arxiv.org/abs/2608.07302) | internal semantic affinities at input premise locations | original object hallucination detector concerns generated tokens; adaptation unproven |
| Visual-head source analysis | [VHR](https://arxiv.org/abs/2412.13949) | split actual attention AV into image-token/nonimage-token summands | original image-removed VHD needs another forward; our local split is not a full causal image/no-image contrast |
| Attention vs FFN trajectories | [FADE](https://arxiv.org/abs/2606.29431), [CALRD](https://ijcai.org/proceedings/2026/153) | passive pre/post attention/FFN semantic trajectories; interpret direction | FFN is not pure text and attention not pure vision; no ground-truth answer directions allowed |
| Sink-aware evidence reliability | [SDPR](https://arxiv.org/abs/2607.16841) | target-specific attention relative to common sink pattern; discount uncertain sources | sink filtering alone fails cached experiment; do not intervene in model for this detector |
| Spectral head reliability | [FLASH](https://arxiv.org/abs/2609.37263) | passive spectral statistics for source selection/discount | spectral complexity is not semantic contradiction; original mitigation not a ready detector |
| Set-valued semantic alternatives | [Dubois/Prade/Smets](https://ord.pwr.edu.pl/Issues/2003/vol13/p200342) | inverse-pignistic nested random sets from frozen semantic affinity distributions | conditional dictionary readouts are not calibrated real-world labels; open-set coverage must become unknown |
| Dependent-source fusion | [Denoeux2008](https://doi.org/10.1016/j.artint.2007.05.008) | avoid treating correlated heads/layers as independent; compare idempotent/dependence assumptions | a fancy fusion rule cannot repair meaningless hypothesis definitions |
| Claim-level image support | [EGVCA](https://arxiv.org/abs/2610.17706) | emphasize premise-level support and abstention | original visual-null comparison violates our single-forward constraint |
| Pure attention geometry | [Attention Never Lie](https://openaccess.thecvf.com/content/CVPR2026F/html/Zhao_Attention_Never_Lie_Visual_Attention_Defocus_Reveals_and_Rectifies_Hallucinations_CVPRF_2026_paper.html) | entropy, covariance, spectrum as fixed reliability cues | original MLP/window detector is supervised; no complete reuse |
| Hidden/attention probes | [HALP](https://arxiv.org/abs/2603.05465), [FaithSCAN](https://arxiv.org/abs/2601.00269), [Attention divergence](https://arxiv.org/abs/2605.05025) | diagnostics and same-feature controls only | trained aggregation excluded from intended method |

## Current preferred construction to test

Parse input premises into typed atoms without answers/labels. For each atom read image-token attention contribution and nonimage-token contribution from the actual operation. On a common typed dictionary plus target description, frozen unembedding gives label affinities. Word alternatives refer to competing local descriptions, not mutually exclusive existence of all objects in the whole image. Preserve uncovered vocabulary mass as unknown; otherwise an irrelevant dictionary could manufacture confidence.

From conditional label probabilities p in decreasing order, define nested focal sets A_i and masses i*(p_i−p_(i+1)). The exact inverse-pignistic mapping preserves the original categorical probabilities while exposing imprecision. Compare visual-vs-input assertion conflict, visual-vs-nonimage internal-view conflict, ignorance and same-feature raw affinity. Source decomposition is local to an already multimodally contextualized state, not complete modality isolation.

## Tested rejection

Spatial random-set conflict between attention support and word-likelihood map was mathematically correct (enumeration error1.11e-16) but frozen validationAUROC0.5484; same-feature matching0.7358. Localization mismatch is not sufficient semantic conflict. Keep this negative result rather than wrap it in a stronger novelty claim.
