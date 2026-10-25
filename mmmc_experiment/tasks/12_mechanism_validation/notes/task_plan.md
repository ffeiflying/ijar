# Step2: mechanism validity before AUROC optimization
User chose step2 after requesting scope discussion before new experiments. Interpret as verifying whether CF depends on image-question semantic matching. Do not resume parameter/layer search.

Fixed boundaries: InternVL, block0, original question/image, post-attention FFN downprojection input as provisional baseline, canonical evidence_v2. Old envidence.py retained. Same score orientation and fixed preprocessing across controls. No environment changes.

1. complete: metadata-only feasibility audit. Official4000-row test contains94 normalized-question groups with both official labels, spanning265 normal and210 conflict records. Official labels are not manually verified truth.
2. in_progress: lock mechanism protocol: same-image paired questions; same-question supported/contradictory images; image-permutation control. Randomly mismatched images must NOT be assigned fabricated gold labels.
3. pending: inspect existing baseline scores for preliminary matched-question coverage; do not confuse small available subset with full mechanism evidence.
4. pending: after protocol is settled, score exact matched controls and report within-question/image differences and uncertainty before tuning AUROC.

Interpretation: positive paired evidence supports the mechanism but does not establish causality or guarantee higher detectionAUROC. If original first-layer CF lacks image-conditioned effects, do not explain its AUC as modality-conflict detection.
