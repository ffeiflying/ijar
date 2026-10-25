# Improve MMMC AUROC
User goal: try multiple methods to improve AUROC. Existing InternVL environment must remain unchanged; isolate new artifacts.

1. in_progress: Freeze image-disjoint development100 / validation200 / final300 pair splits from official test; audit baseline and CF aggregation alternatives.
2. pending: Compare direct visual-premise verification prompts, resolution, and text-only controls using continuous decision logits; inspect all-layer first-token CF and nondegenerate evidence alternatives.
3. pending: Select on development/validation only; freeze winner before evaluating final300 pairs. Quantify AUROC and paired uncertainty, compare baseline on identical heldout records.
4. pending: Write reproducible results with clear distinction between original CF, modified CF, and alternative detectors. Never present a selected dev result as heldout efficacy.
