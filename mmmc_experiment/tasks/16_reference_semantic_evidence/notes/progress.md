# Progress

2026-10-25: task initialized after earlier validation failures; no old results modified. Reference distribution fitted without class labels, only original dev images.

Reference100 images complete. Dev selectedCF L24/attention/A1=0.74275; independent contrast winnerL31/attention/A1 dev0.8095. Iterative validationCF0.7215875 (CI0.68216–0.75844). Single-forward deployment400 rows exactly reproduced offline score to5.33e-15. Fixed configuration layer0CF0.579725; AUC24−0 CI0.08750–0.19324. Median detector time0.0959s on current hardware/load, no model loading or reference building included.

Fresh confirmation protocol frozen before predictions: seed20261025,300 image pairs/600rows from1100 unused images, excluding900 from all prior shared splits/pilot. Arrow shard hashes reverified, original files read-only; new image bytes written only under this task confirmation/images. Conflict types199object43attribute58relation. Same frozen block24 and same-configuration block0 only; no confirmation-based selection.

Fresh600: block24CF0.7399556 (CI0.70680–0.77484), block0CF0.5910333 (CI0.55976–0.62630). AUC difference24−0 CI0.10896–0.18891. Same-layer plain matching0.71265; CF−matching CI−0.00029–0.05585, so superiority over matching not established. Unclipped contrast0.7674611; CF still loses ranking through its one-sided clipping. ObjectCF0.78961, attribute0.68983, relation0.61623. Every prediction audited one vision encoding/one language prefill, no answer generation. Current median block24 latency0.0963s andblock0 0.0617s, not a sampling speed benchmark.

Completed experimental round. Preserve confirmation set from future tuning. Candidate is promising relative toblock0, but richer semantic hypotheses, conflict/ignorance modeling and distinctiveness beyond lens/contrast remain open research. No old work results or environment dependencies changed.
