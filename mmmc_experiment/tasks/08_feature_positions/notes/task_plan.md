# Re-select attention/FFN evidence features
User scope: keep internal attention-level ACF and evidence theory; no LM/A-B scores. Existing environment unchanged.

1. in_progress: Extract true attention output-projection inputs (attention.wo) and post-attention FFN input (w1/w3) at layers0,1,4,8,16,24,31, original image/question. FFN-downprojection baseline already available in task07.
2. pending: Use canonical evidence_v2; compare positions/layers/scales on dev100pairs. Same latent-hypothesis construction, fixed weight axes, no samplewise fitted alpha.
3. pending: Freeze selected configuration; evaluate validation200pairs and fresh ACF-test300pairs. Report rawCF and stable monotone logscore distinctly, with pair-bootstrapCI.
4. pending: Explain theoretical conditions and limitations. These are latent hypotheses, not semantic classes guaranteed by Denoeux classifier equivalence. Exact visual/text attention decomposition remains an additional, separately tested extension.

Progress: original dev200 and validation400 features extracted at21 new projection positions (7layers×wo/w1/w3), with old w2 features retained as matched baseline. Binary polarity-frame CF/IG formulas independently verified. Canonical v2 totals-only API added to avoid recomputing per-token fusion during sweeps; 60-case evidence verifier still passes. Scan launched, using only internal evidence scores. Old w2 frozenL24/T.25 fresh ACFtest AUROC0.60584 CI[0.57901,0.63257]. Internet checked: hf-mirror HTTP200, but all experiments local.

Resume audit: old process handle67026 missing; external pgrep confirmed no live scan_positions.py and log unchanged since23:35. Scan stopped after L8_ffn_w1, preserving completed score files. Added row-ID-validated checkpoint resume per completed site; restart only uncomputed sites, retaining all scores already computed.

用户最新约束：固定第0层作baseline，不再做block选择；之后所有汇报仅给第0层。停止跨层扫描，保留历史结果；新位置脚本默认layers=0。第0层zero/T1验证400条：wo CF0.49269，w1 CF0.48396，w3 CF0.48479，w2 CF0.48011。
