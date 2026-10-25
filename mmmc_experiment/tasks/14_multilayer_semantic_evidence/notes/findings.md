# Findings and constraints

Task13 direct block0 lens failed basic lexical recovery (32 rows,16 images); allpatch control found only5/57 query subwords among any patch top10 at block0 output. This does not establish no semantic information. User now permits mid/late-layer comparison and all methodological changes grounded in evidence theory.

SADT https://arxiv.org/html/2608.07302v1 provides intermediate visual-token semantic readout; it already uses internal logit lens. Merely adding a lens is not a distinctive contribution. Our candidate must establish benefit from meaningful masses/conflict/ignorance over internal evidence, without multiple answer generation. No claim of ICCV-level novelty until results and related-work review support it.

For centered classifier weights B[K,J], features h[J], evidence totals can be computed exactly as a=(abs(h)@abs(B).T+h@B.T)/2 and d=(abs(h)@abs(B).T-h@B.T)/2. This is algebraically identical to summing positive/negative feature contributions, and permits all256 patches without materializing[J,K] per patch. Canonical evidence_v2.combine_evidence then gives stable DS statistics. Normalized plausibility must reconstruct softmax(h@B.T/T).
