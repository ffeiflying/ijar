# V2 working methods and falsification criteria

These are experimental candidates, not accepted contributions. V1 is archived separately.

## A. Preserved head evidence (training free)

Question-only parser supplies object, attribute, relation and count atoms. Question slots such as “what sport” are excluded from asserted premises. Frozen vocabulary projection reads each visual patch during the same prefill; per-head conditional attention weights these patches. Single-token aliases form a union event; multi-subword fallbacks are affinities, not proper full-word probabilities. No answers or conflict labels enter feature construction.

For proposition i and attention head h, signed evidence is ell_hi=log p_hi−log q_i, with the same fixed reference as V1. Positive and negative effective evidence weights accumulate separately. Correlated heads are tempered by their count; optional image-attention discount acts on committed mass before fusion. Exact binary simple-support fusion yields m_i(P), m_i(not P), m_i(Theta) and pre-normalization internal conflict. Keeping two accumulators can distinguish equal net evidence with different conflict. Low lexical likelihood nevertheless does not establish physical absence or logical negation.

Dev-only selection over864 configurations chose layer24,32heads, no discount, temperature1, mean object CF. Dev0.7629; working validation0.72234 with image-bootstrap95%CI[0.67923,0.76588]. This does not improve on V1 validation0.72159. The exact same-head net-contrast control0.72769 is at least as competitive. These results reject a claim that merely preserving head conflict solves the task.

## B. Weakly supervised proposition mass assignment (pending)

Hypothesis frame for each parsed input premise is Theta_i={G_i,not G_i}, operationally grounded versus unsupported premise. Learn two nonnegative evidences e_i+ and e_i− from attention-conditioned visual context, query context, their products/differences, head lexical statistics and proposition type. Label-independent fixed random projection reduces contextual states. Train-only normalization; small shared MLP produces evidences via softplus.

Let S_i=e_i++e_i−+2. Set m_i(G_i)=e_i+/S_i, m_i(not G_i)=e_i−/S_i, m_i(Theta_i)=2/S_i. These are nonnegative and sum to1, but their semantic calibration must be tested. Different premises are different variables, NOT mutually exclusive classes. Under an explicit independent random-set approximation on their Cartesian product, push forward to global C=“at least one asserted premise unsupported”:

* Bel(normal)=product_i m_i(G_i).
* Bel(conflict)=1−product_i[1−m_i(not G_i)].
* m(global ignorance)=product_i[1−m_i(not G_i)]−product_i m_i(G_i).

Use global conflict belief as the CF score: it is also conflict with a categorical textual source asserting all premises hold. Learn from image/question bag labels, with BCE on this global score. A conflict bag only implies at least one unsupported premise; it does NOT label every atom false. A small normal-bag ignorance penalty discourages a vacuous solution. Parser omissions and source dependence remain limitations; product aggregation can favor long questions and needs controls.

Training source: verified official train shard;1000image pairs2000rows. Selection source:200different official train images400rows. Both exclude all2000official test images by ID and content SHA256. Compare layers24/31, two regularization strengths; epochs selected on calibration. Working validation400 is only evaluated after this selection. Existing task16confirmation600 remains unused.

## Mandatory controls

Use same atomic inputs and MLP for ordinary noisy-OR MIL with learned scalar per-atom scores; compare global internal-state MLP and pure input-embedding text-only MLP. These distinguish evidence modeling from additional supervision, feature richness and lexical shortcuts. A superior probe alone is not evidence-theory novelty. Report per-type scores, image-grouped intervals, and high-confidence error/ignorance behavior. A scalar binary evidence head cannot by itself demonstrate better uncertainty semantics.

The global formula in B is algebraically ordinary noisy-OR applied to opposition masses. Therefore a better score from this parameterization alone does not prove a new conflict operator or a richer ranking function class. Only independently useful ignorance/support diagnostics, reliable calibration, or a stronger multi-source construction could support a distinct evidential contribution. Question-only TF-IDF control, trained and selected solely on official train/calibration, already obtains working validation AUROC0.7439 [0.70437,0.78644]. The dataset permits substantial lexical shortcuts, which weakens any claim based on AUROC alone.

All VLM features come from one vision encoder call and one language prefill, stopping before answer logits. Offline training involves repeated MLP optimization; this must not be described as training free. Current extraction computes several layers and diagnostics, so its timing is not a minimal deployed detector latency.
