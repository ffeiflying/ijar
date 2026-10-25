# Findings

SADT https://arxiv.org/html/2608.07302v1 applies frozen unembedding to hidden states of high-attention image tokens. Its evidence concerns mid/late layers and generated object tokens. Transferring to block0 and input-question premises is unverified. Its masking/mitigation entails additional forward passes; this task borrows semantic readout only.

Raw attention magnitude, neuron signs, word similarity and softmax confidence are not automatically semantic belief masses. Missing a word is not proof of its negation. A word-level diagnostic is not a validated semantic-conflict detector.

Architecture audit: InternLM2 block0 = residual + attention(attention_norm(input)), followed by residual + FFN(ffn_norm(state)). Its language_model.model.norm and language_model.output provide the frozen normalized logit lens. Capture actual visual residual states, not projected Q/K head coordinates.

32 dev rows/16 images direct normalized logit-lens diagnostic completed. Common top1 tokens include 户数, 是瑞士, 〒 and punctuation at all three block0 positions. This is qualitative evidence of unsuccessful direct semantic readout, not proof of absence of semantic information. Next fixed diagnostic: cosine similarity to input token embedding and unembedding directions, with no training, no label use, no mass interpretation. This is an alternative readout, not SADT replication.

All three initial readout families completed (96 record outputs): logit lens, cosine, image-centered cosine. Exact query subword overlap in top10 words from top3 attended patches: normalized lens 0/28 normal and0/29 conflict; cosine input embedding1/28 normal and1/29 conflict; centered cosine0/28 and0/29. This overlap is a weak lexical proxy, not semantic accuracy. Importantly88.36% of query-selected top3 patches lie in the bottom row of the16x16 visual grid; could reflect positional attention sinks. Need all-patch coverage control before blaming semantic readout alone.
