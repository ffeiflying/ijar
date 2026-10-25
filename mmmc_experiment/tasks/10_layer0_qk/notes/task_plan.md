# 第0层attention QK证据分解
核心：注意力softmax的类别是key位置，不是LM词表，不是把隐藏神经元强称语义类别。
固定图像patch位置作为共同attention目标框架，query坐标为特征，keys/sqrt(head_dim)为系数；center across key classes，逐坐标拆正负证据。验证normalized plausibility还原条件化的视觉attention softmax。

1. in_progress: two-row numerical smoke, actual post-RoPE Q/K, no prompt edits, layer0 only.
2. pending: dev100pairs/validation200pairs; compare query aggregation and evidence scales. Pool heads after CF calculation, report theoretical vs empirical parts separately.
3. pending: freeze dev-selected CF and evaluate heldout before claiming improvement.
