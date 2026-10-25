# 在第0层attention内部建立证据理论对应

## 真实attention计算
固定某个head、query位置和输入图像，视觉patch位置s是attention可选择的目标：
logit_s = sum_r q_r k_{s,r}/sqrt(d_head)。
视觉patch上的条件attention分布为softmax_s(logit_s)。这不是LM输出头，也未训练新的分类器。

## 分解
以query坐标r为特征，以视觉patch位置s为假设：
beta_{r,s}=(k_{s,r}-mean_{s'}k_{s',r})/sqrt(d_head)，
w_{r,s}=q_r*beta_{r,s}。
中心化仅减去各patch共有的logit项，因此不改变softmax。alpha=0是显式选定的证据规范，不声称满足某个未校准数据分布上的最小承诺最优解。
先分解每个坐标贡献的正/负部分，再沿r累加得到每个patch假设的a_s、d_s，随后用新版evidence_v2.combine_evidence融合。

Denoeux构造给出normalized plausibility ∝ exp(a_s-d_s) ∝ exp(logit_s)，因此能还原patch条件attention softmax。每条样本都做数值检查，误差阈值1e-5。
这里keys由输入决定，映射是固定输入/keys后的条件化对应，不是一个跨所有图像固定类别参数的GLR分类器；假设空间解释为attention目标位置，而非图像语义类别。

## 检测假设与需要验证的部分
不匹配的问题可能使视觉patch对query的证据分散或相抵，因此CF/IG可能变化。该检测命题不由softmax等价性自动推出。
保留post-RoPE Q/K和原始图文输入，第0层固定。比较prediction、最后问题token、所有问题token均值，以及按预先定义停用词表筛选的内容token均值/最大值。内容筛选是启发式，不是理论定理。跨head平均/前4/最大值也是待验证的汇聚选择。
温度改变证据尺度，等价于对该条件attention logits做温度变换；只在开发集选择。

来源：Denoeux (2019), Logistic Regression, Neural Networks and Dempster-Shafer Theory: a New Perspective，式25/27/30，AppendixA。作者公开稿 https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/publi/nnbelief_kbs_v2_clean.pdf
