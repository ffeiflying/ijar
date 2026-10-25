# 取值位置和证据定义的依据

保留原始图文输入与内部证据分解，不使用LM/A-B输出头。

## 位置比较
- attention.wo之前：各head加权value已经汇合，视觉/文本token影响在此进入同一投影；选择该位置可减少后续FFN门控对原始混合证据的影响。
- FFN w1/w3之前：attention残差之后的归一化状态；分别用门控与value支路的已学习投影定义相对响应。
- FFN w2之前：原稿所选位置，经过非线性门控，仍作为基线。
所有位置都以实际线性投影weight[K输出,J特征]和对应输入构造逐特征证据，输入与权重成对匹配。

## 两种假设空间，不能混淆
既有多类别潜在通道法可按DST公式计算，但隐藏通道并非已证明互斥的语义类别；Denoeux的分类器等价性不能直接证明其语义。
本次额外比较“每通道独立二元极性框架”：对固定通道k，命题是相对投影响应的正/负极性。相反符号互斥，跨通道不要求互斥。逐特征线性贡献拆成正证据a_k和负证据d_k。

二元DST精确公式：
CF_k=(1-exp(-a_k))*(1-exp(-d_k))。
IG_k=exp(-a_k-d_k)/(1-CF_k)。
其normalized plausibility为sigmoid(a_k-d_k)，对应这个显式定义的二元线性读出。该读出是内部分析定义，不是模型输出分类器。

按通道平均/上10%均值/最大值聚合，是待验证的检测设计，不属于论文保证；温度也属于显式证据尺度改变。标定均值仅取开发集。证据源独立性/可区分性是建模近似，神经特征相关时不应作严格概率解释。

理论能保证质量函数及冲突计算在指定框架中成立，不能保证输入存在语义冲突时这些潜在响应一定相抵。后者需要图片互斥的验证和最终测试。

参考：Denoeux, Logistic Regression, Neural Networks and Dempster-Shafer Theory: a New Perspective, §3.1/3.2及AppendixA。
作者公开稿：https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/publi/nnbelief_kbs_v2_clean.pdf
