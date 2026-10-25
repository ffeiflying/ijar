# 第0层：从模态来源构造证据冲突

## 可核对的代数部分
对固定query，self-attention每个head输出为 c=sum_s a_s v_s。根据实际IMG_CONTEXT token ID，将key位置划分为V（图像）和T（其他文本/上下文）：
cV=sum_{s in V} a_s v_s；cT=sum_{s in T} a_s v_s，因此c=cV+cT。
拼接head后，输出投影保持线性：Wo(cV+cT)=Wo cV+Wo cT。本实验在原始模型forward中重构并检查cV+cT与实际wo输入，而不是遮图/改提示后比较另一趟forward。

## 证据框架
每个输出通道单独定义相对投影响应的正/负极性二元框架，避免把不同神经元当成互斥语义类别。两种模态使用完全相同的投影系数、中心化规范和通道框架。逐特征贡献由evidence_v2计算，分别形成正/负支持总量a_m,k、d_m,k。

对于每一模态m，先用二元DST构造正规化质量函数：
kappa_m=(1-exp(-a_m))*(1-exp(-d_m))；
m_m(+)=(1-exp(-a_m))*exp(-d_m)/(1-kappa_m)；
m_m(-)=(1-exp(-d_m))*exp(-a_m)/(1-kappa_m)；
m_m(Theta)=exp(-a_m-d_m)/(1-kappa_m)。

再计算两模态之间的冲突：
kappa_VT=mV(+)*mT(-)+mV(-)*mT(+)。
融合后IG=mV(Theta)*mT(Theta)/(1-kappa_VT)。
这明确区分“单一模态内部特征相抵”和“视觉/文本两来源相抵”。后者才是此次待验证的ACF变体。

## 需实验检验的设计
- prediction、question_end、question_mean三种query汇聚；
- 保留原始attention质量，或每head按模态attention质量归一化；后者是条件化上下文，已不再要求两部分相加等于原forward；
- 证据温度和通道平均/上10%/最大汇聚。
以上均是检测设计选择，不能声称由DST保证AUROC提高。尺度在开发集选择；验证图片不参与选择。

## 理论边界
潜在响应极性具有明确的操作定义，但它是否对应输入语义矛盾需要实验证据。视觉和文本贡献共享query且神经特征相关，来源独立性是近似；冲突分数不应称为已校准的语义冲突概率。文本来源含system/assistant等上下文，不等于纯用户问题token。

数学验证见results/math_verification.json：100组独立质量函数集合枚举、完全无知和强对立边界。forward重构检查独立于DS公式验证。
