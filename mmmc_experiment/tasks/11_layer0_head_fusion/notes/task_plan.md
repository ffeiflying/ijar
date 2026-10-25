# 第0层 attention-head 作为证据源
统一视觉patch目标框架，每个head用evidence_v2生成DS质量函数；在同一框架上直接融合各head，而不是简单平均head的内部CF。

1. complete: 严格融合公式经30组独立全质量函数枚举验证，CF与IG误差<5e-16。
2. in_progress: 抽取原始问题与图像，第0层同一query下比较全部heads与按attention熵筛选的4/8heads；第一层特征捕获后提前退出，后续block不执行，不改变第0层特征。
3. pending: dev选择配置，validation验证，保留未经监督CF主结果。单独评估带标注的head融合校准时必须明确标注，不混为无监督ACF。
