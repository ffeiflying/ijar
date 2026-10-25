# ACF 逐层选层策略(定稿版)

> 状态:**定稿**。选层只用 ACF 值与样本来源(clean/STA),全程不看模型回答;评测在大留出集上按"答案是否正确"算 AUROC。
> 相关文件:图 `results/selectlayer/figs/layer_selection.png`(矢量 `.pdf`),实现脚本 `selectlayer/fig_layer_selection.py`、`selectlayer/acf_layer_select_final.py`。

---

## 1. 问题设置

模型含 $B$ 个 transformer block:

| 模型 | $B$ |
|---|---|
| Qwen2.5-VL-7B | 28 |
| InternVL2.5-8B | 32 |
| MoF-7B | 40 |
| DeepSeek-VL2-Tiny | 12 |

对第 $n$ 个样本、第 $b$ 个 block($b=0,\dots,B-1$),在 **FFN 下投影(down-projection)输入处**采集逐 token 的 ACF/CF 冲突并沿生成 token 平均,记为

$$
x_{n,b}\ \in\ \mathbb{R},\qquad n=1,\dots,N,\quad b=0,\dots,B-1 .
$$

**开发集**由来源已知、但答案未知的样本组成:50 个 clean 样本 $D_{\text{c}}$ 与 50 个攻击样本 $D_{\text{a}}$(STA),$N_{\text{c}}=N_{\text{a}}=50$。开发集是选层唯一允许使用的数据。

---

## 2. 选层算法(只读 ACF,不看答案)

### Step 1 — 逐层两组统计量

$$
\mu_{\text{a}}(b)=\frac{1}{N_{\text{a}}}\sum_{n\in D_{\text{a}}} x_{n,b},\qquad
\mu_{\text{c}}(b)=\frac{1}{N_{\text{c}}}\sum_{n\in D_{\text{c}}} x_{n,b},
$$

$$
\sigma(b)=\sqrt{\frac{(N_{\text{a}}-1)\,s_{\text{a}}^2(b)\,+\,(N_{\text{c}}-1)\,s_{\text{c}}^2(b)}{N_{\text{a}}+N_{\text{c}}-2}},
$$

其中 $s_{\text{a}}^2(b),\,s_{\text{c}}^2(b)$ 是两组在 block $b$ 上的无偏样本方差。

### Step 2 — 选层判据(归一化效应量)

$$
g(b)=\dfrac{\big|\,\mu_{\text{a}}(b)-\mu_{\text{c}}(b)\,\big|}{\sigma(b)}
\qquad(\text{Cohen's }d\text{ 型效应量})
$$

**归一化是必须的**:ACF 冲突绝对量随深度单调放大(下图下排,如 Qwen 自 $b{=}0$ 的 $\sim0.04$ 涨到 $b{=}27$ 的 $\sim0.93$)。未归一化的 $|\mu_{\text{a}}-\mu_{\text{c}}|$ 会因深层幅度天然占优而恒选最后一层(dilution trap);除以 $\sigma(b)$ 后各层才在同一尺度上按"信噪比"比较。

### Step 3 — 锁定层与方向

$$
b^{*}=\arg\max_{b}\ g(b),\qquad
\operatorname{sign}\ =\ \operatorname{sign}\!\big(\mu_{\text{a}}(b^{*})-\mu_{\text{c}}(b^{*})\big)\in\{+1,-1\}.
$$

符号不可省:某层上"攻击使冲突升高/降低"是架构特有的(如 mof 在 $b^*{=}32$ 为冲突下降)。

---

## 3. 评测(选层之后才做)

检测器只读固定单层标量并做符号校准:

$$
z_n=\operatorname{sign}\cdot x_{n,b^{*}} .
$$

留出大集 $\mathcal{E}$ = STA 剩余 + clean 剩余($\approx 900$),标签由**答案**给出(GT 恒为 `a`;不安全 $=$ 模型输出≠`a`):

$$
\widehat{\text{AUROC}}=\frac{1}{|\mathcal{P}|\,|\mathcal{N}|}\sum_{i\in\mathcal{P}}\sum_{j\in\mathcal{N}}\mathbf{1}\!\big[z_i>z_j\big],
$$

$\mathcal{P}$: 留出集中答错样本;$\mathcal{N}$: 答对样本。

### 定稿结果

| 模型 | $b^{*}$ | sign | 留出 answer-AUROC(canonical) | 60-seed 稳健性 |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 27 | $+$ | 0.822 | 0.832 ± 0.029 |
| InternVL2.5-8B | 22 | $+$ | 0.959 | 0.912 ± 0.096 |
| MoF-7B | 32 | $-$ | 0.997 | 0.998 ± 0.001 |
| DeepSeek-VL2-Tiny | 6 | $+$ | 0.987 | 0.975 ± 0.008 |

60 次随机重抽开发集(每次 50 clean + 50 STA)后重选层,结果稳定;选中的层落在效应量等价平台内(mof 的 $b{=}25..35$、deepseek 的 $b{=}6/11$),非碰运气。

---

## 4. 与可视化图的对应

- **上排 (a)–(d)**:选择器在开发集 100 样本上实际算到的逐层 $g(b)$(黑色曲线),黑点 = $b^{*}=\arg\max$;面板内标注 $b^{*}$、符号与留出 AUROC。四个模型共用同 $y$ 轴,可比较峰值位置。
- **下排 (e)–(h)**:开发集逐层原始 ACF(蓝 clean / 橙 STA,±1 SE 带,虚线对准 $b^{*}$)。直观展示 $x_{n,b}$ 随深度放大 → 归一化动机。

---

## 5. 如实说明 / 局限

1. **选层监督与评测监督不同**:选层用"是否攻击"(来源,选层时可知,且不需要答案);评测用"答案是否对"(须在留出集标注后才知道)。当攻击命中率不高时(如 Qwen 的 STA 仅约 58% 真把答案改错),来源与"答错"并不等价,answer-AUROC 存在固有上限——Qwen 的 ~0.83 正是这一代价,如实上报。
2. **跨攻击族泛化**:STA 上训出的符号在 RTA 等其他攻击上可能反转(如 Qwen),评测跨攻击需单独测符号或注明局限。
3. 若下游语义是"输入是否被攻击"(而非"答案是否对"),则去掉上面第 1 点的损耗,来源选层的攻击-AUROC 为 qwen 0.997 / internvl 0.953 / mof 0.992 / deepseek 0.912(见 `acf_provenance_final.py`)。
