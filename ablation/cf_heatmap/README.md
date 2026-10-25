# CF 可视化：当前采用的流程

用户于2026-10-17明确指定：后续采用本轮新流程，旧热图实现不再作为默认入口。

## 默认入口

- `run_cf_text_localization.py`：配对输入、已知文字掩码、局部token CF与决策位置CF归因。
- `summarize_cf_text_localization.py`：区域统计、配对bootstrap和统一色标可视化。
- 已核对的本轮结果：`results/cf_text_localization_20261017/README.md`、`index.html`。

## 必须保留的核验

1. CF定义沿用`envidence.py`，可微版本须与原实现核对，不能默默更改公式。
2. 改梯度路径后核验梯度有限性与有限差分；模型、层、token位置和目标标量明确记录。
3. 区分局部image-token CF、decision-CF GradCAM、image-mean-CF GradCAM和gradient×activation，不混称。
4. 局部图像token在causal decoder中看不到后续问题，不将其直接解释为问题条件化的图文语义冲突。
5. 保留零GradCAM；统一对照色标；不挑图、不根据测试热图选层；文字高响应不直接等于语义冲突。
6. 跨模型迁移需要重新验证token空间映射、前向接口与公式一致性，不能直接沿用本轮结论。

旧`run_cf_heatmap_*.py`、`plot_cf_heatmap.py`及旧结果保留作历史参考，不作为后续默认复现实验或结论依据。可复用的基础函数也须通过上述核验；采用新流程不代表默认其无错误。
