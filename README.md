# IJAR：多模态模型的内部冲突与不确定性实验

本仓库研究视觉语言模型内部的证据冲突能否用于识别文字攻击、图文冲突和错误回答。主要包含基于 Dempster–Shafer 证据分解的 ACF（冲突分数）、逐层分析、不同模型规模的比较，以及 MMMC 数据集上的后续探索。

这是研究代码集合，包含历史实现、对照实验和正在探索的方法。不同任务的公式、特征位置、标签含义和运行环境可能不同；不能把不同任务的分数或结果直接合并。研究计划中的假设和目标不代表已验证结论。

本次发布仅包含**源码、配置与说明文档**。模型、图片、数据集、提取特征、实验结果、运行日志和本机环境均不随仓库分发。历史文档中指向这些产物的链接在代码副本中可能不存在。

## 从哪里开始

- **了解原始方法**：先读 [`envidence.py`](envidence.py)，再看 [`scaling/DESIGN.md`](scaling/DESIGN.md) 和 [`scaling/acf_core.py`](scaling/acf_core.py)。`envidence.py` 的拼写是历史文件名，也是现有代码的导入名。
- **了解实验结构和迁移步骤**：读 [运行与迁移指南](docs/GETTING_STARTED.md)。
- **查看 MMMC 后续研究**：读 [`mmmc_experiment/README.md`](mmmc_experiment/README.md)，以及对应任务的 `README.md`、`notes/`。修正版证据计算入口是 [`evidence_v2.py`](mmmc_experiment/tasks/05_cf_correction/code/evidence_v2.py)，与根目录历史实现分开使用。
- **查看冲突可视化**：读 [`ablation/cf_heatmap/README.md`](ablation/cf_heatmap/README.md)，其中说明当前入口和解释边界。

## 目录导航

| 路径 | 用途 |
| --- | --- |
| `envidence.py` | 历史 DST 证据、冲突与 ignorance 计算 |
| `layerauroc/` | 多模型逐层冲突提取及 AUROC/AUPR 评估 |
| `layerwise/` | 逐层 ACF 分析和绘图 |
| `scaling/` | Qwen 系列模型规模实验；通过 forward hook 读取特征，实时使用模型权重 |
| `selectlayer/` | 层选择、开发集规模与选择策略对照 |
| `temperature/` | 温度相关分析 |
| `typo-parameter/` | 字体、透明度、颜色和位置等攻击参数实验 |
| `ablation/` | DINO/CLIP、特征交织、跨模态条件和 CF 可视化消融 |
| `cambrian_run/` | Cambrian 运行、评估和绘图脚本 |
| `myllava/`、`deepseek_vl2/`、`cambrian_src/` | 随实验保留的模型实现或适配代码；参考各文件的上游声明 |
| `mmmc_experiment/` | 按任务组织的图文冲突研究；每个任务有代码与研究记录 |
| `article/`、`docs/` | 写作记录、实验计划和方法说明；部分为历史记录 |
| `tools/export_source.py` | 从实验目录生成独立的源码发布副本 |

## 实验通常怎样运行

1. 准备指定模型、数据集和该模型需要的 Python 环境。
2. 在对应脚本中配置模型位置、数据位置和输出目录。
3. 提取指定层/位置的内部特征，计算冲突或不确定性分数。
4. 用该任务定义的标签计算 AUROC/AUPR，并保留原始配置及数据划分。

原始 STA/RTA 管线通常以模型回答是否错误作为标签；MMMC 管线可使用数据集的冲突标签。两者评估问题不同。新实验应遵循各任务的开发、验证和确认集隔离约定。

目前没有覆盖所有模型的统一安装命令，也没有完成全项目的一键复现封装。尤其是历史脚本中的绝对路径、额外权重文件和 Transformers 补丁，需要按 [运行指南](docs/GETTING_STARTED.md) 处理。

## 发布范围与第三方代码

本仓库未添加统一的开源许可证。保留的第三方文件应遵循其各自已有的许可证和版权声明；不能仅凭本仓库可下载就推定所有代码具有相同授权。数据和模型请从各自来源获取并遵守其使用条件。
