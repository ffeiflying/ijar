# 运行与迁移指南

## 1. 克隆后先选择一个任务

此仓库包含多个时期的研究代码，不应直接批量执行全部脚本。建议先选择 `scaling/`（原始 ACF 管线）或 `mmmc_experiment/tasks/` 中一个具体任务，阅读其说明和代码参数。

仓库不携带模型、数据和实验结果。能查看源码不等于已经具备复现实验所需的全部资产。历史 README 中提及的 `results/`、日志、图片和缓存均需自行生成或另行迁移。

## 2. 环境准备

根据所选模型建立独立 Python 环境，并安装与 GPU/CUDA 匹配的 PyTorch。常见依赖包括 `torch`、`transformers`、`accelerate`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`Pillow`、`tqdm`、`matplotlib` 和 `qwen-vl-utils`；MMMC 数据准备还需要 `pyarrow`，部分语义实验需要 `spacy` 及其语言模型。

以上是依赖导航，不是经验证的统一 requirements。不同模型可能需要不同 Transformers 版本或额外依赖，参见模型子目录。历史环境名称（如 `EUQ`、`qwen`、`internvl`）仅指原服务器的环境，不会随 Git 克隆迁移。

特别注意：部分 `layerauroc/` 历史脚本依赖原环境中修改过的 Transformers，以及外部保存的 MLP 投影权重。安装官方包不能自动重建这些修改。`scaling/run_scaling.py` 采用 forward hook 和实时读取权重，运行机制见 `scaling/DESIGN.md`；这不代表所有历史实验均已迁移到该机制。

## 3. 原始 ACF：最小检查与 Qwen 运行示例

以下命令从仓库根目录执行。

先在装有 PyTorch 的环境中检查闭式计算与历史公式的一致性（不下载模型）：

```bash
python -s scaling/test_acf_equivalence.py
```

此检查只验证其覆盖的公式与数值样例，不代表 MMMC 修正版、模型 hook 或整条实验管线均已验证。

运行模型前需要：

- 在 `scaling/run_scaling.py` 的 `MODELS` 字典中把模型 `path` 改成新服务器位置。
- 准备 `datasets/STA/sta_mini.csv`、`datasets/STA/images/`，或对应的 RTA 目录。
- 核对 CSV 的字段和图片命名；数据读取逻辑在同一脚本中，数据本身不包含在代码发布副本里。

准备完毕后，先用少量样本检查模型加载、特征提取和输出：

```bash
python -s scaling/run_scaling.py \
  --model Qwen2.5-VL-7B-Instruct --datasets STA --limit 2 --gpu 0
```

输出位于 `scaling/results/`。这是少量样本的运行检查，不能作为正式 AUROC 结论。模型的可用显存要求、dtype、预处理和生成配置需要结合所选模型核对。

## 4. MMMC：数据与任务版本

数据下载代码固定了 MMMC 的来源和 revision。示例：

```bash
python -s mmmc_experiment/tasks/01_dataset/code/download_dataset.py \
  --root "$PWD/mmmc_experiment/shared/data/MMMC" \
  --endpoint https://huggingface.co --test-only
```

该命令仅准备官方测试数据，不会生成实验的全部划分、图像文件和特征。训练任务还需官方训练数据。

历史 `prepare_splits.py` 读取 `tasks/01_dataset/results/test_manifest.json`，而下载器会写 `shared/data/MMMC/download_manifest.json`，两者结构不同；源码发布副本不包含前者。运行划分脚本前，需要从可信历史产物恢复该文件，或按脚本要求从下载清单的 `files` 中提取 `test/` 条目，并核对 revision、字节大小和 SHA256。还需预建 `shared/data/splits/` 目录。不要直接把两种 manifest 文件互相改名。

`shared/layout.py` 集中定义部分路径，并处理历史目录迁移，但它不会自动改写所有旧服务器绝对路径。划分文件中记录的图片绝对路径也要随迁移处理。

任务 `05_cf_correction` 的 `evidence_v2.py` 是早期 MMMC 公式修正入口；任务 `17_semantic_evidence_v2` 中的 V2 则指后续语义证据候选，不是同一个版本命名空间。任务 17 的 0.85 是研究目标，不能表述成已达到的成绩。

## 5. 迁移时需要检查的位置

先搜索原服务器路径，再只修改所选任务需要的配置：

```bash
rg -n '/data/|/home/|miniconda|CUDA_VISIBLE_DEVICES' \
  scaling layerauroc mmmc_experiment ablation cambrian_run
```

重点检查模型位置、数据位置、权重位置、输出目录、shell 脚本中的 Python 路径及 GPU 编号。不要全局替换第三方代码和历史研究记录。记录实际环境版本与命令，不要用旧日志中的路径直接启动任务。

## 6. 从实验工作目录生成源码发布副本

```bash
python3 tools/export_source.py
```

默认输出到 `github_export/IJAR/`，拒绝覆盖已有目标。只复制指定类型的源码、配置、文档和许可证，排除数据、图片、权重、结果、缓存、日志与本机工具目录；符号链接不跟随。导出前会检查文本大小及常见凭据格式，检查不是全面的秘密检测保证。

`SOURCE_EXPORT.json` 记录导出数量、体积和排除目录规则。之后可以在副本里初始化 Git 并上传到自己的 GitHub 仓库。原实验目录及其运行产物不会被删除或移动。
