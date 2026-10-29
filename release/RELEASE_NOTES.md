# Release Notes

本目录是工程快照，包含特征对齐、鲁棒预测与可解释预测三个模块的代码、权重、结果和复现说明。

## 模块

| 模块 | 内容 | 位置 |
| --- | --- | --- |
| 特征对齐 | 全部 100 条样本的三模态特征、对齐记录、质量汇总、代码与复现手册 | `01_feature_alignment/` |
| 鲁棒预测 | MoDE-Former+ 代码、3 个随机种子权重、技术设计与运行环境 | `02_robust_prediction/` |
| 可解释预测 | AEF 主模型、后验解释对照、共享骨干、3 个随机种子权重 | `03_explainable_prediction/` |
| 结果 | 缺失测试集预测、全量汇总、可解释测试集预测与解释 | `results/` |

## 结果文件

- `results/missing_modality_predictions.csv`：缺失测试集 30 条样本的极性、强度、不确定度、门控权重与缺失统计。
- `results/missing_modality_summary.md`：30 条样本的逐条汇总表。
- `results/explainable_predictions.csv`：可解释测试集 20 条样本的预测、模态作用度、Shapley 互证与关键证据定位。

## 环境与权重

- Python 3.10，PyTorch 2.0；各模块依赖清单位于对应 `environment/` 目录。
- 模型权重以 fp16 存储，加载时自动转回 fp32；转换方式与精度核验见 `WEIGHT_STORAGE.md`。
- 原始数据、预训练模型缓存和运行中间产物不随仓库分发。

## 复现约定

- 每个模块自包含：代码在 `code/`，数据放 `data/`，输出写 `q*_output/`（已被 `.gitignore` 忽略）。
- `config*.py` 默认把「本文件所在目录的上两级」解析为模块根目录，因此仓库按原结构放好即可运行。
- 需要换盘或换路径时，用环境变量覆盖：`Q1_ROOT`（模块一）、`Q2_ROOT`（模块二与模块三）。
- 发布的权重已随仓库提供；推理时若本地训练输出为空，代码会自动回退到 `weights/`。
