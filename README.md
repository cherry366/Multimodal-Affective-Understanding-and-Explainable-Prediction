# 多模态情感理解与可解释预测系统（MAUEP）

> Multimodal Affective Understanding and Explainable Prediction
> 面向真实交互场景的文本—语音—视觉协同情感计算、局部缺失鲁棒建模与证据级解释。

本系统以多模态时序理解为目标，覆盖从原始视频到结构化特征、从缺失鲁棒预测到原生可解释输出的完整链路。核心设计强调三点：**异构模态的统一时序表征**、**信息不完整条件下的稳定决策**、**预测与解释共享同一条前向计算图**。

---

## 核心能力

- **统一时序表征**：音频解码时长作为全局时间基准，语音 74 维、视觉 35 维、文本 768 维特征统一映射到 50 个时序槽位，并以 `mask + 真实长度` 区分填充与真实缺失。
- **缺失鲁棒预测**：MoDE-Former+ 通过掩码感知 Transformer、跨模态注意力、模态内专家路由（MoE）与覆盖率先验门控，对局部连续缺失进行稳定建模。
- **原生可解释预测**：AEF 将证据权重直接注入前向池化与模态融合，解释与预测同源，可输出主要参考模态、模态作用度与关键证据定位。
- **不确定性建模**：异方差高斯头输出强度预测的置信度，为高风险样本提供可量化的可靠度信号。

---

## 系统架构

```text
原始视频 / 转写文本
        │
        ├─ 语音 74 维 ─┐
        ├─ 视觉 35 维 ─┼─ 50 槽时序对齐 + 有效掩码 ─┐
        └─ 文本 768 维 ┘                           │
                                                   ▼
                                   MoDE-Former+ 鲁棒融合骨干
                        掩码感知编码 · 跨模态注意力 · MoE · 覆盖率先验门控
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
              极性 / 强度 / 不确定度                              AEF 原生证据解释
                                                          主要模态 · 模态作用度 · 关键证据
```

---

## 模块与结果

| 模块 | 技术方案 | 关键结果 |
| --- | --- | --- |
| **I. 多模态特征与对齐** | 统一音频时间轴；Whisper 词级时间戳 + 序列比对；50 时间箱 / 50 词槽双版本；有效掩码与质量审计 | 覆盖全部 100 条原始样本；音视互相关中位滞后 −0.099 s；六项自动化校验通过 |
| **II. 缺失鲁棒预测** | **MoDE-Former+**：mask-aware Transformer、跨模态注意力、模态内专家路由、coverage-prior gating、知识蒸馏、异方差不确定性 | Accuracy **0.6479**、Macro-F1 **0.6179**、MAE **0.6292**、Pearson **0.6795** |
| **III. 可解释预测** | **AEF（Ante-hoc Evidence-weighted Fusion）**：证据权重参与前向池化与融合，预测与解释同源 | Accuracy **0.6113**、Macro-F1 **0.5961**、MAE **0.6258**、Pearson **0.6301**；语音/视觉证据胜率 **0.625 / 0.581** |

### 关键观察

- 文本是主导模态：涉及文本局部缺失的条件性能下降最明显，语音与视觉更多承担辅助证据角色。
- 鲁棒性并非依赖特定缺失几何形态：在 0～50% 缺失率范围内性能平稳缓降，未出现单调崩塌。
- MoE 是缺失鲁棒性的主要正贡献；覆盖率先验门控的收益更多体现在门控机制的单调性与可解释性。
- AEF 以约 1 个百分点的分类代价，换取语音、视觉局部证据显著优于随机删除，实现可定位、可复核的三模态解释。

---

## 仓库结构

```text
liu/
├── README.md
└── release/
    ├── 01_feature_alignment/
    │   ├── code/                 # 特征提取、时序对齐、可视化
    │   ├── features/             # 100 条样本特征、汇总表、对齐记录
    │   ├── docs/                 # 模块说明、复现手册
    │   └── environment/          # 依赖清单
    ├── 02_robust_prediction/
    │   ├── code/                 # 模型、训练、推理、消融、绘图
    │   ├── weights/              # 3 个随机种子权重
    │   ├── docs/                 # 技术设计、复现说明
    │   └── environment/
    ├── 03_explainable_prediction/
    │   ├── code/
    │   │   ├── aef/              # 原生证据加权融合
    │   │   ├── baseline_posthoc/ # 后验解释对照
    │   │   └── shared_backbone/  # 共享鲁棒骨干
    │   ├── weights/              # AEF 三种子权重
    │   ├── docs/
    │   └── environment/
    ├── results/
    │   ├── missing_modality_predictions.csv
    │   ├── missing_modality_summary.md
    │   └── explainable_predictions.csv
    ├── RELEASE_NOTES.md
    └── WEIGHT_STORAGE.md
```

---

## 快速开始

### 环境

- Python 3.10，PyTorch 2.0（CUDA 可选，代码自动选择设备）
- 特征对齐模块另需 OpenFace 2.2.0、`transformers==4.44.2`，以及 BERT / Whisper 预训练模型
- 各模块精确依赖见 `release/*/environment/`

```powershell
python -m pip install -r release/02_robust_prediction/environment/requirements.txt
```

### 数据准备

原始视频、标准化多模态特征与专项测试数据未随仓库分发。数据放入各模块的 `data/` 目录即可（目录约定见 `release/*/docs/`）；需要换盘或换路径时，用 `Q1_ROOT` / `Q2_ROOT` 覆盖模块根目录。

### 运行入口

| 模块 | 工作目录 | 入口 | 文档 |
| --- | --- | --- | --- |
| 特征对齐 | `release/01_feature_alignment/` | `python code/pipeline.py all` | [复现手册](release/01_feature_alignment/docs/reproduction.md) |
| 鲁棒预测 | `release/02_robust_prediction/` | `python code/infer_final_v2.py` | [技术设计](release/02_robust_prediction/docs/technical_design.md) |
| 可解释预测 | `release/03_explainable_prediction/` | `python code/aef/infer_q3v2.py` | [复现手册](release/03_explainable_prediction/docs/reproduction.md) |

权重以 fp16 存储以控制仓库体积，加载时自动转回 fp32；精度核验与转换方式见 [WEIGHT_STORAGE.md](release/WEIGHT_STORAGE.md)。

---

## 数据与边界

- 仅使用 CMU-MOSEI 系列数据，不引入外部情感数据集。
- 原始数据、预训练模型缓存与大体积中间产物不进入仓库。
- 所有结果均由本仓库代码与随附权重产生；随机种子、训练配置和评估口径在模块文档中完整记录。
