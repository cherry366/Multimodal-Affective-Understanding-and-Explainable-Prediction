# 模块二：缺失鲁棒情感预测（MoDE-Former+ v2.1）

> 输入统一为 50 槽位对齐序列（text 768 / audio 74 / vision 35），缺失位置特征置 0 且 mask=0；
> 未引入任何外部情感数据集。

## 一、目录内容

- `docs/`：`technical_design.md` 给出建模原理、网络结构、目标函数、训练方案、关键参数、
  消融规律与错误归因。
- `code/`：模型、配置、数据、训练、评估、推理、消融、网格搜索与绘图脚本。
- `weights/`：`final_v2_s{42,2024,7}.pt` 三种子权重（fp16 存储，加载时转回 fp32）。
- `environment/`：实测依赖清单与版本说明。

## 二、能力与产出对照

| 能力 | 实现 / 产物 |
|---|---|
| 缺失条件下的鲁棒预测 | `code/model_q2.py`（MoDE-Former+）、`weights/final_v2_s*.pt` |
| 情感极性（三分类） | `missing_modality_minimal.csv` 的 `polarity_class / polarity_name` |
| 情感强度（回归） | `missing_modality_minimal.csv` 的 `intensity` |
| 缺失模态类型影响 | `fv04` + `ablation_type_v2.csv` |
| 缺失位置影响 | `fv05` + `ablation_position_v2.csv` |
| 缺失时长影响 | `fv16` + `ablation_duration_v2.csv` |
| 缺失率影响 | `fv06` + `ablation_rate_v2.csv` |
| 缺失测试集推理 | 30 条预测（`../results/`） |
| 结构 / 目标 / 训练 / 参数 | `docs` 第二、三节 + `fv01 / fv15` |
| 组件消融 | `fv11` + `ablation_v2_components.csv` |
| 验证集性能 / 可视化 / 错误归因 | `docs` 第四节 + `fv03 / fv07 / fv08 / fv14` |

## 三、输出格式

- `missing_modality_minimal.csv`：列 `id, polarity_class, polarity_name, intensity`，30 行 + 表头。
  `polarity_class` 取 0=Negative / 1=Neutral / 2=Positive，`intensity` 为连续情感强度。
- `missing_modality_predictions.csv`：18 列过程版，另含三类概率、不确定度 σ、门控 α 与缺失统计，
  用于逐条核查。
- 仓库 `../results/` 下是同一口径的发布副本。

## 四、复现运行

- 环境：Python 3.10、torch 2.0（CUDA 可选）、transformers 4.44.2，完整清单见 `environment/requirements.txt`。
- 数据放置（相对本模块根目录，可用 `Q2_ROOT` 覆盖根路径）：

  | 文件 | 位置 |
  |---|---|
  | `aligned_50.pkl`、`label.xlsx` | `data/standard_features/` |
  | 缺失测试集对齐特征 | `data/missing_modality/aligned/` |

- 运行前设置（离线，避免访问 HuggingFace）：
  `$env:PYTHONIOENCODING='utf-8'`、`$env:HF_HUB_OFFLINE='1'`、`$env:TRANSFORMERS_OFFLINE='1'`
- 关键命令（工作目录 = `02_robust_prediction/`）：
  1. 缺失测试集推理：`python code/infer_final_v2.py`
  2. 全套图：`python code/figures_final_v3.py`
  3. 类型 / 位置 / 缺失率 / 极端条件消融：`python code/ablation_v2_conds.py`
  4. 组件消融：`python code/ablation_v2_components.py`
  5. 连续 vs 分散缺失：`python code/ablation_contig_dispersed.py`
  6. λ 网格搜索：`python code/grid_search.py`
  7. 混淆矩阵 + 不确定度校准：`python code/eval_final_v2.py`

  权重查找顺序为 `q2_output/ckpt` → `weights/`，命中即用；训练输出写入 `q2_output/ckpt/`。

## 五、主结果（final_v2，三种子软集成）

- 缺失测试集（同分布局部缺失）：Acc 0.6479 / F1-macro 0.6179 / MAE 0.6292 / Pearson 0.6795。
- 关键结论：文本局部缺失最敏感（F1 约 −0.6pp）；缺失率 r=0~50% 平稳缓降、无崩塌；
  MoE 是主要正贡献（−0.9pp），KD 小幅正贡献；覆盖率先验的价值在于门控随缺失率单调变化，
  用于机制可解释性而非直接提分。

## 六、版本说明（v2.1-r1）

- 已移除 v1 遗留文件：`eval_final.py`、`final_valid_confusion.csv`、`final_uncertainty_calibration.csv`、
  `final_vs_baseline.csv`。
- 已新增：`eval_final_v2.py`、`final_v2_valid_confusion.csv`、`final_v2_uncertainty_calibration.csv`。
- 错误归因统一为 final_v2 软集成口径（N=728，Acc=0.6195；clean 混淆矩阵 Neg 131/44/31、
  Neu 33/101/50、Pos 39/80/219）。
- 不确定度校准：σ 与 |误差| 的 Pearson 相关 = 0.290，σ 五分桶 MAE 单调上升。
