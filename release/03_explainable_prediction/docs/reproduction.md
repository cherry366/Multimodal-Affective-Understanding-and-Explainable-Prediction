# 模块三 可解释性情感预测建模与验证 —— 实验手册

本手册说明模块三（可解释预测）的全部工作、核心结果、代码与数据组织方式、复现步骤，并给出代码与结果之间的对应关系。
全程统一使用「后验基线 / AEF」称呼；代码/权重文件名中的数字后缀（如 `q3v2_*`、`final_v2_*`）只是版本标记，不代表方案代号，
具体对应关系见第 4 节目录说明。

---

## 0. 一句话结论

在模块二鲁棒融合骨干之上，构建了**原生证据加权融合（Ante-hoc Evidence-weighted Fusion，AEF）**
可解释模型。与“先预测、后解释”的后验思路不同，AEF 让三模态的局部证据权重直接参与前向池化与最终决策，
因此“被解释的位置就是被模型实际使用的位置”。验证集极性准确率 0.6113、宏 F1 0.5961、强度 MAE 0.6258、
Pearson 0.6301；文本、语音、视觉的局部证据忠实度全部优于随机删除；可解释测试集的 20 条样本已完成全量预测
与解释，并按规范输出交付文件。

## 1. 两种方案的关系（最终方案与对照基线）

- **后验基线：后验解释基线**。完整复用模块二骨干做预测，之后在推理期用“留一模态遮挡 + 位置遮挡”
  做事后归因（Shapley / 遮挡 LOPO）。解释与预测不在同一条前向链路，语音与视觉的位置级证据忠实度不优于随机。
- **AEF：原生证据加权融合（AEF，最终方案）**。冻结模块二编码器，把每个模态的均值池化替换为可学习的
  证据加权池化，证据权重在训练中直接参与决策；同时保留跨模态注意力、模态专家路由与覆盖率先验门控，
  输出极性与强度双任务。Shapley 与留一遮挡只作为模型无关互证，不作为主口径。
- **取舍依据**：AEF 以约 1 个百分点的分类代价（Accuracy 0.6195→0.6113，F1 0.6070→0.5961），
  换取语音证据胜率 0.405→0.625（p=1.7e-4）、视觉证据胜率 0.364→0.581（p=9.5e-5），
  即“用少量精度换取显著、可复核的三模态局部证据”。文档与最终交付均以 AEF 为主模型，
  后验基线作为对照基线写入显著性检验。

## 2. 核心方法（AEF）要点

对第 m 个模态（m ∈ {text, audio, vision}）的 50 个时序槽位表示 h_m(t)：

- 证据打分：z_m(t) = Ev_m(h_m(t))，Ev_m 为每模态一个 Linear(128→1)。
- 原生证据：e_m(t) = sigmoid(z_m(t)) · mask_m(t)，用于可视化“原始证据强度”。
- 证据加权池化（替代均值池化）：
  ē_m(t) = softmax_t( z_m(t) | mask_m(t)=1 )，
  p_m = Σ_t ē_m(t) · h_m(t)。
- 模态作用度：α = softmax( Gate([p_T,p_A,p_V, coverage]) + β·log coverage )，
  α_m 即“该模态在融合中占多少”。
- 融合与输出：z = Σ_m α_m · enriched_m(p_m)，再分别进入极性分类头、强度回归头、不确定度头（异方差高斯）。
- 目标函数：L = L_cls + λ1 L_reg + λ2 L_nll，λ1=0.3、λ2=0.1；
  L_cls 为类别逆频加权交叉熵（权重 [1.170, 1.493, 0.678]），
  L_reg = SmoothL1(μ, y)，L_nll = 0.5[(y−μ̄)² exp(−log σ²) + log σ²]。
- 互证归因（模型无关，仅作核对）：Shapley 值 φ_m = Σ_{S⊆M\{m}} w(|S|)[v(S∪{m})−v(S)]，
  与门控 α 做主导一致率与 Spearman 相关互证。
- 证据回看：利用模块一建立的词级时间戳，把证据槽位 t 映射到原始时间区间 [t·T/50, (t+1)·T/50)，
  得到支撑判断的原始词、语音时段与视觉关键帧。

训练采用“冻结编码器、只微调融合与解释头”的热启动：AdamW（lr=1e-4、wd=0.01）、线性预热 + 余弦衰减、
梯度裁剪 5.0、最多 20 轮早停、种子 {42, 2024, 7} 三种子软平均集成，决策阈值 τ 逐种子在验证集上以
宏 F1 最优选取（集成后 τ=0.0833）。

## 3. 核心结果汇总

### 3.1 验证集基础性能（三种子软集成，N=728）

| 指标 | 数值 |
| --- | --- |
| Accuracy | 0.6113 |
| Macro-F1 | 0.5961 |
| Weighted-F1 | 0.6158 |
| MAE（强度） | 0.6258 |
| Pearson（强度） | 0.6301 |
| 决策阈值 τ | 0.0833 |

逐类（样本数 / 准确率 / F1 / 强度 MAE）：

| 类别 | 样本数 | 准确率 | F1 | 强度 MAE |
| --- | --- | --- | --- | --- |
| Negative | 206 | 0.626 | 0.640 | 0.804 |
| Neutral | 184 | 0.505 | 0.465 | 0.498 |
| Positive | 338 | 0.660 | 0.683 | 0.587 |

混淆矩阵（行=真值 Neg/Neu/Pos，列=预测 Neg/Neu/Pos）：
[[129,44,33],[32,93,59],[36,79,223]]。

### 3.2 三模态作用差异与主要参考模态

| 模态 | 平均门控 α | Shapley 主导数（可解释测试集） | Spearman(α,φ) |
| --- | --- | --- | --- |
| 文本 | 0.608 | 16 | +0.398 |
| 语音 | 0.206 | 4 | +0.057 |
| 视觉 | 0.186 | 0 | −0.366 |

门控 α 与 Shapley φ 的主导一致率为 0.745。

### 3.3 局部证据忠实度（top-k 证据删除 vs 随机删除胜率）

| 模态 | 后验基线胜率 | AEF 胜率 | 结论 |
| --- | --- | --- | --- |
| 文本 | 0.695 | 0.580 | 均优于随机 |
| 语音 | 0.405 | 0.625 | 后验基线不优于随机，AEF显著优 |
| 视觉 | 0.364 | 0.581 | 后验基线不优于随机，AEF显著优 |

### 3.4 可解释测试集全量推理（20 条）

- 极性分布：Negative 7、Neutral 6、Positive 7。
- 强度：均值 −0.001、标准差 1.070。
- 平均不确定度 σ：0.692。
- 主导模态：文本 16、语音 4、视觉 0。
- 完整逐条结果见交付文件《explainable_predictions.csv》。

### 3.5 两方案显著性检验（2000 次自助 + 配对 Wilcoxon）

| 指标 | 差值（AEF − 后验基线） | 结论 |
| --- | --- | --- |
| Accuracy | −0.87 pp（CI [−3.30, +1.51]） | 不显著 |
| Macro-F1 | −1.14 pp（CI [−3.71, +1.36]） | 不显著 |
| MAE | +0.021（显著） | AEF略差 |
| Pearson | −0.013（CI [−0.029, +0.004]） | 不显著 |
| 语音证据胜率 | 0.405→0.625（p=1.7e-4） | AEF显著优 |
| 视觉证据胜率 | 0.364→0.581（p=9.5e-5） | AEF显著优 |

### 3.6 模型规模与推理开销

| 项目 | 数值 |
| --- | --- |
| 总参数量 | 1,521,062 |
| 冻结编码器（proj+enc） | 907,520 |
| 可训练参数 | 613,542 |
| 原生证据打分头 ev | 387 |
| 单样本前向（单种子 / 三种子集成） | 27.6 ms / 99.4 ms（CUDA，batch=1） |
| 解释所需前向次数 | AEF 1 次；后验基线 8 次 |

## 4. 目录说明

```
03_explainable_prediction/
├── docs/
│   ├── reproduction.md            # 本文件：运行、验证与结论
│   └── technical_design.md        # 建模原理、结构、目标函数与选型依据
├── code/
│   ├── aef/                       # AEF 主模型：配置/网络/训练/推理/评估/绘图
│   ├── baseline_posthoc/          # 后验解释基线：配置/解释/训练/推理/评估/绘图
│   └── shared_backbone/           # 模块二编码与融合骨干（模块三复用）
├── weights/
│   └── aef/                       # q3v2_s{42,2024,7}.pt（三种子软集成，fp16 存储）
└── environment/requirements_q3.txt
```

**代码 ↔ 方法对照**：

| 目录 | 方法 | 说明 |
| --- | --- | --- |
| `code/aef` | AEF 原生证据加权融合 | 主模型，解释与预测同源 |
| `code/baseline_posthoc` | 后验解释基线 | 推理期 Shapley + 遮挡归因 |
| `code/shared_backbone` | 模块二骨干 | 单模态 Transformer 编码器 + 融合头 |
| `weights/aef` | AEF 三种子权重 | 软集成推理直接可用 |

AEF 训练时以模块二的发布权重热启动并冻结编码器；权重查找顺序为
`03/q2_output/ckpt` → `03/weights/init` → `02_robust_prediction/weights`，命中即用，无需手工拷贝。

## 5. 环境与依赖（与模块二同一 conda 环境实测）

- Python 3.10.13（conda 环境 `<你的Python环境>`，Windows）
- torch 2.0.0+cu118，torchvision 0.15.1+cu118，torchaudio 2.0.1+cu118
- transformers 4.44.2，tokenizers 0.19.1，huggingface-hub 0.36.2，safetensors 0.8.0
- scikit-learn 1.3.0，numpy 1.26.4，scipy 1.11.4，pandas 2.1.4
- matplotlib 3.8.0，seaborn 0.12.2
- opencv-python 4.10.0.84，pillow 10.2.0，tqdm 4.65.0，einops 0.8.2

完整版本见包根 `requirements_q3.txt`。GPU 非必需；代码自动选择 device。文本链路若需重算 BERT 时请设置
`HF_HUB_OFFLINE=1` 并确保 `bert-base-uncased` 已在本机缓存，避免联网阻塞。

## 6. 复现运行步骤

### 6.0 根路径解析

三个配置文件都按「本文件所在目录的上两级」解析 `ROOT`：

- `02_robust_prediction/code/config_q2.py` → `02_robust_prediction/`
- `03_explainable_prediction/code/{aef,baseline_posthoc,shared_backbone}/*.py` → `03_explainable_prediction/`

需要把工程搬到别的盘时，用环境变量覆盖即可，不必改代码：

```powershell
$env:Q2_ROOT = "D:\work\02_robust_prediction"
```

### 6.1 数据与权重

所需数据（同一数据源的 CMU-MOSEI 采样片段，未引入外部数据集）：

| 内容 | 放置位置 |
| --- | --- |
| 标准化特征`aligned_50.pkl`、`label.xlsx` | `02_robust_prediction/data/standard_features/` |
| 可解释测试集对齐特征 `01.pkl…20.pkl` | `03_explainable_prediction/data/explainable/aligned/` |
| 对应原始视频 `01.mp4…20.mp4`（关键帧回看） | `03_explainable_prediction/data/explainable/aligned/videos/` |
| 模块二发布权重 `final_v2_s*.pt` | 仓库已含：`02_robust_prediction/weights/` |
| AEF 发布权重 `q3v2_s*.pt` | 仓库已含：`03_explainable_prediction/weights/aef/` |

### 6.2 运行顺序（工作目录 = `03_explainable_prediction/`）

1）训练（三种子，产出 `q3_v2_output/ckpt/q3v2_s*.pt`）：
   `python code/aef/train_q3v2.py`
   单种子调试：`python code/aef/train_q3v2.py --seed 42 --epochs 20`。

2）可解释测试集推理 + 解释（产出 `explainable_predictions.csv`、
   `explainable_explanations.json`、`explainable_summary.json` 与关键帧）：
   `python code/aef/infer_q3v2.py`

3）验证集评估（产出 `valid_eval.json`、忠实度 CSV 等）：
   `python code/aef/eval_q3v2.py`

4）主图生成（框架图、混淆矩阵、证据热力、解释卡等）：
   `python code/aef/figures_q3v2.py`，再跑 `extra_figures_q3v2.py`、`extra_figures_q3v2_b.py`。

5）超参数敏感性网格：
   `python code/aef/sweep_q3v2.py`

后验基线对照：`python code/baseline_posthoc/train_q3.py` → `infer_q3.py` → `eval_q3.py` → `figures_q3.py`。

说明：所有输出默认写入各模块的 `q3_output/`、`q3_v2_output/`（已被 `.gitignore` 忽略）；
仓库 `results/` 下的 CSV 是同一口径的发布副本，可直接查阅。配置在导入时自动创建输出目录；
若测试集 pkl 由 numpy 2.x 保存，`data_q3.py` 已内置 `numpy._core → numpy.core` 兼容反序列化。

## 7. 交付文件说明

交付文件为 `../results/explainable_predictions.csv`（20 条，UTF-8-SIG，逗号分隔）。列含义：

| 列 | 含义 |
| --- | --- |
| id | 样本编号 01–20 |
| predicted_class / predicted_class_name | 情感极性预测（0/1/2 → Negative/Neutral/Positive） |
| predicted_score | 情感强度预测（连续值，越负越负向、越正越正向） |
| score_uncertainty_sigma | 强度不确定度 σ |
| classification_entropy | 分类熵 |
| prob_neg / prob_neu / prob_pos | 三类概率 |
| dominant_modality / dominant_modality_alpha | 主要参考模态（Shapley / α 口径） |
| alpha_text / alpha_audio / alpha_vision | 三模态作用程度（门控 α） |
| shapley_text / shapley_audio / shapley_vision | Shapley 边际贡献 |
| pi_text / pi_audio / pi_vision | Shapley 归一化作用度 |
| evidence_text | 关键证据词及位置（文本） |
| evidence_audio_windows | 关键证据语音时段（秒） |
| evidence_vision_frames | 关键证据视觉帧文件名 |

## 8. 图表索引

图表由 `code/aef/figures_q3v2.py`、`extra_figures_q3v2.py`、`extra_figures_q3v2_b.py` 生成，
输出到 `q3_v2_output/figs/`，均为 PDF + PNG 双格式。

| 图 | 内容 |
| --- | --- |
| qx2_01_arch | AEF 整体框架（预测主干 + 可解释输出） |
| qx2_02_confusion | 验证集混淆矩阵 |
| qx2_03_reg_mae | 逐类强度 MAE |
| qx2_10_reg_scatter | 强度回归散点 |
| qx2_07_error_attribution | 错误对统计与中性塌缩 |
| qx2_04_modality_diff | 三模态作用差异（α 与扰动重要性） |
| qx2_05_evidence_heatmap | 三模态原生证据时序热力 |
| qx2_06_faithfulness | top-k 证据删除 vs 随机删除 |
| qx2_18_necessity | 主导模态 LOPO 必要/充分性 |
| qx2_20_deletion_scan | 不同删除率证据删除扫描 |
| qx2_24_alpha_phi | α–Shapley 互证散点 |
| qx2_25_shapley_example | 典型样本 Shapley 瀑布图 |
| qx2_12_cards | 典型样本解释卡 |
| qx2_14_word_heat | 词级证据热力与高频证据词 |
| qx2_13_frames | 证据槽位对应视觉关键帧 |
| qx2_27_audio_evidence | 语音主导样本证据时段定位 |
| qx2_23_evidence_pattern | 证据聚合规律（位置 / 高频词 / 集中度） |
| qx2_22_significance | 两种方法显著性检验 |
| qx2_11_a4_summary | 可解释测试集全量结果汇总 |
| qx2_08_v1_vs_v2 | 后验基线 vs AEF 性能与忠实度对照 |
| qx2_09_flow | 模块三整体流程图 |
| qx2_16_tsne | 三模态表示流形可视化 |
| qx2_17neg / qx2_17neu / qx2_17pos | 负 / 中 / 正三类样本证据时序分布 |
| qx2_21_sensitivity | 学习率 × 权重衰减敏感性网格 |

另有 `qx2_15_cross_attention`、`qx2_19_attn_conc`、`qx2_26_complexity` 三张诊断图一并生成，留作核查。
后验基线的对照图由 `code/baseline_posthoc/figures_q3.py` 生成。

## 9. 注意事项

- 同一目录不要并行运行两个训练进程，避免中间 JSON / 权重被并发写坏。
- 关键帧回看依赖 `data/explainable/aligned/videos/` 下的原始视频；缺视频时该字段留空，其余结果不受影响。
- 显存不足时先调小 `BATCH`，或直接在 CPU 上推理（单样本约 0.2 s 量级）。
