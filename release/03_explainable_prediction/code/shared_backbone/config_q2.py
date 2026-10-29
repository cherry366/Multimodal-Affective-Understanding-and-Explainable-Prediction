# -*- coding: utf-8 -*-
"""模块二 统一配置（模态缺失条件下的鲁棒情感预测）

约定：
  - 输入版本统一使用 标准化特征aligned_50.pkl / 缺失测试集对齐版本
  - 三模态统一为 (L=50, d) 的词级对齐序列
  - 缺失位置特征置 0 且 mask=0（继承模块一 的 0+mask 闭环）
"""
import os

ROOT = os.path.abspath(os.environ.get("Q2_ROOT")
                       or os.path.join(os.path.dirname(__file__), "..", ".."))
Q2 = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "q2_output")
CKPT_OUT = os.path.join(OUT, "ckpt")
FIGS = os.path.join(OUT, "figs")
for _d in (OUT, CKPT_OUT, FIGS):
    os.makedirs(_d, exist_ok=True)


def _has_pt(d):
    return os.path.isdir(d) and any(f.endswith(".pt") for f in os.listdir(d))


# 模块二的发布权重通常位于 release/02_robust_prediction/weights/
_ROOT_CANDS = (
    os.path.join(ROOT, "weights"),
    os.path.abspath(os.path.join(ROOT, "..", "02_robust_prediction", "weights")),
)
CKPT = CKPT_OUT if _has_pt(CKPT_OUT) else next(
    (d for d in _ROOT_CANDS if _has_pt(d)), CKPT_OUT)

DATA = os.path.join(ROOT, "data")
P2_ALIGNED = os.path.join(DATA, "standard_features", "aligned_50.pkl")
LABEL_XLSX = os.path.join(DATA, "standard_features", "label.xlsx")
A3_DIR = os.path.join(DATA, "missing_modality", "aligned")

MODS = ["text", "audio", "vision"]
DIM = {"text": 768, "audio": 74, "vision": 35}
L = 50

# ---- 网络 ----
HID = 128
NHEAD = 4
NLAYER = 2
FFN = 256
DROPOUT = 0.1

# ---- 训练 ----
SEEDS = [42, 2024, 7]
BATCH = 64
EPOCHS = 40
PATIENCE = 10
LR = 1e-4
WD = 1e-2
LAMBDA_REG = 0.3          # 强度回归损失权重
MU_RECON = 0.1            # 跨模态重建辅助损失权重（3种子验证后取稳健默认值）
CLS_WEIGHTED = True       # 类别逆频加权（应对 Positive/Neutral/Negative 不平衡）

# ---- 最终方案新增组件（MoDE-Former+）----
USE_MOD_EMB = True        # 模态类型嵌入 e^(m)（思想来源 MA-CMT）
N_EXPERTS = 2             # 每模态专家数（EMOE 模态内路由，思想来源 EMOE/CVPR2025）
LAMBDA_KD = 0.5           # 知识蒸馏损失权重（3种子验证后取稳健默认值，单种子网格搜索过拟合）
LAMBDA_INV = 0.1          # 不变性约束损失权重（思想来源 因果不变表示/ACL2026）
LAMBDA_NLL = 0.1          # 不确定性校准损失权重（3种子验证后取稳健默认值）
KD_TAU = 3.0              # 蒸馏温度

# ---- 训练期缺失增强（对齐缺失测试集实测分布，并覆盖更广）----
AUG_R_RANGE = (0.10, 0.40)      # 总缺失率区间
AUG_K_CHOICES = [1, 2, 3, 4]    # 段数
AUG_K_PROB = [0.35, 0.30, 0.25, 0.10]
AUG_SEG_MAX = 8                 # 单段最大长度（格）
CLEAN_RATIO = 0.25              # 每个 batch 至少 25% 完整样本
P_SUBSET = {0: 0.20, 1: 0.40, 2: 0.30, 3: 0.10}   # 缺失模态个数分布
P_INDEP = 0.30                  # 以该概率使用"各模态独立缺失"形态

# ---- 缺失测试集同分布缺失（本机实测：r≈0.195、段长1~4、三模态同位置）----
A3_R = 0.195
A3_K_CHOICES = [1, 2, 3, 4]
A3_SEG_MAX = 4
A3_INDEP = False

# ---- 评价 ----
TAU_GRID = [round(x, 2) for x in [i * 0.05 for i in range(21)]]   # 中性带宽阈值搜索
N_EVAL_DRAWS = 5                # 缺失评估的重复抽样次数（取均值）

# ---- v2 改进（MoDE-Former+ v2）----
LAMBDA_CTR = 0.1           # 对比式不变性损失（InfoNCE）权重
CTR_TAU = 0.1              # 对比损失温度
KD_ANNEAL = False          # 蒸馏权重保持常数（与 v1 一致）
WARMUP_EPOCHS = 3          # 学习率预热 epoch 数
EMA_DECAY = 0.0            # 0=禁用 EMA（用实时最优权重，与 v1 一致）
FUSION_DROPOUT = 0.0       # 融合层 dropout（0=禁用）
COV_PRIOR_INIT = 0.5       # 覆盖率先验 β 初值（较温和）
USE_MOE_FEATS = True       # MoE 路由器是否加入缺失率/缺失段数
MIN_EPOCHS = 5              # 早停/最优 epoch 的起始保护（避免 MAE 尖峰期）
SCORE_MAE_W = 0.3           # 选参分数中 MAE 的权重（平衡分类与回归）

# ---- λ 网格搜索（坐标下降，验证集选参）----
GRID_KD = [0.1, 0.3, 0.5, 0.7, 1.0]
GRID_CTR = [0.05, 0.1, 0.2, 0.3]
GRID_NLL = [0.05, 0.1, 0.3]
GRID_REG = [0.1, 0.3, 0.5]
GRID_REC = [0.05, 0.1, 0.2]

# ---- 连续 vs 分散缺失对照 ----
CVD_RATES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
CVD_SEG_COUNT = 6          # 分散形态的段数
CVD_SEG_MAX = 4            # 分散形态单段最大长度
