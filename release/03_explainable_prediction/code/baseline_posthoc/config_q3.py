# -*- coding: utf-8 -*-
"""模块三 统一配置（可解释性多模态情感预测）。

复用模块二 的编码/融合骨干（model_q2.Q2Net / config_q2 常量），
模块三 场景为"三模态信息完整"，训练不做缺失增强，且去掉覆盖率先验（无缺失）。
输出统一写到 q3_output。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.environ.get("Q2_ROOT") or os.path.join(HERE, "..", ".."))
SHARED = os.path.join(HERE, "..", "shared_backbone")
for p in (SHARED, HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import config_q2 as C2  # noqa: E402

MODS = C2.MODS                  # ["text","audio","vision"]
DIM = C2.DIM                    # {768,74,35}
L = C2.L                        # 50
SEEDS = C2.SEEDS                # [42,2024,7]
BATCH = C2.BATCH
EPOCHS = C2.EPOCHS
PATIENCE = C2.PATIENCE
LR = C2.LR
WD = C2.WD
HID = C2.HID
NHEAD = C2.NHEAD
NLAYER = C2.NLAYER
FFN = C2.FFN
DROPOUT = C2.DROPOUT
TAU_GRID = C2.TAU_GRID
N_EVAL_DRAWS = C2.N_EVAL_DRAWS
WARMUP_EPOCHS = C2.WARMUP_EPOCHS
MIN_EPOCHS = C2.MIN_EPOCHS
SCORE_MAE_W = C2.SCORE_MAE_W
CLS_WEIGHTED = C2.CLS_WEIGHTED

# 损失权重（模块三：完整模态，去 KD/CTR/覆盖率先验）
LAMBDA_REG = 0.3
LAMBDA_REC = 0.1
LAMBDA_NLL = 0.1

# 数据路径
P2_ALIGNED = C2.P2_ALIGNED
LABEL_XLSX = C2.LABEL_XLSX
DATA = os.path.join(ROOT, "data")
A4_DIR = os.path.join(DATA, "explainable", "aligned")


def _has_pt(d):
    return os.path.isdir(d) and any(f.endswith(".pt") for f in os.listdir(d))


# 模块二权重：本地输出 → 本模块 weights/ → 模块二的发布 weights/
_Q2_CANDS = (
    C2.CKPT_OUT,
    os.path.join(ROOT, "weights", "init"),
    os.path.abspath(os.path.join(ROOT, "..", "02_robust_prediction", "weights")),
)
Q2_CKPT = next((d for d in _Q2_CANDS if _has_pt(d)), C2.CKPT_OUT)

# 输出路径
Q3_OUT = os.path.join(ROOT, "q3_output")
Q3_CKPT_OUT = os.path.join(Q3_OUT, "ckpt")
Q3_CKPT = Q3_CKPT_OUT if _has_pt(Q3_CKPT_OUT) else (
    os.path.join(ROOT, "weights", "baseline_posthoc")
    if _has_pt(os.path.join(ROOT, "weights", "baseline_posthoc")) else Q3_CKPT_OUT)
Q3_FIGS = os.path.join(Q3_OUT, "figs")
Q3_FRAMES = os.path.join(Q3_OUT, "frames")
for _d in (Q3_OUT, Q3_CKPT_OUT, Q3_FIGS, Q3_FRAMES):
    os.makedirs(_d, exist_ok=True)

# 可解释性超参
IG_STEPS = 32              # Integrated Gradients 步数
IG_BASELINE = "zero"       # 基线：零向量
TOP_K = 5                  # 每模态 top-k 关键位置
SHAPLEY_TAU = 1.0          # Shapley 作用占比温度
