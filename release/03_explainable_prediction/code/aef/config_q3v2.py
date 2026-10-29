# -*- coding: utf-8 -*-
"""模块三 新方案（v2）统一配置：原生证据加权融合（Ante-hoc Evidence-weighted Fusion, AEF）。

与 v1（复用 Q2 完整骨干 + 后验 Shapley/遮挡）的区别：
  v2 冻结 Q2 的 per-modality Transformer 编码器，用可学习证据权重 e_m(t) 做加权池化，
  使局部证据在前向中直接参与决策（前验/原生解释），仍复用跨模态注意力、MoE 与覆盖率先验门控。
输出统一写到 q3_v2_output，与后验基线（q3_output）完全隔离。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.environ.get("Q2_ROOT") or os.path.join(HERE, "..", ".."))
SHARED = os.path.join(HERE, "..", "shared_backbone")
BASE = os.path.join(HERE, "..", "baseline_posthoc")
for p in (SHARED, BASE, HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import config_q2 as C2  # noqa: E402

MODS = C2.MODS
DIM = C2.DIM
L = C2.L
SEEDS = C2.SEEDS
BATCH = C2.BATCH
EPOCHS = 20                    # 编码器冻结，只微调融合头，20 轮足够
PATIENCE = C2.PATIENCE
LR = C2.LR
WD = C2.WD
HID = C2.HID
NHEAD = C2.NHEAD
NLAYER = C2.NLAYER
FFN = C2.FFN
DROPOUT = C2.DROPOUT
TAU_GRID = C2.TAU_GRID
WARMUP_EPOCHS = C2.WARMUP_EPOCHS
MIN_EPOCHS = C2.MIN_EPOCHS
SCORE_MAE_W = C2.SCORE_MAE_W
CLS_WEIGHTED = C2.CLS_WEIGHTED

LAMBDA_REG = 0.3
LAMBDA_NLL = 0.1

# 冻结范围：只冻结 per-modality 编码器（proj + enc），其余融合头/证据头可训练
FREEZE_ENCODER = True

# 数据路径（复用模块二 / 可解释测试集）
P2_ALIGNED = C2.P2_ALIGNED
DATA = os.path.join(ROOT, "data")
A4_DIR = os.path.join(DATA, "explainable", "aligned")


def _has_pt(d):
    return os.path.isdir(d) and any(f.endswith(".pt") for f in os.listdir(d))


# 模块二热启动权重：本地输出 → 本模块 weights/init → 模块二的发布 weights/
_Q2_CANDS = (
    C2.CKPT_OUT,
    os.path.join(ROOT, "weights", "init"),
    os.path.abspath(os.path.join(ROOT, "..", "02_robust_prediction", "weights")),
)
Q2_CKPT = next((d for d in _Q2_CANDS if _has_pt(d)), C2.CKPT_OUT)

# 输出路径（v2 独立）
Q3V2_OUT = os.path.join(ROOT, "q3_v2_output")
Q3V2_CKPT_OUT = os.path.join(Q3V2_OUT, "ckpt")
_W_AEF = os.path.join(ROOT, "weights", "aef")
Q3V2_CKPT = Q3V2_CKPT_OUT if _has_pt(Q3V2_CKPT_OUT) else (
    _W_AEF if _has_pt(_W_AEF) else Q3V2_CKPT_OUT)
Q3V2_FIGS = os.path.join(Q3V2_OUT, "figs")
Q3V2_FRAMES = os.path.join(Q3V2_OUT, "frames")
for _d in (Q3V2_OUT, Q3V2_CKPT_OUT, Q3V2_FIGS, Q3V2_FRAMES):
    os.makedirs(_d, exist_ok=True)

TOP_K = 5
