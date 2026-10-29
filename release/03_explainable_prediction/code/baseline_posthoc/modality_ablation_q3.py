# -*- coding: utf-8 -*-
"""模块三 模态子集消融 + "文本主导"必要性/充分性论证图（qx15）。

目的：为"文本主导是数据特性而非模型缺陷"提供定量证据。
不改变模型与可解释测试集预测，仅评估不同模态子集下的验证集性能。
"""
import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

import config_q3 as C
import train_q2 as T2
from explain_q3 import load_models, ens_predict

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 10.5
plt.rcParams["figure.dpi"] = 130

C_TEXT, C_AUDIO, C_VIS = "#4C9BC7", "#E4636F", "#4FB6A6"
C_TITLE, C_GRAY = "#1F3B6E", "#8A8A8A"
MOD_CN = {"text": "文本 T", "audio": "语音 A", "vision": "视觉 V"}
OUT = C.Q3_OUT
FIGS = C.Q3_FIGS


def drop_modalities(Xn, Mn, keep):
    Xo = {k: Xn[k].copy() for k in C.MODS}
    Mo = {k: Mn[k].copy() for k in C.MODS}
    for m in C.MODS:
        if m not in keep:
            Xo[m] = np.zeros_like(Xn[m])
            Mo[m] = np.zeros_like(Mn[m])
    return Xo, Mo


def main():
    models, dt = load_models()
    dev = T2.device_of()
    Xv = dt["X"]["valid"]
    Mv = dt["M"]["valid"]
    yc = dt["items"]["valid"]["y_cls"]
    yr = dt["items"]["valid"]["y_reg"]
    tau = float(np.mean([ck["tau"] for ck, _ in models]))

    subsets = {}
    for r in (1, 2, 3):
        for combo in itertools.combinations(C.MODS, r):
            subsets[combo] = None

    results = {}
    for combo, _ in subsets.items():
        Xd, Md = drop_modalities(Xv, Mv, set(combo))
        P, R, _, _ = ens_predict(models, Xd, Md, dev)
        pred = T2.decision(P, R, tau, "band")
        results["+".join(combo)] = {
            "acc": float(accuracy_score(yc, pred)),
            "f1_macro": float(f1_score(yc, pred, average="macro")),
            "mae": float(np.abs(R - yr).mean()),
            "pearson": float(np.corrcoef(R, yr)[0, 1]),
        }

    # 必要性 = full acc - 去掉该模态后的 acc；充分性 = 只用该模态的 acc
    full = results["text+audio+vision"]["acc"]
    necessity = {
        "text": full - results["audio+vision"]["acc"],
        "audio": full - results["text+vision"]["acc"],
        "vision": full - results["text+audio"]["acc"],
    }
    sufficiency = {
        "text": results["text"]["acc"],
        "audio": results["audio"]["acc"],
        "vision": results["vision"]["acc"],
    }
    majority = float(np.bincount(yc).max() / len(yc))

    out = {
        "full": results["text+audio+vision"],
        "subsets": results,
        "majority_class_baseline_acc": majority,
        "necessity_drop_acc_loss": {m: round(necessity[m], 4) for m in C.MODS},
        "sufficiency_single_modality_acc": {m: round(sufficiency[m], 4) for m in C.MODS},
    }
    with open(os.path.join(OUT, "modality_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)

    # ---------------- qx15 图 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # (a) 子集精度
    ax = axes[0]
    keys = ["text", "audio", "vision", "audio+vision", "text+audio", "text+vision", "text+audio+vision"]
    labels = ["仅T", "仅A", "仅V", "A+V", "T+A", "T+V", "全模态"]
    vals = [results[k]["acc"] for k in keys]
    cols = [C_TEXT, C_AUDIO, C_VIS, C_GRAY, "#7B8DB3", "#7B8DB3", C_TITLE]
    bars = ax.bar(labels, vals, color=cols, alpha=0.92)
    ax.axhline(majority, color="#B3554D", ls="--", lw=1.4)
    ax.text(6.35, majority, f"多数类基线 {majority:.3f}", color="#B3554D", fontsize=8.5, va="bottom", ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.2)
    ax.set_ylim(0, max(vals) + 0.10)
    ax.set_ylabel("验证集 Accuracy")
    ax.set_title("(a) 模态子集性能：A/V 单独低于多数类，但补充到 T 后可提升", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    # (b) 必要性 vs 充分性
    ax = axes[1]
    x = np.arange(3)
    w = 0.36
    b1 = ax.bar(x - w/2, [necessity[m] for m in C.MODS], w, label="必要性（删除该模态的精度损失）", color=[C_TEXT, C_AUDIO, C_VIS], alpha=0.9)
    b2 = ax.bar(x + w/2, [sufficiency[m] for m in C.MODS], w, label="充分性（仅用该模态的精度）", color=[C_TEXT, C_AUDIO, C_VIS], alpha=0.42, hatch="//")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8.0)
    ax.set_xticks(x); ax.set_xticklabels(["文本 T", "语音 A", "视觉 V"])
    ax.set_ylabel("Accuracy 变化 / Accuracy")
    ax.set_title("(b) 必要性 vs 充分性：文本两高，语音/视觉为低必要性补充信息", fontsize=10)
    ax.legend(fontsize=8.2, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("模块三 文本主导的数据特性论证（模态必要性/充分性）", fontsize=12.5, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"qx15_modality_necessity.{ext}"),
                    format=ext, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("FIG -> qx15_modality_necessity", flush=True)


if __name__ == "__main__":
    main()
