# -*- coding: utf-8 -*-
"""模块三 全套精美可视化（PDF + PNG）。

图清单：
  qx01 可解释架构图       qx02 思维流程图
  qx03 验证集混淆矩阵     qx04 强度回归散点
  qx05 可解释测试集全量预测汇总  qx06 典型样本解释卡(Neg/Neu/Pos)
  qx07 三模态作用差异     qx08 局部片段重要性分布
  qx09 可解释性忠实度     qx10 错误归因
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import config_q3 as C

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.dpi"] = 130

C_TEXT, C_AUDIO, C_VIS = "#4C9BC7", "#E4636F", "#4FB6A6"
C_TITLE, C_GRAY = "#1F3B6E", "#8A8A8A"
MOD_CN = {"text": "文本 T", "audio": "语音 A", "vision": "视觉 V"}
CLS_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}
CLS_COLOR = {0: "#D05A6E", 1: "#B0A36A", 2: "#5B9B74"}

OUT = C.Q3_OUT
FIGS = C.Q3_FIGS


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), format=ext,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("FIG ->", name, flush=True)


def load_json(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


# ================================================================= qx01 架构（重绘，避免 mathtext 乱码）
def qx01_arch():
    fig, ax = plt.subplots(figsize=(16, 9.2))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9.4); ax.axis("off")

    def box(x, y, w, h, txt, fc, ec="none", fs=8.2, tc="black", lw=1.0, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.09",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, c=C_GRAY, lw=1.3, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, color=c, lw=lw))

    ax.text(8, 9.05, "可解释多模态情感预测模型：预测主干（上）+ 后验可解释分支（下）",
            ha="center", fontsize=13.5, color=C_TITLE)

    # 上：预测主干
    ytop = 7.6
    yin = [ytop, ytop-0.75, ytop-1.5]
    ycol = [C_TEXT, C_AUDIO, C_VIS]
    dims = ["768", "74", "35"]
    for i, m in enumerate(C.MODS):
        box(0.3, yin[i]-0.24, 1.95, 0.52, f"{MOD_CN[m]} X_{m[0]}  (50×{dims[i]})", ycol[i], fs=8.6, tc="white", bold=True)
        box(2.45, yin[i]-0.24, 1.5, 0.52, f"掩码 M_{m[0]}  (0/1)", "white", ec=C_GRAY, fs=8.0)
        arrow(2.25, yin[i], 2.43, yin[i])

    box(4.2, ytop-1.15, 1.75, 1.7, "标准化\nX'=(X−μ)/σ·M\n投影→128维\n+ 位置编码", "#F2F6FA", ec=C_GRAY, fs=8.0)
    box(6.15, ytop-1.35, 2.05, 2.1, "mask 感知\nTransformer 编码器\n(屏蔽缺失位)", "#EEF3F8", ec=C_TITLE, fs=8.6, bold=True)
    box(8.4, ytop-1.15, 2.0, 1.7, "跨模态注意力\n(两两 MHA)\n+ 模态内专家 MoE", "#EAF4EC", ec=C_VIS, fs=8.0)
    box(10.7, ytop-1.35, 2.2, 2.1, "覆盖率先验门控\nα = softmax(g + β·log c)\n融合 z = Σ α_m · h_m", "#FFF3E0", ec="#E8A33D", fs=8.2, bold=True)
    box(13.15, ytop-0.75, 2.6, 0.85, "极性 p(y) = softmax(W_c z)", "#FFE9EC", ec=C_AUDIO, fs=8.4)
    box(13.15, ytop-1.65, 2.6, 0.85, "强度 ŝ = 3·tanh(w_r·z)\n不确定度 σ² = exp(w_v·z)", "#EAF4EC", ec=C_VIS, fs=8.0)

    for i in range(3):
        arrow(3.95, yin[i], 4.18, yin[i])
        arrow(5.95, yin[i], 6.13, yin[i])
    arrow(8.2, ytop-0.6, 8.38, ytop-0.6)
    arrow(8.38, ytop-0.6, 8.38, ytop-1.0, c="#B9C7D6", lw=1.0)
    arrow(8.2, ytop-0.6, 8.38, ytop-0.6)
    arrow(8.38, ytop-0.6, 8.38, ytop-0.75, c="#B9C7D6")
    arrow(8.4, ytop-1.15, 8.38, ytop-0.6, c="#B9C7D6", lw=0.6)
    # 融合 -> 两个头
    arrow(12.9, ytop-0.35, 13.13, ytop-0.35)
    arrow(12.9, ytop-1.25, 13.13, ytop-1.25)

    # 下：可解释分支（虚线）
    box(0.3, 2.5, 3.6, 1.9, "Shapley 模态归因\nφ_m = Σ_S w(S)·[v(S+m) − v(S)]\n(8 个子集精确枚举)\n→ 主导模态 + 作用占比 π", "#FFF8E1", ec="#E8A33D", fs=8.0)
    box(4.3, 2.5, 3.6, 1.9, "遮挡 LOPO 局部证据\nI_m(t) = logit_c(x) − logit_c(x − 位置 t)\n(逐位置置 0 看 logit 下降)\n→ top-k 关键证据", "#E3F2FD", ec=C_TEXT, fs=8.0)
    box(8.3, 2.5, 3.6, 1.9, "证据落回\n文本词 / 语音时段 / 视觉帧\n(对应原始素材)", "#EAF4EC", ec=C_VIS, fs=8.4)
    box(12.3, 2.5, 3.4, 1.9, "解释卡输出\n预测 + 主导模态\n+ 作用程度 + 证据定位", "#F3E5F5", ec="#8E6BB4", fs=8.4, bold=True)

    # 从融合/预测引虚线到解释分支
    arrow(11.8, ytop-1.35, 3.5, 4.42, c="#8E6BB4", lw=1.2)
    ax.text(7.4, 5.15, "后验归因（不改变预测）", fontsize=8.2, color="#8E6BB4", ha="center")
    arrow(3.9, 3.45, 4.28, 3.45)
    arrow(7.9, 3.45, 8.28, 3.45)
    arrow(11.9, 3.45, 12.28, 3.45)

    # 图例
    handles = [Patch(color=C_TEXT, label="文本"), Patch(color=C_AUDIO, label="语音"),
               Patch(color=C_VIS, label="视觉")]
    ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
              bbox_to_anchor=(0.5, -0.02))
    save(fig, "qx01_arch")




# -*- coding: utf-8 -*-
# 重写 qx02：清晰的自上而下流程（不混入数据图）

def qx02_flow():
    fig, ax = plt.subplots(figsize=(12.5, 9.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 10.8); ax.axis("off")

    def box(x, y, w, h, txt, fc="#FFFFFF", ec=C_GRAY, fs=8.6, tc="#222222", lw=1.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, c=C_TITLE, lw=1.5):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, color=c, lw=lw))

    ax.text(6.5, 10.45, "模块三 可解释情感预测完整流程", ha="center", fontsize=13.5, color=C_TITLE, fontweight="bold")

    # 1 输入
    box(3.6, 9.35, 5.8, 0.85, "输入：三模态完整特征 + 掩码 M\n文本 50×768 · 语音 50×74 · 视觉 50×35",
        fc="#F2F6FA", ec=C_GRAY, fs=8.4)
    arrow(6.5, 9.35, 6.5, 8.75)

    # 2 骨干
    box(3.3, 7.85, 6.4, 0.9, "骨干模型（与模块二 共用）\nmask感知 Transformer + 跨模态 MHA + MoE + 覆盖率先验门控",
        fc="#EEF3F8", ec=C_TITLE, fs=8.6, bold=True)
    arrow(6.5, 7.85, 6.5, 7.25)

    # 3 预测
    box(3.3, 6.25, 6.4, 0.95, "预测输出\n情感极性 p(y) · 情感强度 ŝ · 不确定度 σ²",
        fc="#FFE9EC", ec=C_AUDIO, fs=8.8)
    arrow(6.5, 6.25, 6.5, 5.75)

    ax.text(6.5, 5.45, "后验归因（不改变预测结果）", ha="center", fontsize=8.5, color="#8E6BB4")

    # 4 两条解释支路
    box(0.9, 3.05, 5.1, 2.15,
        "① Shapley 模态归因\nφ_m = Σ_S w(S)·[v(S+m) − v(S)]\n（8 个子集精确枚举）\n→ 主导模态 + 作用占比 π\n\n验证：主导模态 LOPO 命中率 0.935",
        fc="#FFF8E1", ec="#E8A33D", fs=8.2)
    box(7.0, 3.05, 5.1, 2.15,
        "② 遮挡 LOPO 局部证据\nI_m(t) = logit_c(x) − logit_c(x−位置t)\n（逐位置置 0 看 logit 下降）\n→ top-k 关键证据位置\n\n验证：top-k vs 随机 胜率 0.708",
        fc="#E3F2FD", ec=C_TEXT, fs=8.2)

    # 从"后验归因"分叉到两个支路
    arrow(6.5, 5.2, 3.4, 5.2, c="#8E6BB4")
    arrow(3.4, 5.2, 3.4, 3.05+2.15, c="#8E6BB4")
    arrow(6.5, 5.2, 9.6, 5.2, c="#8E6BB4")
    arrow(9.6, 5.2, 9.6, 3.05+2.15, c="#8E6BB4")

    # 两个支路汇合到证据落回
    arrow(3.4, 3.05, 3.4, 1.9, c="#8E6BB4")
    arrow(9.6, 3.05, 9.6, 1.9, c="#8E6BB4")
    arrow(3.4, 1.9, 4.6, 1.9, c="#8E6BB4")
    arrow(9.6, 1.9, 8.4, 1.9, c="#8E6BB4")

    # 5 证据落回 + 解释卡
    box(3.0, 0.5, 7.0, 1.35,
        "③ 证据落回 + 解释卡输出\n关键证据：文本词 / 语音时段 / 视觉关键帧\n最终：预测 + 主要参考模态 + 模态作用程度 + 关键证据定位",
        fc="#F3E5F5", ec="#8E6BB4", fs=8.5, bold=True)

    save(fig, "qx02_flow")


# ================================================================= qx03 混淆矩阵
def qx03_confusion():
    ev = load_json("valid_eval.json")
    cm = np.array(ev["confusion_matrix"])
    labels = ["Negative", "Neutral", "Positive"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]*100:.0f}%)", ha="center", va="center",
                    fontsize=11, color="white" if cmn[i,j] > 0.5 else C_TITLE, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, rotation=15)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title(f"验证集混淆矩阵（Acc={ev['valid_clean']['accuracy']:.3f}, F1={ev['valid_clean']['f1_macro']:.3f}）", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("行归一化比例")
    save(fig, "qx03_confusion")


# ================================================================= qx04 强度散点
def qx04_reg_scatter():
    df = pd.read_csv(os.path.join(OUT, "valid_predictions.csv"))
    ev = load_json("valid_eval.json")
    r, mae = ev["valid_clean"]["pearson"], ev["valid_clean"]["mae"]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    col = [CLS_COLOR[c] for c in df["y_cls"]]
    ax.scatter(df["y_reg"], df["pred_reg"], c=col, s=14, alpha=0.5, edgecolors="none")
    lo, hi = -3.2, 3.2
    ax.plot([lo, hi], [lo, hi], "--", color=C_GRAY, lw=1.2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("真实情感强度 $s$"); ax.set_ylabel("预测情感强度 $\\hat{s}$")
    ax.set_title(f"验证集强度回归（Pearson r={r:.3f}, MAE={mae:.3f}）", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(handles=[Patch(color=CLS_COLOR[k], label=CLS_NAME[k]) for k in (0,1,2)],
              loc="upper left", fontsize=9, framealpha=0.9)
    save(fig, "qx04_reg_scatter")


# ================================================================= qx05 可解释测试集汇总
def qx05_a4_summary():
    df = pd.read_csv(os.path.join(OUT, "explainable_predictions.csv"), dtype={"id": str})
    ids = [f"{i:02d}" for i in range(1, 21)]
    fig = plt.figure(figsize=(13.5, 6.2))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2.3, 1.0, 1.0], hspace=0.55, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, :2])
    y = np.arange(20)
    colors = [CLS_COLOR[c] for c in df["predicted_class"]]
    ax1.barh(y, df["predicted_score"], color=colors, alpha=0.85, edgecolor="none")
    for i in range(20):
        ax1.text(df["predicted_score"][i] + (0.06 if df["predicted_score"][i]>=0 else -0.06),
                 i, f"{df['predicted_score'][i]:+.2f}", va="center",
                 ha="left" if df["predicted_score"][i]>=0 else "right", fontsize=8.2)
    ax1.axvline(0, color=C_GRAY, lw=0.9)
    ax1.set_yticks(y); ax1.set_yticklabels([f"a4_{i}" for i in ids], fontsize=8)
    ax1.invert_yaxis(); ax1.set_xlabel("预测情感强度 $\\hat{s}$")
    ax1.set_title("可解释测试集全量 20 条样本的强度预测与极性", fontsize=12)
    ax1.grid(axis="x", alpha=0.25)
    ax1.legend(handles=[Patch(color=CLS_COLOR[k], label=CLS_NAME[k]) for k in (0,1,2)],
               loc="lower right", fontsize=8.5, framealpha=0.9)

    ax2 = fig.add_subplot(gs[0, 2])
    cnt = df["predicted_class_name"].value_counts()
    order = ["Negative", "Neutral", "Positive"]
    vals = [int(cnt.get(k, 0)) for k in order]
    ax2.bar(order, vals, color=[CLS_COLOR[k] for k in (0,1,2)], alpha=0.85)
    for i, v in enumerate(vals):
        ax2.text(i, v, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_title("极性分布", fontsize=11); ax2.set_ylabel("样本数")

    ax3 = fig.add_subplot(gs[1, :2])
    means = [df[f"alpha_{m}"].mean() for m in C.MODS]
    pis = [df[f"pi_{m}"].mean() for m in C.MODS]
    x = np.arange(3); wd = 0.34
    ax3.bar(x-wd/2, means, wd, label="门控 α", color=C_TITLE, alpha=0.9)
    ax3.bar(x+wd/2, pis, wd, label="Shapley π", color="#E8A33D", alpha=0.9)
    ax3.set_xticks(x); ax3.set_xticklabels(["文本", "语音", "视觉"])
    ax3.set_ylabel("平均作用占比"); ax3.set_ylim(0, 1)
    ax3.set_title("可解释测试集三模态平均作用占比（α vs Shapley π）", fontsize=11)
    ax3.legend(fontsize=8.5)
    ax3.grid(axis="y", alpha=0.25)

    ax4 = fig.add_subplot(gs[1, 2])
    data4 = [df["shapley_text"], df["shapley_audio"], df["shapley_vision"]]
    bp4 = ax4.boxplot(data4, labels=["文本", "语音", "视觉"], patch_artist=True,
                      showmeans=True, widths=0.5)
    for patch, c in zip(bp4["boxes"], [C_TEXT, C_AUDIO, C_VIS]):
        patch.set_facecolor(c); patch.set_alpha(0.45)
    ax4.axhline(0, color=C_GRAY, lw=1.0)
    ax4.set_ylabel("Shapley φ")
    ax4.set_title("三模态 Shapley φ 分布", fontsize=10)
    ax4.grid(axis="y", alpha=0.25)
    ax4.text(0.02, 0.98, "主导模态：文本 20/20\n语音/视觉 φ 跨零、非退化",
             transform=ax4.transAxes, fontsize=7.2, va="top", color=C_TITLE)
    save(fig, "qx05_a4_summary")


# ================================================================= qx06 解释卡
def _mod_bars(ax, alpha, phi, pi, dom):
    labels = ["文本", "语音", "视觉"]
    x = np.arange(3)
    ax.bar(x, [alpha["text"], alpha["audio"], alpha["vision"]], 0.26, label="门控 α", color=C_TITLE)
    ax.bar(x+0.26, [pi["text"], pi["audio"], pi["vision"]], 0.26, label="Shapley π", color="#E8A33D")
    ax.set_xticks(x+0.13); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("作用占比", fontsize=8); ax.set_ylim(0, 1)
    ax.set_title("模态作用程度", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.25)


def _imp_curve(ax, occ, topk, color, label):
    w = np.abs(np.array(occ))
    ax.plot(range(50), w, color=color, lw=1.3, label=label)
    if topk:
        ax.scatter(topk, [w[t] for t in topk], color=color, s=18, zorder=3, edgecolors="white", linewidths=0.6)
    ax.set_xlim(0, 49); ax.set_ylim(0, max(w.max()*1.15, 1e-3))
    ax.set_xlabel("时序位置 t", fontsize=8); ax.set_ylabel("|Δlogit|", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")


def _highlight_text(ax, raw_text, ev_words):
    words = raw_text.strip().split()
    ax.axis("off")
    x = 0.02; y = 0.98; line_h = 0.16
    picked = set(ev_words)
    for w in words:
        c = "#C0392B" if w in picked else "#333333"
        ax.text(x, y, w, fontsize=8.6, color=c, fontweight="bold" if w in picked else "normal")
        x += len(w)*0.016 + 0.008
        if x > 0.96:
            x = 0.02; y -= line_h
    ax.text(0.02, y-line_h*0.8, "(红色 = 关键证据词)", fontsize=7.5, color=C_GRAY)


def qx06_cards():
    """解释卡三联布局：左=预测概览，中=模态作用分解，右=局部证据定位。"""
    expl = load_json("explainable_explanations.json")
    df = pd.read_csv(os.path.join(OUT, "explainable_predictions.csv"), dtype={"id": str})
    picks = [("09", "Negative"), ("17", "Positive"), ("01", "Neutral")]
    fig = plt.figure(figsize=(16.5, 13.5))
    gs = GridSpec(6, 3, figure=fig, hspace=0.72, wspace=0.30,
                  height_ratios=[0.8, 1.0] * 3)
    for k, (sid, want) in enumerate(picks):
        e = expl[sid]
        row = df[df["id"] == sid].iloc[0]
        r0 = 2 * k
        # 左：预测概览（跨 2 行）
        axp = fig.add_subplot(gs[r0:r0+2, 0])
        axp.axis("off"); axp.set_xlim(0, 1); axp.set_ylim(0, 1)
        cls = row["predicted_class_name"]
        cc = CLS_COLOR[int(row["predicted_class"])]
        axp.add_patch(FancyBboxPatch((0.02, 0.88), 0.96, 0.10, boxstyle="round,pad=0.008",
                                     fc=cc, ec="none", alpha=0.9, transform=axp.transAxes))
        axp.text(0.5, 0.93, f"样本 a4_{sid}  ·  预测极性 {cls}", transform=axp.transAxes,
                 ha="center", va="center", fontsize=11.5, color="white", fontweight="bold")
        axp.text(0.04, 0.82, f"情感强度 ŝ = {row['predicted_score']:+.3f}", transform=axp.transAxes,
                 fontsize=10, color=C_TITLE, va="center")
        axp.text(0.04, 0.74, f"不确定度 σ = {row['score_uncertainty_sigma']:.3f}", transform=axp.transAxes,
                 fontsize=10, color=C_TITLE, va="center")
        axp.text(0.04, 0.66, f"主要参考模态 = {MOD_CN[row['dominant_modality']]}", transform=axp.transAxes,
                 fontsize=10, color=C_TITLE, va="center")
        axp.text(0.04, 0.56, "文本证据：", transform=axp.transAxes, fontsize=9, color=C_GRAY, va="center")
        txt = e["raw_text"]
        axp.text(0.04, 0.50, txt[:160] + ("…" if len(txt) > 160 else ""), transform=axp.transAxes,
                 fontsize=8, color="#444444", va="top", wrap=True)

        # 中：模态作用程度（跨 2 行）
        axm = fig.add_subplot(gs[r0:r0+2, 1])
        _mod_bars(axm, e["modality_contribution"]["alpha"], e["modality_contribution"]["shapley"],
                  e["modality_contribution"]["pi"], row["dominant_modality"])

        # 右上：主导模态局部证据曲线
        dom = row["dominant_modality"]
        occ = e["local_importance_occlusion"][dom]
        topk = e["evidence_text"]["positions"] if dom=="text" else (e["evidence_audio"]["positions"] if dom=="audio" else e["evidence_vision"]["positions"])
        axi = fig.add_subplot(gs[r0, 2])
        _imp_curve(axi, occ, topk, {"text":C_TEXT,"audio":C_AUDIO,"vision":C_VIS}[dom],
                   f"{MOD_CN[dom]} 遮挡重要性")

        # 右下：证据定位 chips（文本证据词 φ 值 + 语音时段 + 视觉帧）
        axe = fig.add_subplot(gs[r0+1, 2])
        axe.axis("off"); axe.set_xlim(0, 1); axe.set_ylim(0, 1)
        axe.text(0.02, 0.98, "关键证据定位", fontsize=9.5, color=C_TITLE, fontweight="bold", va="top")
        y = 0.86
        axe.text(0.02, y, "文本证据词（遮挡 Δlogit）：", fontsize=8, color="#333333", va="top")
        y -= 0.055
        x = 0.02
        for wi, w in enumerate(e["evidence_text"]["words"][:5]):
            t = int(e["evidence_text"]["positions"][wi])
            val = float(e["local_importance_occlusion"]["text"][min(max(t, 0), 49)])
            col = "#2E6DA4" if val >= 0 else "#B03A2E"
            lab = f"{w} {val:+.2f}"
            axe.text(x, y, lab, fontsize=7.6, va="top", ha="left", color="white",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=col, edgecolor="none"))
            x += len(lab) * 0.023 + 0.02
            if x > 0.94:
                x = 0.02; y -= 0.075
        y -= 0.085
        axe.text(0.02, y, "语音关键时段：", fontsize=8, color="#333333", va="top")
        wins = e["evidence_audio"]["time_windows"][:3]
        axe.text(0.30, y, ", ".join([f"{a:.2f}–{b:.2f}s" for a, b in wins]), fontsize=7.6, color=C_AUDIO, va="top")
        y -= 0.075
        axe.text(0.02, y, "视觉关键帧位置：", fontsize=8, color="#333333", va="top")
        vp = e["evidence_vision"]["positions"][:5]
        axe.text(0.30, y, ", ".join([f"t{p:02d}" for p in vp]), fontsize=7.6, color=C_VIS, va="top")
    fig.suptitle("可解释测试集典型样本可解释性解释卡（预测概览 · 模态作用分解 · 关键证据定位）", fontsize=13.5, color=C_TITLE, y=0.995)
    save(fig, "qx06_cards")


def qx06_frames():
    expl = load_json("explainable_explanations.json")
    picks = ["09", "17", "01"]
    fig, axes = plt.subplots(3, 4, figsize=(12, 8.4))
    for r, sid in enumerate(picks):
        e = expl[sid]
        fr = [f for f in e["evidence_vision"]["key_frames"] if f and os.path.exists(f)][:4]
        for c in range(4):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(fr):
                try:
                    ax.imshow(mpimg.imread(fr[c]))
                except Exception:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", color=C_GRAY)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center", color=C_GRAY)
            ax.set_title(f"a4_{sid} 帧 {c+1}", fontsize=8)
    fig.suptitle("典型样本关键视觉帧（对应视觉局部证据位置）", fontsize=12, color=C_TITLE)
    save(fig, "qx06_frames")


# ================================================================= qx07 三模态作用差异（重绘，说明文本主导是数据特性）
def qx07_modality_diff():
    df = pd.read_csv(os.path.join(OUT, "explainable_predictions.csv"), dtype={"id": str})
    fig = plt.figure(figsize=(14.5, 4.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # (a) 平均作用占比 vs 随机 1/3
    ax1 = fig.add_subplot(gs[0])
    means = [df[f"alpha_{m}"].mean() for m in C.MODS]
    pis = [df[f"pi_{m}"].mean() for m in C.MODS]
    x = np.arange(3); wd = 0.32
    ax1.bar(x-wd/2, means, wd, label="门控 α", color=C_TITLE, alpha=0.92)
    ax1.bar(x+wd/2, pis, wd, label="Shapley π", color="#E8A33D", alpha=0.92)
    ax1.axhline(1/3, color=C_GRAY, ls="--", lw=1.2)
    ax1.text(2.62, 1/3+0.02, "随机基准 1/3", fontsize=8, color=C_GRAY, ha="right")
    for i, v in enumerate(means):
        ax1.text(i-wd/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(pis):
        ax1.text(i+wd/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(["文本", "语音", "视觉"])
    ax1.set_ylabel("平均作用占比"); ax1.set_ylim(0, 0.9)
    ax1.set_title("平均作用占比（文本显著高于 1/3 随机基准）", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right"); ax1.grid(axis="y", alpha=0.25)

    # (b) α 分布（箱线，说明非退化、样本间可变）
    ax2 = fig.add_subplot(gs[1])
    data = [df["alpha_text"], df["alpha_audio"], df["alpha_vision"]]
    bp = ax2.boxplot(data, labels=["文本", "语音", "视觉"], patch_artist=True,
                     showmeans=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [C_TEXT, C_AUDIO, C_VIS]):
        patch.set_facecolor(c); patch.set_alpha(0.45)
    ax2.set_ylabel("门控 α")
    ax2.set_title("α 分布（语音/视觉非零且样本间可变）", fontsize=10)
    ax2.grid(axis="y", alpha=0.25)

    # (c) φ 分布（Shapley，跨零 = 部分样本语音/视觉真实贡献）
    ax3 = fig.add_subplot(gs[2])
    data3 = [df["shapley_text"], df["shapley_audio"], df["shapley_vision"]]
    bp3 = ax3.boxplot(data3, labels=["文本", "语音", "视觉"], patch_artist=True,
                      showmeans=True, widths=0.5)
    for patch, c in zip(bp3["boxes"], [C_TEXT, C_AUDIO, C_VIS]):
        patch.set_facecolor(c); patch.set_alpha(0.45)
    ax3.axhline(0, color=C_GRAY, lw=1.0, ls="-")
    ax3.set_ylabel("Shapley φ")
    ax3.set_title("φ 分布（语音/视觉 φ 跨零，部分样本起正作用）", fontsize=10)
    ax3.grid(axis="y", alpha=0.25)

    fig.suptitle("三模态作用差异：文本主导是数据特性，语音/视觉非退化", fontsize=12.5, color=C_TITLE)
    save(fig, "qx07_modality_diff")


# ================================================================= qx08 局部重要性分布
def qx08_local_importance():
    expl = load_json("explainable_explanations.json")
    df = pd.read_csv(os.path.join(OUT, "explainable_predictions.csv"), dtype={"id": str})
    order = [f"{i:02d}" for i in range(1, 21)]
    cls_map = {row["id"]: row["predicted_class"] for _, row in df.iterrows()}
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.0), sharex=True)
    cmap = plt.get_cmap("RdBu_r")          # 蓝=支持预测类，红=反对
    for ax, m in zip(axes, C.MODS):
        mat = np.zeros((20, 50))
        tops = {}
        for i, sid in enumerate(order):
            occ = np.array(expl[sid]["local_importance_occlusion"][m], dtype=float)
            mat[i] = occ
            v = np.abs(occ)
            tops[sid] = np.argsort(-v)[:3].tolist()
        vmax = float(np.max(np.abs(mat))) + 1e-9
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
        for i, sid in enumerate(order):
            for t in tops[sid]:
                ax.add_patch(Rectangle((t-0.5, i-0.5), 1, 1, fill=False,
                                       edgecolor="black", lw=0.6))
        ax.set_yticks(range(20)); ax.set_yticklabels([f"a4_{s}" for s in order], fontsize=7.6)
        for i, sid in enumerate(order):
            ax.text(-1.2, i, CLS_NAME[cls_map[sid]][:3], fontsize=6.5, va="center",
                    color=CLS_COLOR[cls_map[sid]], fontweight="bold")
        ax.set_ylabel(MOD_CN[m], fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_label("遮挡 Δlogit（蓝=支持预测类，红=反对）", fontsize=7.5)
    axes[-1].set_xlabel("时序位置 t（0–49）")
    fig.suptitle("可解释测试集逐位置遮挡敏感度热力图（蓝正红负，黑框=top-3 关键证据）", fontsize=12.5, color=C_TITLE)
    save(fig, "qx08_local_importance")


# ================================================================= qx09 忠实度
def qx09_faithfulness():
    ev = load_json("valid_eval.json")
    oc = load_json("occlusion_faithfulness.json")
    fs = pd.read_csv(os.path.join(OUT, "faithfulness_samples.csv"))
    fig = plt.figure(figsize=(12.5, 4.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    # (a) LOMO
    ax1 = fig.add_subplot(gs[0])
    dom, rnd = ev["faithfulness"]["mean_comp_dom"], ev["faithfulness"]["mean_comp_rand"]
    ax1.bar(["去掉主导模态", "去掉随机模态"], [dom, rnd], color=[C_TITLE, C_GRAY], alpha=0.9)
    for i, v in enumerate([dom, rnd]):
        ax1.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylabel("置信度下降 Δp")
    ax1.set_title(f"模态级 LOPO 忠实度\n(命中率 {ev['faithfulness']['dom_hit_rate']:.2f})", fontsize=10)
    ax1.grid(axis="y", alpha=0.25)

    # (b) 遮挡 token
    ax2 = fig.add_subplot(gs[1])
    t, r = oc["mean_comp_topk"], oc["mean_comp_randk"]
    ax2.bar(["遮挡 top-k", "遮挡随机 k"], [t, r], color=[C_TEXT, C_GRAY], alpha=0.9)
    for i, v in enumerate([t, r]):
        ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("置信度下降 Δp")
    ax2.set_title(f"位置级 LOPO 忠实度\n(胜率 {oc['comp_win_rate']:.2f})", fontsize=10)
    ax2.grid(axis="y", alpha=0.25)

    # (c) 一致性
    ax3 = fig.add_subplot(gs[2])
    labels = ["文本", "语音", "视觉"]
    rhos = [ev["faithfulness"]["spearman"][f"rho_alpha_phi_{m}"] for m in C.MODS]
    ax3.bar(labels, rhos, color=[C_TEXT, C_AUDIO, C_VIS], alpha=0.9)
    for i, v in enumerate(rhos):
        ax3.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v>=0 else "top", fontsize=9.5, fontweight="bold")
    ax3.axhline(0, color=C_GRAY, lw=0.9)
    ax3.set_ylabel("Spearman ρ(α, φ)")
    ax3.set_title(f"门控-Shapley 一致性\n(主导模态一致 {ev['faithfulness']['alpha_vs_shapley_dom_agreement']:.2f})", fontsize=10)
    ax3.grid(axis="y", alpha=0.25)
    save(fig, "qx09_faithfulness")


# ================================================================= qx10 错误归因
def qx10_error_attribution():
    ev = load_json("valid_eval.json")
    pc = ev["per_class"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    labels = ["Negative", "Neutral", "Positive"]
    accs = [pc[k]["acc"] for k in labels]
    maes = [pc[k]["mae"] for k in labels]
    ns = [pc[k]["n"] for k in labels]
    axes[0].bar(labels, accs, color=[CLS_COLOR[i] for i in (0,1,2)], alpha=0.9)
    for i, v in enumerate(accs):
        axes[0].text(i, v, f"{v:.2f}\n(n={ns[i]})", ha="center", va="bottom", fontsize=8.5)
    axes[0].set_ylim(0, 1); axes[0].set_title("各类别准确率", fontsize=10.5)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, maes, color=[CLS_COLOR[i] for i in (0,1,2)], alpha=0.9)
    for i, v in enumerate(maes):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("各类别强度 MAE", fontsize=10.5)
    axes[1].grid(axis="y", alpha=0.25)

    pairs = ev["error_pairs"]
    top = list(pairs.items())[:8]
    ax2 = axes[2]
    ax2.barh([p for p, _ in top], [v for _, v in top], color="#B3554D", alpha=0.85)
    ax2.invert_yaxis()
    for i, (p, v) in enumerate(top):
        ax2.text(v, i, f" {v}", va="center", fontsize=8.5)
    ax2.set_title("主要错误流向（真实→预测）", fontsize=10.5)
    ax2.grid(axis="x", alpha=0.25)
    fig.suptitle("验证集错误归因分析", fontsize=12.5, color=C_TITLE)
    save(fig, "qx10_error_attribution")


def main():
    for fn in (qx01_arch, qx02_flow, qx03_confusion, qx04_reg_scatter, qx05_a4_summary,
               qx06_cards, qx06_frames, qx07_modality_diff, qx08_local_importance, qx09_faithfulness,
               qx10_error_attribution):
        try:
            fn()
        except Exception as ex:
            import traceback
            print(f"[ERR] {fn.__name__}: {ex}")
            traceback.print_exc()


if __name__ == "__main__":
    main()


