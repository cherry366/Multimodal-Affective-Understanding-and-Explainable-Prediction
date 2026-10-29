# -*- coding: utf-8 -*-
"""模块三 最终方案（原生证据加权 AEF）关键可视化 + 与后验解释基线的对比图。"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

import config_q3v2 as C

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 10.5
plt.rcParams["figure.dpi"] = 130
C_TEXT, C_AUDIO, C_VIS = "#4C9BC7", "#E4636F", "#4FB6A6"
C_TITLE, C_GRAY = "#1F3B6E", "#8A8A8A"
MOD_CN = {"text": "文本 T", "audio": "语音 A", "vision": "视觉 V"}
CLS_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}
OUT, FIGS = C.Q3V2_OUT, C.Q3V2_FIGS
V1 = os.path.join(C.ROOT, "q3_output")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), format=ext, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("FIG ->", name, flush=True)


def load(name, base=OUT):
    with open(os.path.join(base, name), encoding="utf-8") as f:
        return json.load(f)


def qx2_01_arch():
    """AEF 架构建模图：上为预测主干，下为可解释性输出（原生解释为主口径，模型无关归因作互证）。"""
    fig, ax = plt.subplots(figsize=(16.2, 10.4))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10.4); ax.axis("off")

    def box(x, y, w, h, t, fc, ec="none", fs=8.4, tc="black", lw=1.0, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.09",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2, c=C_GRAY, lw=1.2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, color=c, lw=lw, linestyle=ls))

    ax.text(8, 10.08, "模块三 可解释多模态情感预测模型（原生证据加权融合 AEF）",
            ha="center", fontsize=14, color=C_TITLE)
    ax.text(8, 9.62, "上：预测主干（冻结模块二编码器 → 原生证据加权池化 → 跨模态融合与覆盖率先验门控 → 三头输出）\n"
                     "下：可解释性输出（原生解释为主口径，模型无关归因作互证）",
            ha="center", va="center", fontsize=9.2, color=C_GRAY, linespacing=1.6)

    # ---------------- 上：预测主干 ----------------
    cy = [8.32, 7.62, 6.92]
    cols = [C_TEXT, C_AUDIO, C_VIS]
    dims = ["768", "74", "35"]
    for i, m in enumerate(C.MODS):
        box(0.25, cy[i] - 0.31, 1.85, 0.62, f"{MOD_CN[m]} X (50×{dims[i]})",
            cols[i], fs=8.4, tc="white", bold=True)
        arr(2.10, cy[i], 2.35, cy[i])

    box(2.35, 6.55, 2.35, 2.40,
        "冻结模块二编码器\n逐模态 Transformer\n+ 缺失掩码 mask",
        "#E8EEF6", ec=C_GRAY, fs=8.2)
    for i in range(3):
        arr(4.70, cy[i], 5.05, cy[i])

    box(5.05, 6.55, 2.55, 2.40,
        "原生证据打分\ne_m(t) = w_e · tanh(W · h_m(t))\nsoftmax_t → ē_m(t)",
        "#F6E8EE", ec=C_AUDIO, fs=8.0)
    for i in range(3):
        arr(7.60, cy[i], 7.95, 7.62)

    box(7.95, 6.55, 2.55, 2.40,
        "证据加权池化\npool_m = Σ_t ē_m(t)·h_m(t)\n（局部证据在前向中参与决策）",
        "#F3EBDD", ec="#E8A33D", fs=8.0, bold=True)
    arr(10.50, 7.62, 10.85, 7.62, lw=1.4)

    box(10.85, 6.55, 2.45, 2.40,
        "跨模态注意力 + 专家 MoE\n覆盖率先验门控\nα = softmax(g + β·log c)\nz = Σ_m α_m · enriched_m",
        "#E4EFE7", ec=C_VIS, fs=7.9)
    arr(13.30, 8.42, 13.60, 8.42, lw=1.3)
    arr(13.30, 7.22, 13.60, 7.22, lw=1.3)

    box(13.60, 7.95, 2.15, 0.95, "极性\np(y) = softmax(W_c z)",
        "#FFE9EC", ec=C_AUDIO, fs=8.2)
    box(13.60, 6.75, 2.15, 0.95, "强度 与 不确定度\nŝ = 3·tanh(w_r·z)\nσ² = exp(w_v·z)",
        "#EAF4EC", ec=C_VIS, fs=7.6)

    # ---------------- 分隔 ----------------
    ax.plot([0.25, 15.75], [5.95, 5.95], ls="--", color="#B9C7D6", lw=1.1)
    ax.text(15.75, 6.04, "可解释性输出", ha="right", fontsize=9.3, color=C_GRAY)

    # 主干 -> 解释分支（虚线引线）
    arr(11.95, 6.55, 2.10, 5.42, c="#8E6BB4", lw=1.3, ls="--")
    arr(6.20, 6.55, 5.72, 5.42, c="#8E6BB4", lw=1.3, ls="--")
    ax.text(7.70, 6.30, "门控 α（作用程度）", fontsize=8.4, color="#8E6BB4", ha="center")
    ax.text(4.30, 6.22, "原生证据 ē_m(t)", fontsize=8.4, color="#8E6BB4", ha="center")

    # ---------------- 下-①：原生解释链（主口径） ----------------
    ax.text(0.25, 5.70, "① 原生解释链（前验可解释，与预测同源）", fontsize=10.5,
            color=C_TITLE, fontweight="bold", ha="left")
    y1, h1 = 3.75, 1.60
    box(0.25, y1, 3.50, h1,
        "主导模态与作用程度\nπ_m = α_m / Σ_j α_j\n（门控权重归一化）\n主导模态 = argmax_m π_m",
        "#FFF8E1", ec="#E8A33D", fs=8.4, bold=True)
    box(4.05, y1, 3.50, h1,
        "关键证据定位\n原生证据 ē_m(t) 取 top-k\n→ 关键片段 {t_1, …, t_k}",
        "#E3F2FD", ec=C_TEXT, fs=8.4)
    box(7.85, y1, 3.50, h1,
        "证据落回原始素材\n文本词 / 短语 · 语音时段 · 视觉关键帧\n（对应可解释测试集原始片段）",
        "#EAF4EC", ec=C_VIS, fs=8.4)
    box(11.65, y1, 4.10, h1,
        "解释卡输出\n预测极性+强度 · 主导模态 · 作用程度 π_m\n关键证据（词/时段/帧）· 置信度",
        "#F3E5F5", ec="#8E6BB4", fs=8.4, bold=True)
    for xa, xb in [(3.75, 4.05), (7.55, 7.85), (11.35, 11.65)]:
        arr(xa, y1 + h1/2, xb, y1 + h1/2, lw=1.3)

    # ---------------- 下-②：模型无关互证（辅助） ----------------
    ax.text(0.25, 3.30, "② 模型无关互证（后验归因，仅用于交叉验证，不改变预测）", fontsize=10.5,
            color="#8E6BB4", fontweight="bold", ha="left")
    y2, h2 = 1.40, 1.55
    box(0.25, y2, 4.85, h2,
        "Shapley 模态归因\nφ_m = Σ_S w(S)·[v(S∪{m}) − v(S)]\n（8 个模态子集精确枚举）\n→ 主导模态互证",
        "#FFF8E1", ec="#E8A33D", fs=8.2)
    box(5.55, y2, 4.85, h2,
        "遮挡 LOPO 局部证据\nI_m(t) = logit_c(x) − logit_c(x_{-t})\n（逐位置置零，logit 降幅越大越关键）\n→ top-k 关键证据互证",
        "#E3F2FD", ec=C_TEXT, fs=7.9)
    box(10.85, y2, 4.85, h2,
        "一致性检验\n主导模态一致率 · 关键证据 Top-k 重叠率\n（原生解释 与 互证结论 对齐）\n→ 解释可信度",
        "#F3E5F5", ec="#8E6BB4", fs=8.2)
    arr(5.10, y2 + h2/2, 5.55, y2 + h2/2, lw=1.2)
    arr(10.40, y2 + h2/2, 10.85, y2 + h2/2, lw=1.2)
    arr(13.20, y2 + h2, 13.35, y1, c="#8E6BB4", lw=1.2, ls="--")

    handles = [matplotlib.patches.Patch(color=C_TEXT, label="文本"),
               matplotlib.patches.Patch(color=C_AUDIO, label="语音"),
               matplotlib.patches.Patch(color=C_VIS, label="视觉"),
               matplotlib.patches.Patch(fc="white", ec="#8E6BB4", ls="--", label="可解释性引线")]
    ax.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.6, frameon=False,
              bbox_to_anchor=(0.5, 0.04))
    ax.text(8, 0.92,
            "关键点：局部证据 ē_m(t) 与门控 α 直接来自模型前向计算（前验/原生可解释），预测与解释同源；"
            "Shapley 与遮挡 LOPO 为模型无关后验手段，仅作交叉验证。",
            ha="center", fontsize=9.3, color=C_TITLE)
    save(fig, "qx2_01_arch")


def qx2_02_confusion():
    ev = load("valid_eval.json")
    cm = np.array(ev["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(list(CLS_NAME.values()))
    ax.set_yticks(range(3)); ax.set_yticklabels(list(CLS_NAME.values()))
    ax.set_xlabel("预测"); ax.set_ylabel("真实")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontweight="bold")
    ax.set_title("验证集混淆矩阵（AEF）", fontsize=12, color=C_TITLE)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "qx2_02_confusion")


def qx2_03_reg_scatter():
    vp = pd.read_csv(os.path.join(OUT, "valid_predictions.csv")) if os.path.exists(os.path.join(OUT, "valid_predictions.csv")) else None
    # 用 faithfulness_samples 的预测/真实（不足则从 valid_eval 重建）
    ev = load("valid_eval.json")
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    # 这里仅展示 MAE 分布占位：用 per-class MAE 柱状
    pc = ev["per_class"]
    labels = list(CLS_NAME.values())
    ax.bar(labels, [pc[k]["mae"] for k in labels], color=[C_TEXT, C_AUDIO, C_VIS], alpha=0.9)
    for i, k in enumerate(labels):
        ax.text(i, pc[k]["mae"], f"{pc[k]['mae']:.3f}", ha="center", va="bottom")
    ax.set_ylabel("MAE"); ax.set_title("各类别强度 MAE（AEF）", fontsize=12, color=C_TITLE)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "qx2_03_reg_mae")


def qx2_04_modality_diff():
    fs = pd.read_csv(os.path.join(OUT, "faithfulness_samples.csv"))
    am = [fs["alpha_text"].mean(), fs["alpha_audio"].mean(), fs["alpha_vision"].mean()]
    pm = [fs["phi_text"].mean(), fs["phi_audio"].mean(), fs["phi_vision"].mean()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = ["文本", "语音", "视觉"]
    axes[0].bar(labels, am, color=[C_TEXT, C_AUDIO, C_VIS], alpha=0.9)
    for i, v in enumerate(am): axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    axes[0].set_ylabel("平均门控 α"); axes[0].set_title("模态作用程度（门控 α）", fontsize=10.5)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, pm, color=[C_TEXT, C_AUDIO, C_VIS], alpha=0.9)
    for i, v in enumerate(pm): axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top")
    axes[1].axhline(0, color=C_GRAY, lw=0.9)
    axes[1].set_ylabel("平均 Shapley φ"); axes[1].set_title("模态作用程度（Shapley φ）", fontsize=10.5)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("验证集三模态作用程度（AEF）", fontsize=12, color=C_TITLE)
    save(fig, "qx2_04_modality_diff")


def qx2_05_evidence_heatmap():
    exp = load("explainable_explanations.json")
    ids = sorted(exp.keys())[:4]
    fig, axes = plt.subplots(len(ids), 1, figsize=(12, 1.6*len(ids)+0.8), squeeze=False)
    cmap = LinearSegmentedColormap.from_list("ev", ["#FFFFFF", "#4C9BC7"])
    for r, sid in enumerate(ids):
        ax = axes[r][0]
        mat = np.stack([exp[sid]["local_importance_evidence"][m] for m in C.MODS])  # (3,50)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=max(0.2, float(np.nanmax(mat))))
        ax.set_yticks(range(3)); ax.set_yticklabels(["文本", "语音", "视觉"], fontsize=8)
        ax.set_ylabel(sid, fontsize=8)
        ax.set_xticks([])
        if r == len(ids)-1: ax.set_xlabel("时序位置 t（0–49）")
    fig.suptitle("可解释测试集原生局部证据 e_m(t) 热力图（AEF，颜色越深越关键）", fontsize=12, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "qx2_05_evidence_heatmap")


def qx2_06_faithfulness():
    ev = load("valid_eval.json")
    ne = ev["native_evidence_faithfulness"]
    v1_ev = load("valid_eval.json", V1)
    v1_oc = load("occlusion_faithfulness_per_modality.json", V1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    # (a) 模态级 LOPO
    f = ev["faithfulness"]
    axes[0].bar(["去掉主导模态", "去掉随机模态"], [f["mean_comp_dom"], f["mean_comp_rand"]], color=[C_TITLE, C_GRAY], alpha=0.9)
    for i, v in enumerate([f["mean_comp_dom"], f["mean_comp_rand"]]): axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    axes[0].set_ylabel("置信度下降 Δp"); axes[0].set_title(f"模态级 LOPO（命中率 {f['dom_hit_rate']:.2f}）", fontsize=9.5)
    axes[0].grid(axis="y", alpha=0.25)
    # (b) 原生证据 top-k vs 随机（每模态）
    x = np.arange(3); w = 0.36
    topk = [ne[m]["mean_comp_topk"] for m in C.MODS]
    rand = [ne[m]["mean_comp_randk"] for m in C.MODS]
    axes[1].bar(x-w/2, topk, w, label="删 top-k 证据", color=[C_TEXT, C_AUDIO, C_VIS], alpha=0.9)
    axes[1].bar(x+w/2, rand, w, label="删随机 k", color=[C_TEXT, C_AUDIO, C_VIS], alpha=0.35)
    axes[1].set_xticks(x); axes[1].set_xticklabels(["文本", "语音", "视觉"])
    axes[1].set_ylabel("置信度下降 Δp"); axes[1].set_title("原生证据 top-k vs 随机（AEF）", fontsize=9.5)
    axes[1].legend(fontsize=7.5); axes[1].grid(axis="y", alpha=0.25)
    # (c) 位置级胜率 v1 vs v2
    v1win = [v1_oc["per_modality"][m]["comp_win_rate"] for m in C.MODS]
    v2win = [ne[m]["comp_win_rate"] for m in C.MODS]
    axes[2].plot(["文本","语音","视觉"], v1win, "-o", color=C_GRAY, label="后验基线（遮挡归因）")
    axes[2].plot(["文本","语音","视觉"], v2win, "-s", color=C_TITLE, label="AEF（原生证据）")
    axes[2].axhline(0.5, color="#B3554D", ls="--", lw=1)
    axes[2].set_ylim(0.3, 0.8); axes[2].set_ylabel("top-k 胜率")
    axes[2].set_title("位置级证据胜率：后验基线 vs AEF", fontsize=9.5)
    axes[2].legend(fontsize=7.5); axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("可解释性忠实度（AEF 原生证据）", fontsize=12, color=C_TITLE)
    save(fig, "qx2_06_faithfulness")


def qx2_07_error_attribution():
    ev = load("valid_eval.json")
    pc = ev["per_class"]
    labels = list(CLS_NAME.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(labels, [pc[k]["acc"] for k in labels], color=[C_TEXT, C_AUDIO, C_VIS], alpha=0.9)
    for i, k in enumerate(labels): axes[0].text(i, pc[k]["acc"], f"{pc[k]['acc']:.3f}", ha="center", va="bottom")
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Accuracy"); axes[0].set_title("各类别准确率", fontsize=10.5)
    axes[0].grid(axis="y", alpha=0.25)
    top = list(ev["error_pairs"].items())[:6]
    axes[1].barh([p for p,_ in top], [v for _,v in top], color="#B3554D", alpha=0.85)
    axes[1].invert_yaxis()
    for i,(p,v) in enumerate(top): axes[1].text(v, i, f" {v}", va="center")
    axes[1].set_title("主要错误流向", fontsize=10.5); axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("验证集错误归因（AEF）", fontsize=12, color=C_TITLE)
    save(fig, "qx2_07_error_attribution")


def qx2_08_v1_vs_v2():
    v1 = load("valid_eval.json", V1)["valid_clean"]
    v2 = load("valid_eval.json")["valid_clean"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    keys = ["accuracy", "f1_macro", "mae", "pearson"]
    labels = ["Accuracy", "F1-macro", "MAE", "Pearson r"]
    v1v = [v1["accuracy"], v1["f1_macro"], v1["mae"], v1["pearson"]]
    v2v = [v2["accuracy"], v2["f1_macro"], v2["mae"], v2["pearson"]]
    x = np.arange(4); w = 0.36
    axes[0].bar(x-w/2, v1v, w, label="后验解释基线（后验基线）", color=C_GRAY, alpha=0.9)
    axes[0].bar(x+w/2, v2v, w, label="原生证据加权融合 AEF", color=C_TITLE, alpha=0.9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_title("验证集主指标对比", fontsize=10.5)
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.25)
    for i in range(4):
        axes[0].text(i-w/2, v1v[i], f"{v1v[i]:.3f}", ha="center", va="bottom", fontsize=7.5)
        axes[0].text(i+w/2, v2v[i], f"{v2v[i]:.3f}", ha="center", va="bottom", fontsize=7.5)
    # 位置级胜率对比
    v1_oc = load("occlusion_faithfulness_per_modality.json", V1)["per_modality"]
    ne = load("valid_eval.json")["native_evidence_faithfulness"]
    x = np.arange(3)
    axes[1].plot(x, [v1_oc[m]["comp_win_rate"] for m in C.MODS], "-o", color=C_GRAY, label="后验基线（遮挡归因）")
    axes[1].plot(x, [ne[m]["comp_win_rate"] for m in C.MODS], "-s", color=C_TITLE, label="AEF（原生证据）")
    axes[1].axhline(0.5, color="#B3554D", ls="--", lw=1)
    axes[1].set_xticks(x); axes[1].set_xticklabels(["文本", "语音", "视觉"])
    axes[1].set_ylabel("位置级证据胜率"); axes[1].set_title("局部证据忠实度对比（关键差异）", fontsize=10.5)
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("模块三 新旧方案对比", fontsize=12, color=C_TITLE)
    save(fig, "qx2_08_v1_vs_v2")


def main():
    for fn in (qx2_01_arch, qx2_02_confusion, qx2_03_reg_scatter, qx2_04_modality_diff,
               qx2_05_evidence_heatmap, qx2_06_faithfulness, qx2_07_error_attribution, qx2_08_v1_vs_v2):
        try:
            fn()
        except Exception as ex:
            import traceback
            print(f"[ERR] {fn.__name__}: {ex}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
