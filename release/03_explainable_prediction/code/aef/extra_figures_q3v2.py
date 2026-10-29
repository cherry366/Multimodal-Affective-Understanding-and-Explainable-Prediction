# -*- coding: utf-8 -*-
"""模块三 最终方案（AEF）补充图：把后验解释基线的图全集补齐到最终方案。"""
import os
import json
import numpy as np
import pandas as pd
import torch
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE

import config_q3v2 as C
import data_q2 as D
import train_q2 as T
import data_q3 as D3
from model_q3v2 import Q3V2Net

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 10.5
plt.rcParams["figure.dpi"] = 130
C_TEXT, C_AUDIO, C_VIS = "#4C9BC7", "#E4636F", "#4FB6A6"
C_TITLE, C_GRAY = "#1F3B6E", "#8A8A8A"
MOD_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}
CLS_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}
CLS_COLOR = {0: "#D05A6E", 1: "#B0A36A", 2: "#5B9B74"}
OUT, FIGS = C.Q3V2_OUT, C.Q3V2_FIGS


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), format=ext, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("FIG ->", name, flush=True)


def build(dev):
    return Q3V2Net().to(dev)


def load_models():
    dev = T.device_of()
    models = []
    for s in C.SEEDS:
        ck = T.load_ckpt(os.path.join(C.Q3V2_CKPT, f"q3v2_s{s}.pt"))
        m = build(dev)
        m.load_state_dict(ck["state"])
        m.eval()
        models.append((ck, m))
    return models, dev


@torch.no_grad()
def ens_output(models, Xn, Mn, dev, want_z=False):
    Ps, Rs, As, Evs, Zs = [], [], [], {}, []
    for _, m in models:
        out = m(T.to_t(Xn, dev), T.to_t(Mn, dev))
        Ps.append(torch.softmax(out["logits"], 1).cpu().numpy())
        Rs.append(out["reg"].cpu().numpy())
        As.append(out["alpha"].cpu().numpy())
        for k in C.MODS:
            Evs.setdefault(k, []).append(out["evidence"][k].cpu().numpy())
        if want_z:
            Zs.append(out["z"].cpu().numpy())
    res = [np.mean(Ps, 0), np.mean(Rs, 0), np.mean(As, 0), {k: np.mean(Evs[k], 0) for k in C.MODS}]
    if want_z:
        res.append(np.mean(Zs, 0))
    return tuple(res)


# ================================================================= qx2_09 思维流程
def qx2_09_flow():
    fig, ax = plt.subplots(figsize=(12.5, 9.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 10.8); ax.axis("off")
    def box(x, y, w, h, t, fc="#FFFFFF", ec=C_GRAY, fs=8.6, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
        ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
    def arrow(x1,y1,x2,y2,c=C_TITLE):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>", mutation_scale=13, color=c, lw=1.5))
    ax.text(6.5, 10.45, "模块三 原生证据加权融合 AEF完整流程", ha="center", fontsize=13.5, color=C_TITLE, fontweight="bold")
    box(3.6, 9.3, 5.8, 0.85, "输入：三模态完整特征 + 掩码\n文本 50×768 · 语音 50×74 · 视觉 50×35", fc="#F2F6FA")
    arrow(6.5, 9.3, 6.5, 8.7)
    box(3.2, 7.8, 6.6, 0.9, "冻结 Q2 编码器：mask感知 Transformer（与模块二 共用）", fc="#EEF3F8", ec=C_TITLE, bold=True)
    arrow(6.5, 7.8, 6.5, 7.2)
    box(3.2, 6.2, 6.6, 1.0, "原生证据打分 ē_m(t) = softmax(Ev_m(h_m(t)))\n→ 证据加权池化 pool_m = Σ_t ē_m(t)·h_m(t)", fc="#E3F2FD", ec=C_TEXT)
    arrow(6.5, 6.2, 6.5, 5.6)
    box(3.2, 4.7, 6.6, 0.9, "跨模态注意力 + MoE + 覆盖率先验门控 α\nz = Σ_m α_m·enriched_m", fc="#E4EFE7", ec="#5B9B74")
    arrow(6.5, 4.7, 6.5, 4.1)
    box(3.2, 3.2, 6.6, 0.9, "三头输出：情感极性 p(y) · 强度 ŝ · 不确定度 σ²", fc="#FFE9EC", ec=C_AUDIO)
    ax.text(6.5, 2.7, "前验解释（与预测同源，一次前向输出）", ha="center", fontsize=8.8, color="#8E6BB4", fontweight="bold")
    box(0.9, 0.9, 5.1, 1.6, "① 模态作用：门控 α（原生）+ Shapley φ（互证）\n→ 主要参考模态 + 作用程度", fc="#FFF8E1", ec="#E8A33D", fs=8.2)
    box(7.0, 0.9, 5.1, 1.6, "② 关键证据：原生证据 ē_m(t) top-k\n→ 文本词 / 语音时段 / 视觉关键帧", fc="#E3F2FD", ec=C_TEXT, fs=8.2)
    arrow(6.5, 3.2, 3.4, 2.6, c="#8E6BB4"); arrow(3.4, 2.6, 3.4, 2.5, c="#8E6BB4")
    arrow(6.5, 3.2, 9.6, 2.6, c="#8E6BB4"); arrow(9.6, 2.6, 9.6, 2.5, c="#8E6BB4")
    save(fig, "qx2_09_flow")


# ================================================================= qx2_10 回归散点
def qx2_10_reg_scatter():
    models, dev = load_models()
    dt = T.get_data()
    Xv, Mv = dt["X"]["valid"], dt["M"]["valid"]
    yr = dt["items"]["valid"]["y_reg"]
    yc = dt["items"]["valid"]["y_cls"]
    P, R, _, _ = ens_output(models, Xv, Mv, dev)
    ev = json.load(open(os.path.join(OUT, "valid_eval.json"), encoding="utf-8"))
    r, mae = ev["valid_clean"]["pearson"], ev["valid_clean"]["mae"]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    col = [CLS_COLOR[c] for c in yc]
    ax.scatter(yr, R, c=col, s=14, alpha=0.5, edgecolors="none")
    ax.plot([-3.2, 3.2], [-3.2, 3.2], "--", color=C_GRAY, lw=1.2)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("真实情感强度 s"); ax.set_ylabel("预测情感强度 ŝ")
    ax.set_title(f"验证集强度回归（Pearson r={r:.3f}, MAE={mae:.3f}）", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(handles=[Patch(color=CLS_COLOR[k], label=CLS_NAME[k]) for k in (0,1,2)], loc="upper left", fontsize=9, framealpha=0.9)
    save(fig, "qx2_10_reg_scatter")


# ================================================================= qx2_11 可解释测试集汇总
def qx2_11_a4_summary():
    df = pd.read_csv(os.path.join(OUT, "explainable_predictions.csv"), dtype={"id": str})
    fig = plt.figure(figsize=(13.5, 6.2))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2.3, 1.0, 1.0], hspace=0.55, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, :2])
    y = np.arange(len(df))
    colors = [CLS_COLOR[c] for c in df["predicted_class"]]
    ax1.barh(y, df["predicted_score"], color=colors, alpha=0.85)
    for i in range(len(df)):
        ax1.text(df["predicted_score"][i] + (0.06 if df["predicted_score"][i]>=0 else -0.06), i,
                 f"{df['predicted_score'][i]:+.2f}", va="center",
                 ha="left" if df["predicted_score"][i]>=0 else "right", fontsize=8.2)
    ax1.axvline(0, color=C_GRAY, lw=0.9)
    ax1.set_yticks(y); ax1.set_yticklabels([f"a4_{i}" for i in df["id"]], fontsize=8)
    ax1.invert_yaxis(); ax1.set_xlabel("预测情感强度 ŝ")
    ax1.set_title("可解释测试集全量 20 条样本的强度预测与极性（AEF）", fontsize=12)
    ax1.grid(axis="x", alpha=0.25)
    ax1.legend(handles=[Patch(color=CLS_COLOR[k], label=CLS_NAME[k]) for k in (0,1,2)], loc="lower right", fontsize=8.5, framealpha=0.9)
    ax2 = fig.add_subplot(gs[0, 2])
    order = ["Negative", "Neutral", "Positive"]
    cnt = df["predicted_class_name"].value_counts()
    vals = [int(cnt.get(k, 0)) for k in order]
    ax2.bar(order, vals, color=[CLS_COLOR[k] for k in (0,1,2)], alpha=0.85)
    for i, v in enumerate(vals): ax2.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")
    ax2.set_title("极性分布", fontsize=11); ax2.set_ylabel("样本数")
    ax3 = fig.add_subplot(gs[1, :2])
    means = [df[f"alpha_{m}"].mean() for m in C.MODS]
    pis = [df[f"pi_{m}"].mean() for m in C.MODS]
    x = np.arange(3); wd = 0.34
    ax3.bar(x-wd/2, means, wd, label="门控 α（原生）", color=C_TITLE, alpha=0.9)
    ax3.bar(x+wd/2, pis, wd, label="Shapley π（互证）", color="#E8A33D", alpha=0.9)
    ax3.set_xticks(x); ax3.set_xticklabels(["文本", "语音", "视觉"])
    ax3.set_ylim(0, 1); ax3.set_ylabel("平均作用占比")
    ax3.set_title("可解释测试集三模态平均作用占比（α vs Shapley π）", fontsize=11)
    ax3.legend(fontsize=8.5); ax3.grid(axis="y", alpha=0.25)
    ax4 = fig.add_subplot(gs[1, 2])
    bp4 = ax4.boxplot([df["shapley_text"], df["shapley_audio"], df["shapley_vision"]],
                      labels=["文本", "语音", "视觉"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp4["boxes"], [C_TEXT, C_AUDIO, C_VIS]): patch.set_facecolor(c); patch.set_alpha(0.45)
    ax4.axhline(0, color=C_GRAY, lw=1.0); ax4.set_ylabel("Shapley φ")
    ax4.set_title("三模态 Shapley φ 分布", fontsize=10); ax4.grid(axis="y", alpha=0.25)
    save(fig, "qx2_11_a4_summary")


# ================================================================= qx2_12 解释卡
def _mod_bars(ax, alpha, pi, dom):
    labels = ["文本", "语音", "视觉"]
    x = np.arange(3)
    ax.bar(x, [alpha["text"], alpha["audio"], alpha["vision"]], 0.26, label="门控 α", color=C_TITLE)
    ax.bar(x+0.26, [pi["text"], pi["audio"], pi["vision"]], 0.26, label="Shapley π", color="#E8A33D")
    ax.set_xticks(x+0.13); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("作用占比", fontsize=8); ax.set_ylim(0, 1)
    ax.set_title("模态作用程度", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.25)


def _imp_curve(ax, evd, topk, color, label):
    w = np.abs(np.array(evd))
    ax.plot(range(50), w, color=color, lw=1.3, label=label)
    if topk:
        tp = [t for t in topk if 0 <= int(t) < 50]
        if tp:
            ax.scatter(tp, [w[t] for t in tp], color=color, s=18, zorder=3, edgecolors="white", linewidths=0.6)
    ax.set_xlim(0, 49); ax.set_ylim(0, max(w.max()*1.15, 1e-3))
    ax.set_xlabel("时序位置 t", fontsize=8); ax.set_ylabel("|证据 ē|", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")


def qx2_12_cards():
    expl = json.load(open(os.path.join(OUT, "explainable_explanations.json"), encoding="utf-8"))
    df = pd.read_csv(os.path.join(OUT, "explainable_predictions.csv"), dtype={"id": str})
    mod_cn_card = {"text": "文本 T", "audio": "语音 A", "vision": "视觉 V"}
    picks = [("09", "Negative"), ("17", "Positive"), ("01", "Neutral")]
    fig = plt.figure(figsize=(16.5, 13.5))
    gs = GridSpec(6, 3, figure=fig, hspace=0.72, wspace=0.30, height_ratios=[0.8, 1.0] * 3)
    for k, (sid, want) in enumerate(picks):
        e = expl[sid]
        row = df[df["id"] == sid].iloc[0]
        r0 = 2 * k
        p_score = row["predicted_score"]
        p_sigma = row["score_uncertainty_sigma"]
        p_dom = row["dominant_modality"]
        p_cls = row["predicted_class_name"]
        p_cls_id = int(row["predicted_class"])
        # 左：预测概览（跨 2 行）
        axp = fig.add_subplot(gs[r0:r0+2, 0])
        axp.axis("off"); axp.set_xlim(0, 1); axp.set_ylim(0, 1)
        cc = CLS_COLOR[p_cls_id]
        axp.add_patch(FancyBboxPatch((0.02, 0.88), 0.96, 0.10, boxstyle="round,pad=0.008", fc=cc, ec="none", alpha=0.9, transform=axp.transAxes))
        axp.text(0.5, 0.93, f"样本 a4_{sid}  ·  预测极性 {p_cls}", transform=axp.transAxes, ha="center", va="center", fontsize=11.5, color="white", fontweight="bold")
        axp.text(0.04, 0.82, f"情感强度 ŝ = {p_score:+.3f}", transform=axp.transAxes, fontsize=10, color=C_TITLE, va="center")
        axp.text(0.04, 0.74, f"不确定度 σ = {p_sigma:.3f}", transform=axp.transAxes, fontsize=10, color=C_TITLE, va="center")
        axp.text(0.04, 0.66, f"主要参考模态 = {mod_cn_card[p_dom]}", transform=axp.transAxes, fontsize=10, color=C_TITLE, va="center")
        axp.text(0.04, 0.56, "文本证据：", transform=axp.transAxes, fontsize=9, color=C_GRAY, va="center")
        txt = e["raw_text"]
        axp.text(0.04, 0.50, txt[:160] + ("…" if len(txt) > 160 else ""), transform=axp.transAxes, fontsize=8, color="#444444", va="top", wrap=True)

        # 中：模态作用程度（跨 2 行）
        axm = fig.add_subplot(gs[r0:r0+2, 1])
        _mod_bars(axm, e["modality_contribution"]["alpha"], e["modality_contribution"]["pi"], p_dom)

        # 右上：主导模态局部证据曲线
        dom = p_dom
        evd = e["local_importance_evidence"][dom]
        if dom == "text":
            topk = e["evidence_text"]["positions"]
        elif dom == "audio":
            topk = e["evidence_audio"]["positions"]
        else:
            topk = e["evidence_vision"]["positions"]
        axi = fig.add_subplot(gs[r0, 2])
        _imp_curve(axi, evd, topk, {"text": C_TEXT, "audio": C_AUDIO, "vision": C_VIS}[dom], f"{mod_cn_card[dom]} 原生证据")

        # 右下：证据定位 chips
        axe = fig.add_subplot(gs[r0+1, 2])
        axe.axis("off"); axe.set_xlim(0, 1); axe.set_ylim(0, 1)
        axe.text(0.02, 0.98, "关键证据定位", fontsize=9.5, color=C_TITLE, fontweight="bold", va="top")
        y = 0.86
        axe.text(0.02, y, "文本证据词（原生证据 ē）：", fontsize=8, color="#333333", va="top")
        y -= 0.055
        x = 0.02
        for wi, w in enumerate(e["evidence_text"]["words"][:5]):
            t = int(e["evidence_text"]["positions"][wi])
            val = float(e["local_importance_evidence"]["text"][min(max(t, 0), 49)])
            col = "#2E6DA4" if val >= 0 else "#B03A2E"
            lab = f"{w} {val:+.2f}"
            axe.text(x, y, lab, fontsize=7.6, va="top", ha="left", color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor=col, edgecolor="none"))
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
    fig.suptitle("可解释测试集典型样本可解释性解释卡（预测概览 · 模态作用分解 · 关键证据定位，AEF 原生证据）", fontsize=13.5, color=C_TITLE, y=0.995)
    save(fig, "qx2_12_cards")


# ================================================================= qx2_13 关键帧拼图
def qx2_13_frames():
    exp = json.load(open(os.path.join(OUT, "explainable_explanations.json"), encoding="utf-8"))
    picks = {}
    for sid, e in exp.items():
        c = e["prediction"]["class"]
        if c not in picks: picks[c] = sid
    rows = []
    for c in ["Negative", "Neutral", "Positive"]:
        e = exp[picks[c]]
        for f in e["evidence_vision"]["key_frames"][:4]:
            if f and os.path.exists(f):
                rows.append((picks[c], e["prediction"]["class"], f))
    n = len(rows)
    cols = min(4, n)
    lines = int(np.ceil(n / cols))
    fig, axes = plt.subplots(lines, cols, figsize=(3.2*cols, 2.6*lines), squeeze=False)
    for k, (sid, cls, f) in enumerate(rows):
        ax = axes[k//cols][k%cols]
        ax.imshow(mpimg.imread(f)); ax.axis("off")
        ax.set_title(f"a4_{sid} {cls}", fontsize=8, color=CLS_COLOR[list(CLS_NAME.keys())[list(CLS_NAME.values()).index(cls)]])
    for k in range(n, lines*cols):
        axes[k//cols][k%cols].axis("off")
    fig.suptitle("典型样本关键视觉证据帧（AEF）", fontsize=12.5, color=C_TITLE)
    save(fig, "qx2_13_frames")


# ================================================================= qx2_14 词级热力
def qx2_14_word_heat():
    exp = json.load(open(os.path.join(OUT, "explainable_explanations.json"), encoding="utf-8"))
    cmap = plt.get_cmap("YlOrRd")
    picks = {}
    for sid, e in exp.items():
        c = e["prediction"]["class"]
        if c not in picks: picks[c] = sid
    order = ["Negative", "Neutral", "Positive"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.8))
    for ax, c in zip(axes, order):
        e = exp[picks[c]]
        words = e["raw_text"].strip().split()
        ev = e["local_importance_evidence"]["text"]
        imp = np.zeros(len(words))
        L = C.L
        for i in range(len(words)):
            pos = i if len(words) <= L else int(i * L / len(words))
            imp[i] = abs(float(ev[min(pos, L-1)]))
        mx = imp.max() or 1.0
        evwords = set(e["evidence_text"]["words"])
        x = 0.02; y = 0.86; line_h = 0.15; char_w = 0.0106
        for i, w in enumerate(words):
            v = imp[i] / mx
            fc = cmap(v) if v > 0.02 else "#FFFFFF"
            is_ev = w in evwords
            ax.text(x, y, w, fontsize=9.2, va="center", ha="left", color="#111111",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=fc, edgecolor="#B03A2E" if is_ev else "none", lw=0.8))
            ww = len(w)*char_w + 0.010
            if is_ev:
                ax.text(x + ww*0.35, y-0.03, f"{float(imp[i]):.2f}", fontsize=5.4, va="top", color="#1F6E43", fontweight="bold")
            x += ww
            if x > 0.965: x = 0.02; y -= line_h
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(f"a4_{picks[c]} · 预测 {e['prediction']['class']} · 主导 {MOD_CN[e['dominant_modality']]}（色深=原生证据强度）", fontsize=10, color=C_TITLE)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.015).set_label("归一化证据强度")
    save(fig, "qx2_14_word_heat")


# ================================================================= qx2_15 跨模态注意力（每类一条代表样本，3 种子平均，逐位置热力图）
def qx2_15_cross_attention():
    models, dev = load_models()
    dt = T.get_data()
    Xv, Mv = dt["X"]["valid"], dt["M"]["valid"]
    yc = dt["items"]["valid"]["y_cls"]
    keys = [("text","audio"), ("text","vision"), ("audio","text"),
            ("audio","vision"), ("vision","text"), ("vision","audio")]
    pair_cn = ["T ← A", "T ← V", "A ← T", "A ← V", "V ← T", "V ← A"]
    picks = [int(np.where(yc == c)[0][0]) for c in (0, 1, 2)]
    pick_cn = ["Negative", "Neutral", "Positive"]
    fig, axes = plt.subplots(3, 6, figsize=(19.5, 8.6))
    for r, si in enumerate(picks):
        Xb = {m: Xv[m][si:si+1] for m in C.MODS}
        Mb = {m: Mv[m][si:si+1] for m in C.MODS}
        agg = {k: np.zeros((50, 50), dtype=np.float64) for k in keys}
        n_models = 0
        for _, model in models:
            model.eval()
            caps = {k: None for k in keys}
            hooks = {}
            def make_hook(key):
                def h(module, inp, outp):
                    w = outp[1].detach().cpu().float()  # (B, L, S)
                    if w.ndim == 4:
                        w = w.mean(1)
                    caps[key] = w[0].numpy()
                return h
            for k in keys:
                hooks[k] = model.cross[f"{k[0]}<-{k[1]}"].register_forward_hook(make_hook(k))
            with torch.no_grad():
                model(T.to_t(Xb, dev), T.to_t(Mb, dev))
            for h in hooks.values():
                h.remove()
            for k in keys:
                if caps[k] is not None:
                    agg[k] += caps[k]
            n_models += 1
        for k in keys:
            agg[k] /= max(n_models, 1)
        vmax = max([float(m.max()) for m in agg.values()])
        for c, (k, cn) in enumerate(zip(keys, pair_cn)):
            ax = axes[r, c]
            im = ax.imshow(agg[k], cmap="YlGnBu", aspect="auto", vmin=0, vmax=vmax)
            if r == 0:
                ax.set_title(cn, fontsize=9.5, color=C_TITLE)
            if r == 2:
                ax.set_xlabel("源位置", fontsize=7.5)
            if c == 0:
                ax.set_ylabel(f"{pick_cn[r]}\n目标位置", fontsize=7.5)
            ax.tick_params(labelsize=6)
    fig.suptitle("跨模态注意力权重热力图（AEF，3 种子平均，每行一条代表样本）", fontsize=12.5, color=C_TITLE)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015).set_label("注意力权重", fontsize=8)
    save(fig, "qx2_15_cross_attention")
# ================================================================= qx2_16 t-SNE
def qx2_16_tsne():
    models, dev = load_models()
    dt = T.get_data()
    Xv, Mv = dt["X"]["valid"], dt["M"]["valid"]
    yc = dt["items"]["valid"]["y_cls"]
    P, R, _, _, Z = ens_output(models, Xv, Mv, dev, want_z=True)
    z2 = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in (0,1,2):
        ax.scatter(z2[yc==c,0], z2[yc==c,1], s=12, alpha=0.55, c=CLS_COLOR[c], label=CLS_NAME[c])
    ax.set_title("融合表示 z 的 t-SNE 嵌入（验证集，AEF）", fontsize=11, color=C_TITLE)
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    save(fig, "qx2_16_tsne")


# ================================================================= qx2_17 证据时间轴
def qx2_17_timeline():
    exp = json.load(open(os.path.join(OUT, "explainable_explanations.json"), encoding="utf-8"))
    picks = {}
    for sid, e in exp.items():
        c = e["prediction"]["class"]
        if c not in picks: picks[c] = sid
    tags = ["neg", "neu", "pos"]
    order = ["Negative", "Neutral", "Positive"]
    for k, c in enumerate(order):
        sid = picks[c]; e = exp[sid]
        dur, fps, nf = D3.video_meta(int(sid))
        D = max(dur, 1e-6)
        fig = plt.figure(figsize=(11.2, 6.4))
        gs = GridSpec(3, 1, height_ratios=[1.0, 0.5, 1.0], hspace=0.42)
        # 文本轨
        axT = fig.add_subplot(gs[0])
        axT.axis("off"); axT.set_xlim(0, D); axT.set_ylim(0, 1)
        words = e["raw_text"].strip().split()
        ev = e["local_importance_evidence"]["text"]
        evw = set(e["evidence_text"]["words"])
        half = int(np.ceil(len(words)/2)); n = len(words)
        cmap_txt = plt.get_cmap("YlOrRd")
        mx = max([abs(float(ev[min(max(int(t),0),49)])) for t in e["evidence_text"]["positions"]] + [1e-9])
        def place(ax_, nw, line):
            m = 0.05*D; span = max(D-2*m, 1e-6)
            y = 0.72 if line==0 else 0.28
            for i in range(nw):
                j = i if line==0 else i+half
                x = m + (i+0.5)/max(nw,1)*span
                if words[j] in evw:
                    v = abs(float(ev[min(max(int(j if n<=C.L else int(j*C.L/n)),0),49)]))/mx
                    ax_.text(x, y, words[j], ha="center", va="center", fontsize=11,
                             color="white" if v>0.55 else "#111111",
                             bbox=dict(boxstyle="round,pad=0.28", facecolor=cmap_txt(0.2+0.8*v), edgecolor="#B03A2E", lw=0.9))
                else:
                    ax_.text(x, y, words[j], ha="center", va="center", fontsize=11, color="#333333")
        place(axT, half, 0)
        if n > half: place(axT, n-half, 1)
        axT.set_title(f"a4_{sid} · 文本轨（证据词高亮，色深∝证据强度）", fontsize=10, color=C_TITLE)
        # 语音轨
        axA = fig.add_subplot(gs[1]); axA.set_ylim(0,1); axA.set_xlim(0,D)
        eva = e["local_importance_evidence"]["audio"]
        axA.plot(np.linspace(0,D,50), np.abs(eva)/max(np.abs(eva).max(),1e-9), color=C_AUDIO, lw=1.6)
        for w in e["evidence_audio"]["time_windows"]:
            axA.axvspan(w[0], w[1], color=C_AUDIO, alpha=0.25)
        axA.set_yticks([]); axA.set_ylabel("语音证据", fontsize=9)
        axA.set_title("语音轨（关键时段高亮）", fontsize=10, color=C_TITLE)
        # 视觉轨
        axV = fig.add_subplot(gs[2]); axV.set_ylim(0,1.15); axV.set_xlim(0,D); axV.set_yticks([])
        evv = e["local_importance_evidence"]["vision"]
        axV.plot(np.linspace(0,D,50), np.abs(evv)/max(np.abs(evv).max(),1e-9), color=C_VIS, lw=1.4, alpha=0.5)
        items = []
        for f, p in zip(e["evidence_vision"]["key_frames"], e["evidence_vision"]["positions"]):
            if f and os.path.exists(f):
                items.append((float((int(p)+0.5)*D/50.0), f))
        for t, f in items:
            axV.add_patch(Rectangle((t-0.05*D, 0.35), 0.10*D, 0.45, fill=False, edgecolor=C_VIS, lw=1.5))
            axV.imshow(mpimg.imread(f), extent=[t-0.05*D, t+0.05*D, 0.35, 0.80], aspect="auto")
            axV.plot([t, t], [0.80, 0.10], ls=":", color=C_GRAY, lw=0.8)
        axV.axhline(0.10, color="#BBBBBB", lw=0.8)
        axV.set_xlabel("时间 (s)"); axV.set_title("视觉轨（关键帧按真实时间落位）", fontsize=10, color=C_TITLE)
        fig.suptitle(f"多模态关键证据时间轴对齐（a4_{sid} · {c}）", fontsize=12, color=C_TITLE)
        save(fig, f"qx2_17{tags[k]}_evidence_timeline")


# ================================================================= qx2_18 必要性/充分性
def qx2_18_necessity():
    models, dev = load_models()
    dt = T.get_data()
    Xv, Mv = dt["X"]["valid"], dt["M"]["valid"]
    yc = dt["items"]["valid"]["y_cls"]
    tau = float(np.mean([ck["tau"] for ck, _ in models]))
    res = {}
    for r in (1,2,3):
        for combo in itertools.combinations(C.MODS, r):
            Xd = {m: (Xv[m] if m in combo else np.zeros_like(Xv[m])) for m in C.MODS}
            Md = {m: (Mv[m] if m in combo else np.zeros_like(Mv[m])) for m in C.MODS}
            P, R, _, _ = ens_output(models, Xd, Md, dev)
            pred = T.decision(P, R, tau, "band")
            res["+".join(combo)] = float((pred == yc).mean())
    full = res["text+audio+vision"]
    nec = {"text": full-res["audio+vision"], "audio": full-res["text+vision"], "vision": full-res["text+audio"]}
    suf = {"text": res["text"], "audio": res["audio"], "vision": res["vision"]}
    majority = float(np.bincount(yc).max()/len(yc))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    keys = ["text","audio","vision","audio+vision","text+audio","text+vision","text+audio+vision"]
    labels = ["仅T","仅A","仅V","A+V","T+A","T+V","全模态"]
    vals = [res[k] for k in keys]
    cols = [C_TEXT,C_AUDIO,C_VIS,C_GRAY,"#7B8DB3","#7B8DB3",C_TITLE]
    b = axes[0].bar(labels, vals, color=cols, alpha=0.92)
    axes[0].axhline(majority, color="#B3554D", ls="--", lw=1.4)
    axes[0].text(6.35, majority, f"多数类 {majority:.3f}", color="#B3554D", fontsize=8.5, va="bottom", ha="right")
    for bb, v in zip(b, vals): axes[0].text(bb.get_x()+bb.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0, max(vals)+0.1); axes[0].set_ylabel("Accuracy"); axes[0].set_title("模态子集性能（AEF）", fontsize=10)
    axes[0].grid(axis="y", alpha=0.25)
    x = np.arange(3); w = 0.36
    axes[1].bar(x-w/2, [nec[m] for m in C.MODS], w, label="必要性", color=[C_TEXT,C_AUDIO,C_VIS], alpha=0.9)
    axes[1].bar(x+w/2, [suf[m] for m in C.MODS], w, label="充分性", color=[C_TEXT,C_AUDIO,C_VIS], alpha=0.4, hatch="//")
    axes[1].set_xticks(x); axes[1].set_xticklabels(["文本","语音","视觉"])
    axes[1].set_ylabel("Accuracy 变化/Accuracy"); axes[1].set_title("必要性 vs 充分性（AEF）", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("模块三 模态必要性/充分性（AEF）", fontsize=12, color=C_TITLE)
    save(fig, "qx2_18_necessity")


def qx2_19_attn_conc():
    models, dev = load_models()
    dt = T.get_data()
    Xv, Mv = dt["X"]["valid"], dt["M"]["valid"]
    N = len(dt["items"]["valid"]["y_cls"])
    keys = [("text","audio"), ("text","vision"), ("audio","text"),
            ("audio","vision"), ("vision","text"), ("vision","audio")]
    pair_cn = ["T←A", "T←V", "A←T", "A←V", "V←T", "V←A"]
    B = 64
    diag = {k: 0.0 for k in keys}
    ent = {k: 0.0 for k in keys}
    cnt = {k: 0 for k in keys}
    for _, model in models:
        model.eval()
        caps = {k: [] for k in keys}
        hooks = {}
        def make_hook(key):
            def h(module, inp, outp):
                w = outp[1].detach().cpu().float()  # (B, L, S)
                if w.ndim == 4:
                    w = w.mean(1)
                caps[key].append(w.numpy())
            return h
        for k in keys:
            hooks[k] = model.cross[f"{k[0]}<-{k[1]}"].register_forward_hook(make_hook(k))
        for s in range(0, N, B):
            idx = np.arange(s, min(s + B, N))
            Xb = {m: Xv[m][idx] for m in C.MODS}
            Mb = {m: Mv[m][idx] for m in C.MODS}
            with torch.no_grad():
                model(T.to_t(Xb, dev), T.to_t(Mb, dev))
        for h in hooks.values():
            h.remove()
        for k in keys:
            for W in caps[k]:
                nb = W.shape[0]
                dvals = W.diagonal(axis1=1, axis2=2)
                diag[k] += float(dvals.mean()) * nb
                eps = 1e-12
                hh = -(W * np.log(W + eps)).sum(2).mean()
                ent[k] += float(hh) * nb
                cnt[k] += nb
    for k in keys:
        if cnt[k] > 0:
            diag[k] /= cnt[k]
            ent[k] = (ent[k] / cnt[k]) / np.log(50.0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(6)
    uni = 1.0 / 50.0
    axes[0].bar(x, [diag[k] for k in keys], color=C_TITLE, alpha=0.85)
    axes[0].axhline(uni, color="#B3554D", ls="--", lw=1.2, label=f"均匀基线 1/50={uni:.3f}")
    axes[0].set_xticks(x); axes[0].set_xticklabels(pair_cn, fontsize=9)
    axes[0].set_ylabel("平均对角线注意力质量")
    axes[0].set_title("时序对齐强度（接近基线=无对齐）", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [ent[k] for k in keys], color=C_AUDIO, alpha=0.85)
    axes[1].axhline(1.0, color="#B3554D", ls="--", lw=1.2, label="均匀=1.0")
    axes[1].set_xticks(x); axes[1].set_xticklabels(pair_cn, fontsize=9)
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("归一化注意力熵")
    axes[1].set_title("注意力集中度（1.0=完全弥散）", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("跨模态注意力浓度分析（AEF，全验证集 × 3 种子）", fontsize=12, color=C_TITLE)
    save(fig, "qx2_19_attn_conc")


def main():
    for fn in (qx2_09_flow, qx2_10_reg_scatter, qx2_11_a4_summary, qx2_12_cards,
               qx2_13_frames, qx2_14_word_heat, qx2_15_cross_attention, qx2_16_tsne,
               qx2_17_timeline, qx2_18_necessity, qx2_19_attn_conc):
        try:
            fn()
        except Exception as ex:
            import traceback
            print(f"[ERR] {fn.__name__}: {ex}")
            traceback.print_exc()


if __name__ == "__main__":
    main()