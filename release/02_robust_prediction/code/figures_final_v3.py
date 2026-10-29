# -*- coding: utf-8 -*-
"""模块二 最终方案 v2.1（cov-prior 门控）全套图。

图清单（均基于 final_v2 3 种子模型 / v2 消融数据）：
  fv01 架构  fv02 缺失测试集缺失形态  fv03 训练曲线  fv04 缺失类型
  fv05 缺失位置  fv06 缺失率  fv07 混淆矩阵  fv08 强度散点
  fv09 覆盖率先验门控  fv10 缺失测试集预测  fv11 组件消融  fv12 极端缺失
  fv13 完整vs缺失权衡  fv14 不确定度校准  fv15 思维流程图  fv16 缺失时长
  fv17 λ网格搜索  fv18 连续vs分散
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import confusion_matrix

import config_q2 as C
import data_q2 as D
import train_q2 as T
from model_q2 import Q2Net
from figures_q2 import save, a3_masks_only, C_TEXT, C_AUDIO, C_VIS, C_TITLE, C_GRAY, MOD_CN

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def build_v2(dev):
    return Q2Net(use_key_mask=True, use_gate=True, use_cross=True, use_recon=True,
                 use_moe=True, use_nll=True, use_cov_prior=True,
                 use_moe_feats=True, fusion_dropout=C.FUSION_DROPOUT).to(dev)


def load_v2(dev):
    dt = T.get_data()
    models = []
    for s in C.SEEDS:
        ck = T.load_ckpt(os.path.join(C.CKPT, f"final_v2_s{s}.pt"))
        m = build_v2(dev)
        m.load_state_dict(ck["state"])
        m.eval()
        models.append((ck, m))
    return dt, models


@torch.no_grad()
def ens_predict(models, Xb, Mb, dev):
    Ps, Rs, As, Vs = [], [], [], []
    for ck, m in models:
        p, r, a, lv = T.predict(m, Xb, Mb, dev)
        Ps.append(p); Rs.append(r); As.append(a)
        if lv is not None:
            Vs.append(np.exp(lv))
    P = np.mean(Ps, 0); R = np.mean(Rs, 0); A = np.mean(As, 0)
    V = np.mean(Vs, 0) if Vs else None
    return P, R, A, V


# ---------------- fv01 架构 ----------------
def fv01_arch():
    fig, ax = plt.subplots(figsize=(15.5, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")

    def box(x, y, w, h, txt, fc, ec="none", fs=8.6, tc="black", lw=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.1",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2, c=C_GRAY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, color=c, lw=1.2))

    ax.text(8, 10.55, "模块二 最终方案 MoDE-Former+ v2.1：mask感知编码 + 覆盖率先验门控 + 专家路由 + 不确定性输出",
            ha="center", fontsize=13.5, color=C_TITLE)
    ytxt = [8.9, 8.0, 7.1]
    ycol = [C_TEXT, C_AUDIO, C_VIS]
    dims = ["768", "74", "35"]
    for i, m in enumerate(C.MODS):
        box(0.2, ytxt[i] - 0.28, 2.1, 0.6, f"{MOD_CN[m]} $X^{{({m[0]})}}$ 50×{dims[i]}", ycol[i], fs=9.5, tc="white")
        box(2.55, ytxt[i] - 0.28, 1.6, 0.6, f"mask $M^{{({m[0]})}}$\n0/1 有效", "white", ec=C_GRAY, fs=8.5)
        arrow(2.3, ytxt[i], 2.53, ytxt[i])
        arrow(4.15, ytxt[i], 4.55, ytxt[i])
    box(4.6, 6.4, 1.75, 2.75, "标准化 + 投影\n(→128 维)\n+ 正弦位置编码", "#F2F6FA", ec=C_GRAY, fs=9)
    box(6.55, 6.05, 2.0, 3.45, "mask 感知\nTransformer\n编码器\n(key_padding_mask\n屏蔽缺失位)", "#EEF3F8", ec=C_TITLE, fs=9)
    for i in range(3):
        arrow(6.35, ytxt[i], 6.53, ytxt[i])
    box(8.75, 6.4, 1.85, 2.75, "跨模态注意力\n(每两两 MHA)\n+ 模态内专家路由\nMoE(K=2)", "#EAF4EC", ec="#4FB6A6", fs=8.8)
    for i in range(3):
        arrow(8.55, ytxt[i], 8.73, ytxt[i])
    box(10.8, 6.4, 1.6, 2.75, "覆盖率 c_m\n+ 缺失率/段数\n(有效区内)", "#FFF5E8", ec="#D9A441", fs=8.6)
    box(12.6, 6.9, 2.5, 1.35, "覆盖率先验门控\nα = softmax(g([h̄;c]) + β⊙log c_eff)", "#FDE9E7", ec=C_AUDIO, fs=8.8)
    box(12.6, 5.25, 2.5, 1.0, "加权融合 z = Σ α_m ê⁽ᵐ⁾\n+ LayerNorm", "#F2F6FA", ec=C_GRAY, fs=8.8)
    arrow(10.8, 7.8, 12.58, 7.6)
    arrow(12.6, 6.9, 13.85, 6.35)
    box(12.6, 3.2, 2.5, 1.6, "三头输出\n分类 softmax\n强度 μ=3tanh\n不确定度 log σ²", "#EDEAF6", ec="#6C5B9E", fs=8.8)
    arrow(13.85, 5.25, 13.85, 4.82)
    ax.text(1.15, 4.35, "训练期缺失增强：r~U(0.10,0.40)、段长≤8、25% 完整样本；缺失位特征=0、mask=0", fontsize=9, color=C_GRAY)
    ax.text(1.15, 3.75, "总损失：L = L_cls + 0.3 L_reg + 0.1 L_rec + 0.5 L_kd(τ=3) + 0.1 L_nll", fontsize=9.5, color=C_TITLE)
    ax.text(1.15, 3.1, "新增(对比v1)：覆盖率先验门控（修复门控不敏感）；去掉无效的模态嵌入/均值不变性损失", fontsize=9, color=C_GRAY)
    save(fig, "fv01_arch_v2")


# ---------------- fv02 缺失测试集缺失形态 ----------------
def fv02_missing():
    Mt, Ma, Mv, leff, ids = a3_masks_only()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    ax.imshow(Mt, aspect="auto", cmap="Blues", vmin=0, vmax=1.6, interpolation="nearest")
    for i, le in enumerate(leff):
        ax.plot([le - 0.5, le - 0.5], [i - 0.5, i + 0.5], color="#D08C60", lw=1.4)
    ax.set_xlabel("词级对齐位置 (0~49)"); ax.set_ylabel("缺失测试集样本序号")
    ax.set_title("(a) 缺失测试集文本掩码矩阵（蓝=有效，白=缺失/填充）")
    ax.set_yticks(range(0, 30, 5)); ax.set_yticklabels([i + 1 for i in range(0, 30, 5)])
    ax.text(0.98, 0.02, "橙线=有效长度边界", transform=ax.transAxes, ha="right",
            fontsize=8.5, color="#D08C60")
    st = np.array([D.missing_stats(Mt[i:i + 1], leff[i:i + 1])[0] for i in range(len(ids))])
    ax = axes[1]
    ax.hist(st[:, 0], bins=np.arange(-0.5, st[:, 0].max() + 1.5), color=C_TEXT, edgecolor="white")
    ax.set_xlabel("有效区内缺失格数"); ax.set_ylabel("样本数")
    ax.set_title(f"(b) 缺失规模（均值 {st[:,0].mean():.1f} 格，缺失率 {100*st[:,2].mean():.1f}%）")
    lens = []
    for i in range(len(ids)):
        z = (Mt[i, 1:int(leff[i])] == 0).astype(int)
        s = 0
        for v in list(z) + [0]:
            if v == 1:
                s += 1
            elif s > 0:
                lens.append(s); s = 0
    ax = axes[2]
    vc = pd.Series(lens).value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=C_AUDIO, edgecolor="white")
    ax.set_xlabel("单段缺失长度（格）"); ax.set_ylabel("段数")
    ax.set_title(f"(c) 缺失段长分布（共 {len(lens)} 段，均值 {np.mean(lens):.2f} 格）")
    fig.suptitle("缺失测试集模态局部缺失形态实测（对齐版 30 条）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fv02_missing_scheme_v2")


# ---------------- fv03 训练曲线 ----------------
def fv03_curves():
    hs = []
    for s in C.SEEDS:
        p = os.path.join(C.CKPT, f"final_v2_s{s}.pt")
        if os.path.exists(p):
            ck = T.load_ckpt(p)
            hist = ck.get("metrics", {}).get("hist", []) if isinstance(ck, dict) else []
            if hist:
                hs.append(hist)
    if not hs:
        print("skip fv03"); return
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    for h in hs:
        ax.plot([e["epoch"] for e in h], [e["loss"] for e in h], alpha=0.5, lw=1.1)
    L = min(len(h) for h in hs)
    m = np.mean([[e["loss"] for e in h[:L]] for h in hs], 0)
    ax.plot(range(1, L + 1), m, color=C_TITLE, lw=2.2, label="均值")
    ax.set_xlabel("epoch"); ax.set_ylabel("训练总损失")
    ax.set_title("(a) 训练损失"); ax.grid(alpha=0.25); ax.legend()
    for k, (key, title, col) in enumerate([("f1_macro", "(b) 验证 F1-macro", C_AUDIO),
                                            ("mae", "(c) 验证 MAE", C_TEXT),
                                            ("pearson", "(d) 验证 Pearson", C_VIS)]):
        ax = axes[(k + 1) // 2, (k + 1) % 2]
        for h in hs:
            ax.plot([e["epoch"] for e in h], [e["val_a3"][key] for e in h], alpha=0.45, lw=1.1, color=col)
        m = np.mean([[e["val_a3"][key] for e in h[:L]] for h in hs], 0)
        sd = np.std([[e["val_a3"][key] for e in h[:L]] for h in hs], 0)
        ax.plot(range(1, L + 1), m, color=C_TITLE, lw=2.2)
        ax.fill_between(range(1, L + 1), m - sd, m + sd, color=C_TITLE, alpha=0.15)
        ax.set_xlabel("epoch"); ax.set_ylabel(key); ax.set_title(title); ax.grid(alpha=0.25)
    fig.suptitle("最终方案 v2.1 训练曲线（3 随机种子，验证集缺失测试集同分布缺失）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fv03_train_curves_v2")


# ---------------- fv04 缺失类型消融 ----------------
def fv04_type():
    p = os.path.join(C.OUT, "ablation_type_v2.csv")
    if not os.path.exists(p):
        print("skip fv04"); return
    df = pd.read_csv(p)
    order = ["无缺失", "仅文本", "仅语音", "仅视觉", "文本+语音", "文本+视觉", "语音+视觉", "三模态"]
    metrics = [("f1_macro", "F1-macro"), ("acc", "Accuracy"), ("mae", "MAE"), ("pearson", "Pearson")]
    stdmap = {"f1_macro": "f1_std", "acc": "acc_std", "mae": "mae_std", "pearson": "pearson_std"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.2))
    x = np.arange(len(order))
    for k, (key, lab) in enumerate(metrics):
        ax = axes[k // 2, k % 2]
        mus = [df[df["condition"] == c][key].mean() for c in order]
        sds = [df[df["condition"] == c][stdmap[key]].mean() for c in order]
        bars = ax.bar(x, mus, 0.6, yerr=sds, capsize=3, color=C_TITLE, alpha=0.85,
                      error_kw=dict(ecolor="#555", lw=1))
        for b, v, sd in zip(bars, mus, sds):
            ax.text(b.get_x() + b.get_width() / 2, v + sd + 0.002, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
        _lo, _hi = min(mus), max(mus)
        _pad = 0.12 * (_hi - _lo) if _hi > _lo else 0.01
        ax.set_ylim(_lo - _pad, _hi + _pad)
        ax.set_ylabel(lab); ax.set_title(f"({'abcd'[k]}) 缺失模态类型 → {lab}")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("缺失模态类型对性能的影响（final_v2，10 次注入 mean±std，验证集）",
                 fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fv04_ablation_type_v2")


# ---------------- fv05 缺失位置消融 ----------------
def fv05_position():
    p = os.path.join(C.OUT, "ablation_position_v2.csv")
    if not os.path.exists(p):
        print("skip fv05"); return
    df = pd.read_csv(p)
    mods = ["文本", "语音", "视觉", "三模态"]; poss = ["前段", "中段", "后段"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for k, (key, lab) in enumerate([("f1_macro", "F1-macro"), ("mae", "MAE")]):
        M = np.zeros((len(mods), len(poss)))
        for i, m in enumerate(mods):
            for j, q in enumerate(poss):
                sel = df[df["condition"] == f"{m}-{q}"][key]
                M[i, j] = sel.mean() if len(sel) else np.nan
        ax = axes[k]
        im = ax.imshow(M, cmap="RdYlGn" if key == "f1_macro" else "RdYlGn_r", aspect="auto")
        for i in range(len(mods)):
            for j in range(len(poss)):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=11)
        ax.set_xticks(range(len(poss))); ax.set_xticklabels(poss)
        ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods)
        ax.set_title(f"({'ab'[k]}) 缺失位置 → {lab}（r=30%，单段）")
        plt.colorbar(im, ax=ax)
    fig.suptitle("缺失位置对性能的影响（最终方案 v2.1，验证集）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fv05_ablation_position_v2")


# ---------------- fv06 缺失率消融 ----------------
def fv06_rate():
    p = os.path.join(C.OUT, "ablation_rate_v2.csv")
    if not os.path.exists(p):
        print("skip fv06"); return
    df = pd.read_csv(p)
    order = ["无缺失（r=0）"] + [f"r={r}%" for r in [10, 20, 30, 40, 50]]
    xs = [0, 10, 20, 30, 40, 50]
    metrics = [("f1_macro", "F1-macro"), ("acc", "Accuracy"), ("mae", "MAE"), ("pearson", "Pearson")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    stdmap = {"f1_macro": "f1_std", "acc": "acc_std", "mae": "mae_std", "pearson": "pearson_std"}
    for k, (key, lab) in enumerate(metrics):
        ax = axes[k]
        mus = [df[df["condition"] == c][key].mean() for c in order]
        sds = [df[df["condition"] == c][stdmap[key]].mean() for c in order]
        ax.errorbar(xs, mus, yerr=sds, marker="o", ms=5, lw=1.8,
                    color=C_TITLE, capsize=3, label="final_v2")
        ax.fill_between(xs, [a - b for a, b in zip(mus, sds)],
                        [a + b for a, b in zip(mus, sds)], color=C_TITLE, alpha=0.12)
        ax.set_xlabel("缺失率 r (%)"); ax.set_ylabel(lab)
        ax.set_title(f"({'abcd'[k]}) {lab}"); ax.grid(alpha=0.25)
        ax.set_xticks(xs)
    fig.suptitle("缺失率对性能的影响（final_v2，10 次注入 mean±std，验证集）",
                 fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, "fv06_ablation_rate_v2")


# ---------------- fv07 混淆矩阵 ----------------
@torch.no_grad()
def fv07_confusion():
    dt, models = load_v2(T.device_of())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, mode, name in [(axes[0], "clean", "无缺失(clean)"), (axes[1], "a3", "缺失测试集同分布缺失")]:
        Ps, Rs = [], []
        for ck, m in models:
            P, R, yc, yr = T.eval_split(m, dt, "valid", T.device_of(), mode=mode, draws=C.N_EVAL_DRAWS)
            Ps.append(P); Rs.append(R)
        P = np.mean(Ps, 0); R = np.mean(Rs, 0)
        tau = float(np.mean([ck["tau"] for ck, _ in models]))
        pred = T.decision(P, R, tau, "band")
        cm = confusion_matrix(yc, pred, labels=[0, 1, 2])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["负", "中", "正"])
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["负", "中", "正"])
        ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title(name, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("最终方案 v2.1 混淆矩阵（3 种子集成，验证集）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fv07_confusion_v2")


# ---------------- fv08 强度散点 ----------------
@torch.no_grad()
def fv08_scatter():
    dt, models = load_v2(T.device_of())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for j, mode in enumerate(["clean", "a3"]):
        Ps, Rs = [], []
        for ck, m in models:
            P, R, yc, yr = T.eval_split(m, dt, "valid", T.device_of(), mode=mode, draws=C.N_EVAL_DRAWS)
            Ps.append(P); Rs.append(R)
        R = np.mean(Rs, 0)
        ax = axes[j]
        ax.scatter(yr, R, s=14, alpha=0.55,
                   c=[C_TEXT if abs(a - b) < 0.5 else C_AUDIO for a, b in zip(yr, R)],
                   edgecolors="none")
        lim = [-3.2, 3.2]
        ax.plot(lim, lim, "--", color=C_GRAY, lw=1.2)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("真实情感强度"); ax.set_ylabel("预测情感强度")
        mae = float(np.abs(R - yr).mean()); r = float(np.corrcoef(R, yr)[0, 1])
        ax.set_title(f"({'ab'[j]}) {'无缺失' if mode=='clean' else '缺失测试集同分布缺失'}  MAE={mae:.3f}  r={r:.3f}")
        ax.grid(alpha=0.25)
    fig.suptitle("情感强度回归散点（最终方案 v2.1，验证集）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fv08_scatter_v2")


# ---------------- fv09 覆盖率先验门控 ----------------
@torch.no_grad()
def fv09_gate():
    dt, models = load_v2(T.device_of())
    X, M, Lf = dt["X"]["valid"], dt["M"]["valid"], dt["L"]["valid"]
    rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    cols = {"text": C_TEXT, "audio": C_AUDIO, "vision": C_VIS}
    means = {m: [] for m in C.MODS}
    for r in rates:
        acc = {m: [] for m in C.MODS}
        for k in range(3):
            rng = np.random.default_rng(2024 + 61 * k)
            Xb = {m: X[m] for m in C.MODS}
            Mb = {m: M[m] for m in C.MODS}
            Lb = {m: Lf[m] for m in C.MODS}
            if r > 0:
                Xb, Mb, _ = D.inject(Xb, Mb, Lb, rng, modes=["text"], r=r,
                                     K=1, segmax=100, indep=False, clean_ratio=0.0)
            _, _, A, _ = ens_predict(models, Xb, Mb, T.device_of())
            for j, m in enumerate(C.MODS):
                acc[m].append(A[:, j])
        for m in C.MODS:
            means[m].append(float(np.mean(np.concatenate(acc[m]))))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for m in C.MODS:
        ax.plot(rates, means[m], "-o", label=f"α {MOD_CN[m]}", color=cols[m], lw=2)
    ax.set_xlabel("文本模态缺失率 r（有效区内）")
    ax.set_ylabel("平均门控权重 α")
    ax.set_title("覆盖率先验门控：文本缺失越多，α_text 单调下降（修复 v1 门控不敏感）", fontsize=11.5, color=C_TITLE)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    save(fig, "fv09_gate_v2")


# ---------------- fv10 缺失测试集预测 ----------------
def fv10_a3():
    df = pd.read_csv(os.path.join(C.OUT, "missing_modality_predictions.csv"))
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.2))
    vc = df["predicted_class_name"].value_counts().reindex(["Negative", "Neutral", "Positive"]).fillna(0)
    axes[0].bar(vc.index, vc.values, color=[C_AUDIO, "#B58B3A", C_VIS], edgecolor="white")
    for i, v in enumerate(vc.values):
        axes[0].text(i, v + 0.15, int(v), ha="center")
    axes[0].set_ylabel("样本数"); axes[0].set_title("(a) 预测极性分布"); axes[0].grid(alpha=0.25, axis="y")
    axes[1].hist(df["predicted_score"], bins=12, color=C_TEXT, edgecolor="white")
    axes[1].axvline(0, color=C_GRAY, ls="--")
    axes[1].set_xlabel("预测情感强度"); axes[1].set_ylabel("样本数")
    axes[1].set_title(f"(b) 强度分布（均值 {df['predicted_score'].mean():.2f}）")
    A = df[["alpha_text", "alpha_audio", "alpha_vision"]].values
    axes[2].bar(range(len(df)), A[:, 0], color=C_TEXT, label="文本")
    axes[2].bar(range(len(df)), A[:, 1], bottom=A[:, 0], color=C_AUDIO, label="语音")
    axes[2].bar(range(len(df)), A[:, 2], bottom=A[:, 0] + A[:, 1], color=C_VIS, label="视觉")
    axes[2].set_xlabel("缺失测试集样本序号"); axes[2].set_ylabel(r"模态作用程度 $\alpha_m$")
    axes[2].set_title("(c) 各样本模态门控权重"); axes[2].legend(fontsize=9)
    axes[3].scatter(df["score_uncertainty_sigma"], df["classification_entropy"], s=22,
                    c=C_TITLE, alpha=0.8, edgecolors="none")
    axes[3].set_xlabel("强度不确定度 σ"); axes[3].set_ylabel("分类熵")
    axes[3].set_title("(d) 不确定度（σ 与熵）")
    axes[3].grid(alpha=0.25)
    fig.suptitle("缺失测试集测试集最终推理结果汇总（30 条，v2.1）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fv10_a3_predictions_v2")


# ---------------- fv11 组件消融 ----------------
def fv11_component():
    p = os.path.join(C.OUT, "ablation_v2_components.csv")
    if not os.path.exists(p):
        print("skip fv11"); return
    df = pd.read_csv(p)
    ORDER = ["final_v2", "v2_nocov", "v2_nomoe", "v2_nokd", "v2_nonll"]
    LABEL = ["全开", "−覆盖率先验", "−专家路由", "−知识蒸馏", "−不确定性头"]
    COLOR = ["#2E5E9E", "#C55A11", "#4C8C4A", "#8A5A91", "#B8B8B8"]
    df = df[df["variant"].isin(ORDER)]
    means = [df[df["variant"] == v]["valid_a3_f1"].mean() for v in ORDER]
    stds = [df[df["variant"] == v]["valid_a3_f1"].std() for v in ORDER]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(range(len(ORDER)), means, yerr=stds, capsize=4, color=COLOR, alpha=0.9, width=0.62)
    ax.axhline(means[0], color="#C55A11", ls="--", lw=1.2, label=f"全开 F1={means[0]:.4f}")
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 0.001, f"{m:.4f}", ha="center", fontsize=9.5)
        if i > 0:
            ax.text(i, m - 0.012, f"{m-means[0]:+.4f}", ha="center", fontsize=9, color="#B00020")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(LABEL, fontsize=11)
    ax.set_ylabel("valid_a3 F1-macro")
    ax.set_ylim(min(means) - 0.04, max(means) + 0.03)
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("v2.1 组件逐项消融（去掉单个组件后的验证集 F1，3 种子）", fontsize=12, color=C_TITLE)
    save(fig, "fv11_component_v2")


# ---------------- fv12 极端缺失 ----------------
def fv12_extreme():
    p = os.path.join(C.OUT, "ablation_extreme_v2.csv")
    if not os.path.exists(p):
        print("skip fv12"); return
    df = pd.read_csv(p)
    conds = ["仅文本全缺失", "仅语音全缺失", "仅视觉全缺失", "三模态各50%", "三模态各70%",
             "散点缺失20%", "散点缺失40%"]
    metrics = [("f1_macro", "F1-macro"), ("acc", "Accuracy"), ("mae", "MAE"), ("pearson", "Pearson")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2))
    x = np.arange(len(conds))
    for k, (key, lab) in enumerate(metrics):
        ax = axes[k // 2, k % 2]
        mus = [df[df["condition"] == c][key].mean() for c in conds]
        bars = ax.bar(x, mus, 0.6, color=C_AUDIO, alpha=0.85)
        for b, v in zip(bars, mus):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=22, ha="right")
        _lo, _hi = min(mus), max(mus)
        _pad = 0.12 * (_hi - _lo) if _hi > _lo else 0.01
        ax.set_ylim(_lo - _pad, _hi + _pad)
        ax.set_ylabel(lab); ax.set_title(f"({'abcd'[k]}) 极端/未见缺失 → {lab}")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("极端与分布外缺失形态下的鲁棒性（最终方案 v2.1，验证集）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fv12_ablation_extreme_v2")


# ---------------- fv13 权衡 ----------------
def fv13_tradeoff():
    p = os.path.join(C.OUT, "ablation_v2_components.csv")
    if not os.path.exists(p):
        print("skip fv13"); return
    df = pd.read_csv(p)
    ORDER = ["final_v2", "v2_nocov", "v2_nomoe", "v2_nokd", "v2_nonll"]
    LABEL = ["全开", "−覆盖率先验", "−专家路由", "−知识蒸馏", "−不确定性头"]
    COLOR = ["#2E5E9E", "#C55A11", "#4C8C4A", "#8A5A91", "#B8B8B8"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for v, lab, c in zip(ORDER, LABEL, COLOR):
        d = df[df["variant"] == v]
        if len(d) == 0:
            continue
        ax.errorbar(d["valid_clean_f1"].mean(), d["valid_a3_f1"].mean(),
                    xerr=d["valid_clean_f1"].std(), yerr=d["valid_a3_f1"].std(),
                    marker="o", ms=9, color=c, capsize=3, lw=1.4, label=lab)
    lo = min(df["valid_clean_f1"].min(), df["valid_a3_f1"].min()) - 0.008
    hi = max(df["valid_clean_f1"].max(), df["valid_a3_f1"].max()) + 0.008
    ax.plot([lo, hi], [lo, hi], "--", color=C_GRAY, lw=1)
    ax.text(hi, hi, "无退化线", fontsize=9, color=C_GRAY, ha="right")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("无缺失 F1-macro"); ax.set_ylabel("缺失测试集同分布缺失 F1-macro")
    ax.set_title("完整性能 vs 缺失鲁棒性（v2.1 组件变体，3 种子）")
    ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.25)
    fig.suptitle("缺失几乎不造成性能退化", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fv13_tradeoff_v2")


# ---------------- fv14 不确定度校准 ----------------
@torch.no_grad()
def fv14_uncertainty():
    dt, models = load_v2(T.device_of())
    X, M = dt["X"]["valid"], dt["M"]["valid"]
    Ps, Rs, Vs = [], [], []
    for ck, m in models:
        p, r, a, lv = T.predict(m, X, M, T.device_of())
        Ps.append(p); Rs.append(r)
        if lv is not None:
            Vs.append(np.exp(lv))
    P = np.mean(Ps, 0); R = np.mean(Rs, 0); V = np.mean(Vs, 0)
    sig = np.sqrt(np.maximum(V, 0))
    err = np.abs(R - dt["items"]["valid"]["y_reg"])
    order = np.argsort(sig)
    buckets = np.array_split(order, 5)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs = [float(np.mean(sig[b])) for b in buckets]
    ys = [float(np.mean(err[b])) for b in buckets]
    ax.plot(xs, ys, "-o", color=C_TITLE, lw=2)
    ax.set_xlabel("预测不确定度 σ（均值）"); ax.set_ylabel("平均 |误差| (MAE)")
    ax.set_title(f"不确定度校准（验证集 clean，σ 与误差 Pearson={np.corrcoef(sig, err)[0,1]:.3f}）", fontsize=11.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fv14_uncertainty_v2")


# ---------------- fv15 思维流程图 ----------------
def fv15_mindmap():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 17); ax.set_ylim(0, 12.5); ax.axis("off")

    def box(x, y, w, h, txt, fc="#EAF1FA", ec=C_TITLE, fs=8.6, tc="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, color=C_GRAY, lw=1.3))

    ax.text(8.5, 12.0, "模块二 端到端方法论思维流程（MoDE-Former+ v2.1）", ha="center", fontsize=14, color=C_TITLE)

    box(0.3, 10.2, 2.7, 1.5, "① 数据理解\n标准化特征aligned_50.pkl\n三模态 (50,d) + 双标签", "#EAF1FA")
    box(0.3, 8.0, 2.7, 1.5, "② 缺失建模\n缺失测试集实测：r≈19.5%\n段长1~4、三模态同位置", "#EAF1FA")
    arrow(1.65, 10.2, 1.65, 9.55)
    box(3.6, 8.0, 2.8, 2.2, "③ 预处理\n有效格统计 μ/σ\n缺失/填充置 0 + mask=0\n标准化仅用有效格", "#F2F6FA")
    arrow(3.0, 8.7, 3.6, 8.9); arrow(3.0, 10.2, 3.6, 10.0)
    box(6.9, 8.0, 3.0, 2.2, "④ 模型结构 v2.1\n投影 → mask感知Transformer\n→ 跨模态+MoE → 覆盖率先验门控", "#EEF3F8")
    arrow(6.4, 8.7, 6.9, 8.7)
    box(10.4, 8.0, 3.0, 2.2, "⑤ 训练目标\nL = L_cls + 0.3L_reg + 0.1L_rec\n+ 0.5L_kd(τ=3) + 0.1L_nll", "#FFF3E0", ec="#C55A11")
    arrow(9.9, 8.7, 10.4, 8.7)
    box(13.9, 8.0, 2.7, 2.2, "⑥ 验证选参\n验证集选结构/超参/阈值\nλ坐标下降 + 3种子集成", "#FBECEE", ec=C_AUDIO)
    arrow(13.4, 8.7, 13.9, 8.7)

    box(0.3, 4.6, 2.7, 2.0, "⑦ 规律分析\n缺失类型/位置/缺失率\n/时长/连续vs分散", "#E8F5F1", ec=C_VIS)
    arrow(1.65, 8.0, 1.65, 6.65)
    box(3.6, 4.6, 2.8, 2.0, "⑧ 组件消融\n−cov / −MoE / −KD / −NLL\n归因各组件贡献", "#E8F5F1", ec=C_VIS)
    arrow(3.0, 6.6, 3.6, 6.0)
    box(6.9, 4.6, 3.0, 2.0, "⑨ 缺失测试集推理\nmask 感知前向\n3 种子集成平均", "#E8F5F1", ec=C_VIS)
    arrow(6.4, 6.6, 6.9, 6.0)
    box(10.4, 4.6, 3.0, 2.0, "⑩ 输出\n极性 + 强度 + 不确定度 σ\n门控 α + 缺失统计", "#F5F0E6", ec="#B58B3A")
    arrow(9.9, 6.6, 10.4, 6.0)
    box(13.9, 4.6, 2.7, 2.0, "11 结果交付\nmissing_modality_predictions.csv\n18 列全量预测", "#F5F0E6", ec="#B58B3A")
    arrow(13.4, 6.6, 13.9, 6.0)

    ax.text(8.5, 2.9, "核心思想①：0+mask 闭环 —— 缺失位特征=0、mask=0、Transformer 内逐层归零，补位 0 不参与统计；", ha="center", fontsize=9.8, color="#444444")
    ax.text(8.5, 2.15, "核心思想②：覆盖率先验门控 —— α = softmax(g([h̄;c]) + β⊙log c_eff)，让模型按“有效区内还剩下多少信息”分配信任；", ha="center", fontsize=9.8, color="#444444")
    ax.text(8.5, 1.4, "核心思想③：知识蒸馏让残缺输入模仿完整输入的软标签；异方差头输出 σ 量化不确定度", ha="center", fontsize=9.8, color="#444444")
    save(fig, "fv15_mindmap_v2")


# ---------------- fv16 缺失时长 ----------------
def fv16_duration():
    p = os.path.join(C.OUT, "ablation_duration_v2.csv")
    if not os.path.exists(p):
        print("skip fv16"); return
    df = pd.read_csv(p)
    groups = ["三模态", "仅文本", "仅语音", "仅视觉"]
    cols = {"三模态": C_TITLE, "仅文本": C_TEXT, "仅语音": C_AUDIO, "仅视觉": C_VIS}
    metrics = [("f1_macro", "F1-macro"), ("acc", "Accuracy"), ("mae", "MAE"), ("pearson", "Pearson")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0))
    for k, (key, lab) in enumerate(metrics):
        ax = axes[k]
        for g in groups:
            d = df[df["group"] == g].sort_values("duration")
            ax.plot(d["duration"], d[key], marker="o", ms=4, lw=1.7, color=cols[g], label=g)
        ax.set_xlabel("连续缺失时长（格，单段）")
        ax.set_ylabel(lab)
        ax.set_title(f"({'abcd'[k]}) {lab}")
        ax.grid(alpha=0.25)
        if k == 0:
            ax.legend(fontsize=9)
    fig.suptitle("连续缺失时长对性能的影响（最终方案 v2.1，单段中间缺失，验证集）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, "fv16_duration_v2")


# ---------------- fv17 网格搜索 ----------------
def fv17_grid():
    df = pd.read_csv(os.path.join(C.OUT, "grid_search_raw.csv"))
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, param in zip(axes, ["kd", "nll", "reg", "rec"]):
        sub = df[df["param"] == param].sort_values("value")
        if sub.empty:
            ax.set_title(f"{param} (无数据)"); continue
        ax.plot(sub["value"], sub["f1"], "-o", label="F1(a3)", color=C_TEXT)
        ax.plot(sub["value"], sub["mae"], "-s", label="MAE(a3)", color=C_AUDIO)
        ax2 = ax.twinx()
        ax2.plot(sub["value"], sub["pearson"], "-^", label="Pearson", color=C_TITLE, alpha=0.8)
        ax.set_xlabel("λ"); ax.set_title(f"λ_{param}"); ax.grid(alpha=0.3)
        if param == "kd":
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("λ 网格搜索（坐标下降，验证集选超参，seed42）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fv17_grid_search_v2")


# ---------------- fv18 连续 vs 分散 ----------------
def fv18_contig_disp():
    df = pd.read_csv(os.path.join(C.OUT, "ablation_contig_dispersed_v2.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, nm in [(axes[0], "f1", "F1-macro"), (axes[1], "mae", "MAE"), (axes[2], "pearson", "Pearson")]:
        for pat, lab, c in [("continuous", "连续单段", C_AUDIO), ("dispersed", "多段分散", C_TEXT)]:
            sub = df[df["pattern"] == pat].sort_values("rate")
            ax.errorbar(sub["rate"], sub[key], yerr=sub[key + "_std"], marker="o", ms=5, lw=2, color=c, capsize=3, label=lab)
            ax.fill_between(sub["rate"], sub[key] - sub[key + "_std"], sub[key] + sub[key + "_std"], color=c, alpha=0.12)
        ax.set_xlabel("缺失率 r"); ax.set_ylabel(nm); ax.grid(alpha=0.3); ax.legend()
        ax.set_xticks(list(sub["rate"]))
    fig.suptitle("同样缺失格数：连续单段 vs 多段分散（最终方案 v2.1，10次注入 mean±std，验证集）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fv18_contig_dispersed_v2")


def main():
    fns = [fv01_arch, fv02_missing, fv03_curves, fv04_type, fv05_position, fv06_rate,
           fv07_confusion, fv08_scatter, fv09_gate, fv10_a3, fv11_component, fv12_extreme,
           fv13_tradeoff, fv14_uncertainty, fv15_mindmap, fv16_duration, fv17_grid, fv18_contig_disp]
    for fn in fns:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"[skip] {fn.__name__}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print("all v2.1 figures done", flush=True)


if __name__ == "__main__":
    main()

