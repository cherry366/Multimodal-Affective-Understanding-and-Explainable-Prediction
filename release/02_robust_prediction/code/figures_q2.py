# -*- coding: utf-8 -*-
"""模块二 全套图（PDF + PNG，中文已嵌入）

用法: python figures_q2.py
输出: q2_output\figs\*.pdf / *.png
"""
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import confusion_matrix

import config_q2 as C
import data_q2 as D
import train_q2 as T
from model_q2 import Q2Net

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.dpi"] = 120

C_TEXT, C_AUDIO, C_VIS = "#4C9BC7", "#E4636F", "#4FB6A6"
C_TITLE, C_GRAY = "#1F3B6E", "#8A8A8A"
MOD_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}
VARIANT_CN = {"full": "完整模型(mask+门控+增强+重建)", "norecon": "无重建项",
              "mask_only": "仅mask注意力", "gate_only": "仅门控",
              "nomask": "朴素基线(0当真值)", "noaug": "无缺失增强"}
VAR_ORDER = ["full", "norecon", "mask_only", "gate_only", "noaug", "nomask"]
VAR_COLOR = {"full": C_TITLE, "norecon": C_TEXT, "mask_only": C_AUDIO,
             "gate_only": C_VIS, "noaug": "#B58B3A", "nomask": "#999999"}


def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(C.FIGS, f"{name}.{ext}")
        fig.savefig(p, format=ext, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("FIG ->", name, flush=True)


# ------------------------------------------------------------------ fig01
def fig01_arch():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")

    def box(x, y, w, h, txt, fc, ec="none", fs=10, tc="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2, c=C_GRAY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, color=c, lw=1.3))

    ytxt = [5.3, 4.6, 3.9]
    ycol = [C_TEXT, C_AUDIO, C_VIS]
    yld = [r"$X^{(T)}\!\in\mathbb{R}^{50\times768}$", r"$X^{(A)}\!\in\mathbb{R}^{50\times74}$",
           r"$X^{(V)}\!\in\mathbb{R}^{50\times35}$"]
    for i, m in enumerate(C.MODS):
        box(0.15, ytxt[i] - 0.28, 1.9, 0.56, f"{MOD_CN[m]}模态\n{yld[i]}", ycol[i], fs=9.5, tc="white")
        box(2.35, ytxt[i] - 0.28, 1.5, 0.56, f"mask $M^{{({m[0].upper()})}}$\n0/1 序列", "#FFFFFF",
            ec=C_GRAY, fs=9)
        arrow(2.05, ytxt[i], 2.33, ytxt[i])
        arrow(3.85, ytxt[i], 4.35, 3.9 + 1.4 * (1 - i * 0.5))
        box(4.4, ytxt[i] - 0.28, 1.75, 0.56, "标准化+投影\n→128维 + 位置编码", "#F2F6FA",
            ec=C_GRAY, fs=8.5)
    box(6.35, 3.05, 1.85, 2.55, "mask 感知\nTransformer\n编码器\n(key_padding_mask\n屏蔽缺失位)",
        "#EEF3F8", ec=C_TITLE, fs=9)
    for i in range(3):
        arrow(6.15, ytxt[i], 6.33, ytxt[i])
    box(8.35, 4.35, 1.5, 1.15, "跨模态注意力\nMulT(6 对)\n缺失位不作 key", "#E8F5F1",
        ec=C_VIS, fs=8.5)
    box(8.35, 3.05, 1.5, 1.15, "模态置信度门控\n$\\alpha=\\mathrm{softmax}(\\cdot)$\n(覆盖率 $c_m$ 参与)",
        "#FBECEE", ec=C_AUDIO, fs=8.5)
    arrow(8.25, 4.9, 8.33, 4.9)
    arrow(8.25, 3.6, 8.33, 3.6)
    box(5.6, 1.55, 4.25, 1.05, "融合表示 $z$ → 标准化(LayerNorm)", "#F5F0E6", ec="#B58B3A", fs=10)
    arrow(9.1, 3.03, 8.6, 2.62)
    box(5.55, 0.25, 2.0, 0.95, "分类头 softmax\n→ 情感极性", "#E7EEF6", ec=C_TITLE, fs=9)
    box(7.85, 0.25, 2.0, 0.95, "回归头 $3\\tanh(\\cdot)$\n→ 情感强度[-3,3]", "#FBE9E7", ec=C_AUDIO, fs=9)
    arrow(7.0, 1.53, 6.6, 1.22); arrow(8.4, 1.53, 8.85, 1.22)
    ax.text(5.0, 6.15, "模块二 鲁棒多模态情感预测模型结构（mask 感知 + 跨模态融合 + 双头输出）",
            ha="center", fontsize=13, color=C_TITLE)
    ax.text(0.15, 2.55, "目标函数：$\\mathcal{L}=\\mathcal{L}_{CE}+\\lambda\\mathcal{L}_{SmoothL1}"
                        "+\\mu\\mathcal{L}_{recon}$   缺失增强：随机模态/位置/时长置0且mask=0",
            fontsize=9.5, color="#444444")
    save(fig, "fig01_model_arch")


# ------------------------------------------------------------------ fig02
def a3_masks_only():
    files = sorted(glob.glob(os.path.join(C.A3_DIR, "*.pkl")))
    Mt, Ma, Mv, leff = [], [], [], []
    for f in files:
        s = pickle.load(open(f, "rb"))["test"]
        A = np.asarray(s["audio"], dtype=np.float32)[0]
        V = np.asarray(s["vision"], dtype=np.float32)[0]
        TB = np.asarray(s["text_bert"]).astype(np.int64).reshape(3, C.L)
        mt = TB[1].astype(np.float32) * (TB[0] != 100) * (TB[0] != 0)
        Mt.append(mt); Ma.append(D.row_valid(A[None])[0]); Mv.append(D.row_valid(V[None])[0])
        leff.append(D.leff_of(mt[None])[0])
    return (np.stack(Mt), np.stack(Ma), np.stack(Mv), np.array(leff),
            [os.path.splitext(os.path.basename(f))[0] for f in files])


def fig02_missing_scheme():
    Mt, Ma, Mv, leff, ids = a3_masks_only()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    ax.imshow(Mt, aspect="auto", cmap="Blues", vmin=0, vmax=1.6, interpolation="nearest")
    for i, le in enumerate(leff):
        ax.plot([le - 0.5, le - 0.5], [i - 0.5, i + 0.5], color="#D08C60", lw=1.4)
    ax.set_xlabel("词级对齐位置 (0~49)"); ax.set_ylabel("缺失测试集样本序号")
    ax.set_title("(a) 缺失测试集掩码矩阵（蓝=有效，白=缺失/填充）")
    ax.set_yticks(range(0, 30, 5)); ax.set_yticklabels([i + 1 for i in range(0, 30, 5)])
    ax.text(0.98, 0.02, "橙线=有效长度边界", transform=ax.transAxes, ha="right",
            fontsize=8.5, color="#D08C60")

    st = np.array([D.missing_stats(Mt[i:i + 1], leff[i:i + 1])[0] for i in range(len(ids))])
    ax = axes[1]
    ax.hist(st[:, 0], bins=np.arange(-0.5, st[:, 0].max() + 1.5), color=C_TEXT, edgecolor="white")
    ax.set_xlabel("有效区内缺失格数"); ax.set_ylabel("样本数")
    ax.set_title(f"(b) 缺失测试集缺失规模（均值 {st[:,0].mean():.1f} 格，"
                 f"缺失率 {100*st[:,2].mean():.1f}%）")

    lens = []
    for i in range(len(ids)):
        z = Mt[i, 1:int(leff[i])] == 0
        z = z.astype(int)
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
    fig.suptitle("图2  缺失测试集模态局部缺失形态实测（对齐版 30 条）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig02_missing_scheme")


# ------------------------------------------------------------------ fig03
def load_hist():
    hs = []
    for s in C.SEEDS:
        p = os.path.join(C.CKPT, f"full_s{s}.pt")
        if os.path.exists(p):
            hs.append(T.load_ckpt(p)["metrics"]["hist"])
    return hs


def fig03_train_curves():
    hs = load_hist()
    if not hs:
        print("skip fig03: no ckpt"); return
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    for h in hs:
        ax.plot([e["epoch"] for e in h], [e["loss"] for e in h], alpha=0.55, lw=1.2)
    L = min(len(h) for h in hs)
    m = np.mean([[e["loss"] for e in h[:L]] for h in hs], 0)
    ax.plot(range(1, L + 1), m, color=C_TITLE, lw=2.2, label="均值")
    ax.set_xlabel("epoch"); ax.set_ylabel("训练损失"); ax.set_title("(a) 训练损失")
    ax.legend(); ax.grid(alpha=0.25)

    keys = [("f1_macro", "(b) 验证 F1-macro（缺失测试集同分布缺失）", C_AUDIO),
            ("mae", "(c) 验证 MAE", C_TEXT),
            ("pearson", "(d) 验证 Pearson 相关", C_VIS)]
    for k, (key, title, col) in enumerate(keys):
        ax = axes[(k + 1) // 2, (k + 1) % 2]
        for h in hs:
            ax.plot([e["epoch"] for e in h], [e["val_a3"][key] for e in h], alpha=0.45, lw=1.1, color=col)
        m = np.mean([[e["val_a3"][key] for e in h[:L]] for h in hs], 0)
        sd = np.std([[e["val_a3"][key] for e in h[:L]] for h in hs], 0)
        ax.plot(range(1, L + 1), m, color=C_TITLE, lw=2.2, label="均值")
        ax.fill_between(range(1, L + 1), m - sd, m + sd, color=C_TITLE, alpha=0.15)
        ax.set_xlabel("epoch"); ax.set_ylabel(key); ax.set_title(title); ax.grid(alpha=0.25)
    fig.suptitle("图3  完整模型训练曲线（3 个随机种子）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig03_train_curves")


# ------------------------------------------------------------------ fig04
def fig04_ablation_type():
    p = os.path.join(C.OUT, "ablation_type.csv")
    if not os.path.exists(p):
        print("skip fig04"); return
    df = pd.read_csv(p)
    df = df[df["split"] == "valid"]
    order = ["无缺失", "仅文本", "仅语音", "仅视觉", "文本+语音", "文本+视觉", "语音+视觉", "三模态"]
    variants = ["full", "nomask", "noaug"]
    metrics = [("f1_macro", "F1-macro"), ("acc", "Accuracy"), ("mae", "MAE"), ("pearson", "Pearson")]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6))
    x = np.arange(len(order)); w = 0.26
    for k, (key, lab) in enumerate(metrics):
        ax = axes[k // 2, k % 2]
        for j, v in enumerate(variants):
            d = df[df["variant"] == v]
            mus, sds = [], []
            for c in order:
                g = d[d["condition"] == c][key]
                mus.append(g.mean()); sds.append(g.std())
            ax.bar(x + (j - 1) * w, mus, w, yerr=sds, capsize=3,
                   color=VAR_COLOR[v], label=VARIANT_CN[v])
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
        _all = np.array([m for m, sd in zip(mus, sds)])
        _lo = float(np.min([m - sd for m, sd in zip(mus, sds)]))
        _hi = float(np.max([m + sd for m, sd in zip(mus, sds)]))
        _pad = max(0.10 * (_hi - _lo), 2e-3)
        ax.set_ylim(_lo - _pad, _hi + _pad)
        ax.set_ylabel(lab); ax.set_title(f"({'abcd'[k]}) 缺失模态类型 → {lab}（纵轴截断放大）")
        ax.grid(alpha=0.25, axis="y")
        if k == 0:
            _h, _l = ax.get_legend_handles_labels()
    fig.legend(_h, _l, loc="lower center", ncol=3, fontsize=9.5, frameon=False)
    fig.suptitle("图4  缺失模态类型对性能的影响（缺失率 20%，验证集，mean±std/3种子）",
                 fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save(fig, "fig04_ablation_type")


# ------------------------------------------------------------------ fig05
def fig05_ablation_position():
    p = os.path.join(C.OUT, "ablation_position.csv")
    if not os.path.exists(p):
        print("skip fig05"); return
    df = pd.read_csv(p)
    df = df[(df["split"] == "valid") & (df["variant"] == "full")]
    mods = ["文本", "语音", "视觉", "三模态"]
    poss = ["前段", "中段", "后段"]
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
    fig.suptitle("图5  缺失位置对性能的影响（验证集，完整模型）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig05_ablation_position")


# ------------------------------------------------------------------ fig06
def fig06_ablation_rate():
    p = os.path.join(C.OUT, "ablation_rate.csv")
    if not os.path.exists(p):
        print("skip fig06"); return
    df = pd.read_csv(p)
    df = df[df["split"] == "valid"]
    order = ["无缺失（r=0）"] + [f"r={r}%" for r in [10, 20, 30, 40, 50]]
    xs = [0, 10, 20, 30, 40, 50]
    metrics = [("f1_macro", "F1-macro"), ("acc", "Accuracy"), ("mae", "MAE"), ("pearson", "Pearson")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    for k, (key, lab) in enumerate(metrics):
        ax = axes[k]
        for v in VAR_ORDER:
            d = df[df["variant"] == v]
            if len(d) == 0:
                continue
            ys = [d[d["condition"] == c][key].mean() for c in order]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=VAR_COLOR[v], label=VARIANT_CN[v])
        ax.set_xlabel("缺失率 r (%)"); ax.set_ylabel(lab)
        ax.set_title(f"({'abcd'[k]}) {lab}"); ax.grid(alpha=0.25)
        if k == 3:
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("图6  缺失率对性能的影响（缺失模态按训练同分布随机，验证集）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, "fig06_ablation_rate")


# ------------------------------------------------------------------ fig07/08/09
def get_full_models(dev):
    out = []
    for s in C.SEEDS:
        p = os.path.join(C.CKPT, f"full_s{s}.pt")
        if not os.path.exists(p):
            continue
        ck = T.load_ckpt(p)
        m = Q2Net(use_key_mask=True, use_gate=True, use_cross=True,
                  use_recon=ck["spec"]["use_recon"]).to(dev)
        m.load_state_dict(ck["state"]); m.eval()
        out.append((ck, m))
    return out


def fig07_confusion():
    dev = T.device_of()
    dt = T.get_data()
    mdls = get_full_models(dev)
    if not mdls:
        print("skip fig07"); return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
    labels = ["Negative", "Neutral", "Positive"]
    for j, mode in enumerate(["clean", "a3"]):
        Ps, Rs = [], []
        for ck, m in mdls:
            P, R, yc, yr = T.eval_split(m, dt, "valid", dev, mode=mode, draws=C.N_EVAL_DRAWS)
            Ps.append(P); Rs.append(R)
        P = np.mean(Ps, 0); R = np.mean(Rs, 0)
        tau, tmode = mdls[0][0]["tau"], mdls[0][0]["tau_mode"]
        pred = T.decision(P, R, tau, tmode)
        cm = confusion_matrix(yc, pred, labels=[0, 1, 2])
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        for i, (mat, ttl, fmt) in enumerate([(cm, "计数", "d"), (cmn, "行归一化", ".2f")]):
            ax = axes[i, j]
            im = ax.imshow(mat, cmap="Blues")
            for a in range(3):
                for b in range(3):
                    v = mat[a, b]
                    ax.text(b, a, (f"{v:.2f}" if fmt == ".2f" else f"{int(v)}"),
                            ha="center", va="center",
                            color="white" if v > mat.max() * 0.6 else "black", fontsize=10)
            ax.set_xticks(range(3)); ax.set_xticklabels(labels, rotation=15)
            ax.set_yticks(range(3)); ax.set_yticklabels(labels)
            ax.set_xlabel("预测"); ax.set_ylabel("真值")
            ax.set_title(f"({'ac'[j] if i == 0 else 'bd'[j]}) {'无缺失' if mode=='clean' else '缺失测试集同分布缺失'}-{ttl}")
            plt.colorbar(im, ax=ax)
    fig.suptitle("图7  混淆矩阵（完整模型，3 种子集成，验证集）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig07_confusion")


def fig08_scatter():
    dev = T.device_of()
    dt = T.get_data()
    mdls = get_full_models(dev)
    if not mdls:
        print("skip fig08"); return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for j, mode in enumerate(["clean", "a3"]):
        Ps, Rs = [], []
        for ck, m in mdls:
            P, R, yc, yr = T.eval_split(m, dt, "valid", dev, mode=mode, draws=C.N_EVAL_DRAWS)
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
        ax.set_title(f"({'ab'[j]}) {'无缺失' if mode=='clean' else '缺失测试集同分布缺失'}"
                     f"   MAE={mae:.3f}  r={r:.3f}")
        ax.grid(alpha=0.25)
    fig.suptitle("图8  情感强度回归散点（完整模型，验证集）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig08_scatter")


def fig09_gate():
    dev = T.device_of()
    dt = T.get_data()
    mdls = get_full_models(dev)
    if not mdls:
        print("skip fig09"); return
    X, M, Lf = dt["X"]["valid"], dt["M"]["valid"], dt["L"]["valid"]
    N = X["text"].shape[0]
    alphas = {0: [], 1: [], 2: [], 3: []}
    for k in range(C.N_EVAL_DRAWS):
        rng = np.random.default_rng(999 + 31 * k)
        Xb = {m: X[m] for m in C.MODS}; Mb = {m: M[m] for m in C.MODS}
        Lb = {m: Lf[m] for m in C.MODS}
        Xb2, Mb2, _ = D.a3_like_inject(Xb, Mb, Lb, rng)
        A = np.mean([T.predict(m, Xb2, Mb2, dev)[2] for ck, m in mdls], 0)
        st = np.array([D.missing_stats(Mb2[m], Lb[m]) for m in C.MODS])
        nmod = (np.stack([st[i][:, 0] > 0 for i in range(3)]).sum(0))
        for n in range(4):
            sel = nmod == n
            if sel.sum():
                alphas[n].append(A[sel])
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    xs, data, cols = [], [], []
    for n in range(4):
        if not alphas[n]:
            continue
        A = np.concatenate(alphas[n], 0)
        xs.append(n)
        data.append([A[:, 0], A[:, 1], A[:, 2]])
    if not data:
        print("skip fig09 (no data)"); return
    pos = np.arange(len(xs))
    w = 0.25
    for i, m in enumerate(C.MODS):
        mus = [np.mean(d[i]) for d in data]
        sds = [np.std(d[i]) for d in data]
        ax.bar(pos + (i - 1) * w, mus, w, yerr=sds, capsize=3,
               color=[C_TEXT, C_AUDIO, C_VIS][i], label=f"{MOD_CN[m]}模态权重")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{n} 个模态缺失\n(有效样本 {len(np.concatenate(alphas[n], 0))} 条次)" for n in xs])
    ax.set_ylabel(r"门控权重 $\alpha_m$")
    ax.set_title("图9  模态置信度门控随缺失程度的变化（验证集，完整模型）")
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    save(fig, "fig09_gate")


# ------------------------------------------------------------------ fig10
def fig10_a3_predictions():
    p = os.path.join(C.OUT, "missing_modality_predictions.csv")
    if not os.path.exists(p):
        print("skip fig10"); return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    vc = df["predicted_class_name"].value_counts().reindex(["Negative", "Neutral", "Positive"]).fillna(0)
    ax.bar(vc.index, vc.values, color=[C_AUDIO, "#B58B3A", C_VIS], edgecolor="white")
    for i, v in enumerate(vc.values):
        ax.text(i, v + 0.15, int(v), ha="center")
    ax.set_ylabel("样本数"); ax.set_title("(a) 缺失测试集预测极性分布"); ax.grid(alpha=0.25, axis="y")
    ax = axes[1]
    ax.hist(df["predicted_score"], bins=12, color=C_TEXT, edgecolor="white")
    ax.axvline(0, color=C_GRAY, ls="--")
    ax.set_xlabel("预测情感强度"); ax.set_ylabel("样本数")
    ax.set_title(f"(b) 缺失测试集预测强度分布（均值 {df['predicted_score'].mean():.2f}）")
    ax = axes[2]
    A = df[["alpha_text", "alpha_audio", "alpha_vision"]].values
    ax.bar(range(len(df)), A[:, 0], color=C_TEXT, label="文本")
    ax.bar(range(len(df)), A[:, 1], bottom=A[:, 0], color=C_AUDIO, label="语音")
    ax.bar(range(len(df)), A[:, 2], bottom=A[:, 0] + A[:, 1], color=C_VIS, label="视觉")
    ax.set_xlabel("缺失测试集样本序号"); ax.set_ylabel(r"模态作用程度 $\alpha_m$")
    ax.set_title("(c) 各样本模态作用程度（门控权重）"); ax.legend(fontsize=9)
    fig.suptitle("图10  缺失测试集测试集预测结果汇总（全量 30 条）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig10_a3_predictions")


# ------------------------------------------------------------------ fig11
def fig11_component():
    p = os.path.join(C.OUT, "ablation_component.csv")
    if not os.path.exists(p):
        print("skip fig11"); return
    df = pd.read_csv(p)
    df = df[df["split"] == "valid"]
    conds = ["无缺失", "缺失测试集同分布(r≈19.5%)", "三模态r=20%", "仅文本r=20%", "仅语音r=20%", "仅视觉r=20%"]
    metrics = [("f1_macro", "F1-macro"), ("acc", "Accuracy"), ("mae", "MAE"), ("pearson", "Pearson")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.6))
    x = np.arange(len(conds)); w = 0.13
    vs = VAR_ORDER
    for k, (key, lab) in enumerate(metrics):
        ax = axes[k // 2, k % 2]
        for j, v in enumerate(vs):
            d = df[df["variant"] == v]
            mus = [d[d["condition"] == c][key].mean() for c in conds]
            ax.bar(x + (j - (len(vs) - 1) / 2) * w, mus, w, color=VAR_COLOR[v], label=VARIANT_CN[v])
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=18, ha="right")
        _v = df[key].values.astype(float)
        _lo, _hi = float(np.min(_v)), float(np.max(_v))
        _pad = max(0.10 * (_hi - _lo), 2e-3)
        ax.set_ylim(_lo - _pad, _hi + _pad)
        ax.set_ylabel(lab); ax.set_title(f"({'abcd'[k]}) 组件消融 → {lab}（纵轴截断放大）")
        ax.grid(alpha=0.25, axis="y")
        if k == 0:
            _h, _l = ax.get_legend_handles_labels()
    fig.legend(_h, _l, loc="lower center", ncol=6, fontsize=8.5, frameon=False)
    fig.suptitle("图11  组件消融：mask / 门控 / 缺失增强 / 重建项 的贡献（验证集）",
                 fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    save(fig, "fig11_component")


# ------------------------------------------------------------------ fig12
def fig12_ablation_extreme():
    p = os.path.join(C.OUT, "ablation_extreme.csv")
    if not os.path.exists(p):
        print("skip fig12"); return
    df = pd.read_csv(p)
    df = df[df["split"] == "valid"]
    conds = ["仅文本全缺失", "仅语音全缺失", "仅视觉全缺失", "三模态各50%", "三模态各70%",
             "散点缺失20%", "散点缺失40%"]
    variants = ["full", "mask_only", "gate_only", "noaug", "nomask"]
    metrics = [("f1_macro", "F1-macro"), ("acc", "Accuracy"), ("mae", "MAE"), ("pearson", "Pearson")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.8))
    x = np.arange(len(conds)); w = 0.16
    for k, (key, lab) in enumerate(metrics):
        ax = axes[k // 2, k % 2]
        for j, v in enumerate(variants):
            d = df[df["variant"] == v]
            mus = [d[d["condition"] == c][key].mean() for c in conds]
            ax.bar(x + (j - (len(variants) - 1) / 2) * w, mus, w,
                   color=VAR_COLOR[v], label=VARIANT_CN[v])
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=22, ha="right")
        _v = df[key].values.astype(float)
        _lo, _hi = float(np.min(_v)), float(np.max(_v))
        _pad = max(0.10 * (_hi - _lo), 2e-3)
        ax.set_ylim(_lo - _pad, _hi + _pad)
        ax.set_ylabel(lab); ax.set_title(f"({'abcd'[k]}) 极端/未见缺失 → {lab}（纵轴截断放大）")
        ax.grid(alpha=0.25, axis="y")
        if k == 0:
            _h, _l = ax.get_legend_handles_labels()
    fig.legend(_h, _l, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("图12  极端与分布外缺失形态下的鲁棒性对比（验证集）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save(fig, "fig12_ablation_extreme")


# ------------------------------------------------------------------ fig13
def fig13_tradeoff():
    p = os.path.join(C.OUT, "train_summary.csv")
    if not os.path.exists(p):
        print("skip fig13"); return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for v in VAR_ORDER:
        d = df[df["variant"] == v]
        if len(d) == 0:
            continue
        ax.errorbar(d["valid_clean_f1_macro"].mean(), d["valid_a3_f1_macro"].mean(),
                    xerr=d["valid_clean_f1_macro"].std(), yerr=d["valid_a3_f1_macro"].std(),
                    marker="o", ms=9, color=VAR_COLOR[v], capsize=3, lw=1.4,
                    label=VARIANT_CN[v])
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.set_xlabel("无缺失 F1-macro"); ax.set_ylabel("缺失测试集同分布缺失 F1-macro")
    ax.set_title("(a) 完整性能 vs 缺失鲁棒性")
    ax.grid(alpha=0.25)
    lo = min(df["valid_clean_f1_macro"].min(), df["valid_a3_f1_macro"].min()) - 0.02
    hi = max(df["valid_clean_f1_macro"].max(), df["valid_a3_f1_macro"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color=C_GRAY, lw=1)
    ax.text(hi, hi, "无退化线", fontsize=8.5, color=C_GRAY, ha="right")

    ax = axes[1]
    d = df.groupby("variant")["valid_a3_f1_macro"].mean() - df.groupby("variant")["valid_clean_f1_macro"].mean()
    d = d.reindex([v for v in VAR_ORDER if v in d.index])
    cols = [VAR_COLOR[v] for v in d.index]
    ax.barh([VARIANT_CN[v] for v in d.index], d.values, color=cols)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("ΔF1-macro（缺失 − 无缺失；>0 表示提升）")
    ax.set_title("(b) 缺失测试集同分布缺失导致的 F1 变化（均在因果波动区间内）")
    ax.grid(alpha=0.25, axis="x")
    fig.suptitle("图13  鲁棒性与基础性能的权衡（验证集，3 种子）", fontsize=13, color=C_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig13_tradeoff")


FIGS = [("fig01_model_arch", fig01_arch), ("fig02_missing_scheme", fig02_missing_scheme),
        ("fig03_train_curves", fig03_train_curves), ("fig04_ablation_type", fig04_ablation_type),
        ("fig05_ablation_position", fig05_ablation_position), ("fig06_ablation_rate", fig06_ablation_rate),
        ("fig07_confusion", fig07_confusion), ("fig08_scatter", fig08_scatter),
        ("fig09_gate", fig09_gate), ("fig10_a3_predictions", fig10_a3_predictions),
        ("fig11_component", fig11_component), ("fig12_ablation_extreme", fig12_ablation_extreme),
        ("fig13_tradeoff", fig13_tradeoff)]


def main():
    for name, fn in FIGS:
        try:
            fn()
        except Exception as e:
            import traceback
            print("FAILED", name, repr(e), flush=True)
            traceback.print_exc()
    print("figures done", flush=True)


if __name__ == "__main__":
    main()