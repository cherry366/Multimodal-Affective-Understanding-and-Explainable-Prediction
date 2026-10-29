# -*- coding: utf-8 -*-
"""模块三 补充可视化：词级热力着色 / 跨模态注意力 / t-SNE。"""
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.manifold import TSNE

import config_q3 as C
import data_q3 as D3
import data_q2 as D
import train_q2 as T2
from explain_q3 import load_models, ens_predict

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 130

C_TEXT, C_AUDIO, C_VIS = "#4C9BC7", "#E4636F", "#4FB6A6"
C_TITLE, C_GRAY = "#1F3B6E", "#8A8A8A"
MOD_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}
CLS_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}
CLS_COLOR = {0: "#D05A6E", 1: "#B0A36A", 2: "#5B9B74"}
FIGS = C.Q3_FIGS


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), format=ext, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("FIG ->", name, flush=True)


# ================================================================= 词级热力着色
def word_importance(text_occ, nwords, L=50):
    imp = np.zeros(nwords, dtype=np.float32)
    for i in range(nwords):
        pos = i if nwords <= L else int(i * L / nwords)
        imp[i] = abs(float(text_occ[min(pos, L - 1)]))
    return imp


def draw_word_heat(ax, raw_text, imp, cmap, title, ev_vals=None):
    """词级热力着色：证据词旁标注遮挡 Δlogit（φ 数值）。"""
    words = raw_text.strip().split()
    mx = imp.max()
    if mx <= 1e-9:
        mx = 1.0
    ev_vals = ev_vals or {}
    x = 0.02; y = 0.86; line_h = 0.135; char_w = 0.0106
    for i, w in enumerate(words):
        v = float(imp[i]) / mx
        fc = cmap(v) if v > 0.02 else "#FFFFFF"
        is_ev = w in ev_vals
        ax.text(x, y, w, fontsize=9.2, va="center", ha="left", color="#111111",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=fc,
                          edgecolor="#B03A2E" if is_ev else "none", linewidth=0.8))
        ww = len(w) * char_w + 0.010
        if is_ev:
            val = float(ev_vals[w])
            col = "#1F6E43" if val >= 0 else "#B03A2E"
            ax.text(x + ww * 0.35, y - 0.030, f"{val:+.2f}", fontsize=5.4,
                    va="top", ha="left", color=col, fontweight="bold")
            ww += 0.055
        x += ww
        if x > 0.965:
            x = 0.02; y -= line_h
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(title, fontsize=11, color=C_TITLE)


def qx11_word_heat():
    expl = json.load(open(os.path.join(C.Q3_OUT, "explainable_explanations.json"), encoding="utf-8"))
    df = pd.read_csv(os.path.join(C.Q3_OUT, "explainable_predictions.csv"), dtype={"id": str})
    cmap = plt.get_cmap("YlOrRd")
    picks = [("09", "Negative"), ("17", "Positive"), ("01", "Neutral")]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.8))
    for ax, (sid, want) in zip(axes, picks):
        e = expl[sid]
        occ = e["local_importance_occlusion"]["text"]
        words = e["raw_text"].strip().split()
        imp = word_importance(occ, len(words))
        # 证据词 -> 其对应位置的遮挡 Δlogit（φ 数值，带符号）
        ev_vals = {}
        for w, t in zip(e["evidence_text"]["words"], e["evidence_text"]["positions"]):
            ev_vals[w] = float(occ[min(max(int(t), 0), 49)])
        pred = e["prediction"]
        dom = e["dominant_modality"]
        draw_word_heat(ax, e["raw_text"], imp, cmap,
                       f"a4_{sid} · 预测 {pred['polarity_name']} · ŝ={pred['intensity']:+.2f} · 主导 {MOD_CN[dom]}",
                       ev_vals)
    fig.suptitle("文本词级重要性热力着色（颜色越深 = 越关键；红框证据词下方为遮挡 Δlogit 数值）", fontsize=12.5, color=C_TITLE)
    fig.suptitle("文本词级重要性热力着色（颜色越深 = 对预测越关键，遮挡 LOPO 归因）", fontsize=12.5, color=C_TITLE)
    # 色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.015)
    cbar.set_label("归一化重要性")
    save(fig, "qx11_word_heat")


# ================================================================= 跨模态注意力
def cross_attention(models, Xn, Mn, dev):
    """返回 {name: (50,50)} 三种子平均的跨模态注意力权重。"""
    agg = {}
    for ck, m in models:
        att = {}
        def make_hook(name):
            def hook(module, inp, outp):
                w = outp[1].detach().cpu()
                if w.ndim == 4:
                    w = w.mean(1)
                att[name] = w[0]
            return hook
        handles = []
        for name, mod in m.cross.items():
            handles.append(mod.register_forward_hook(make_hook(name)))
        Xt = {k: torch.tensor(Xn[k], device=dev) for k in C.MODS}
        Mt = {k: torch.tensor(Mn[k], device=dev) for k in C.MODS}
        m.eval()
        with torch.no_grad():
            m(Xt, Mt)
        for h in handles:
            h.remove()
        for name, w in att.items():
            agg.setdefault(name, []).append(w.numpy())
    return {n: np.mean(v, 0) for n, v in agg.items()}


def qx12_cross_attention():
    models, dt = load_models()
    dev = T2.device_of()
    stats = T2._C["stats"]
    a4 = D3.load_all_a4()
    Xn = {m: D.apply_norm(a4[m], a4["mask"][m], *stats[m]) for m in C.MODS}
    Mn = a4["mask"]

    picks = [("09", "Negative"), ("17", "Positive")]
    pairs = ["text<-audio", "text<-vision", "audio<-text", "audio<-vision", "vision<-text", "vision<-audio"]
    pair_cn = ["T ← A", "T ← V", "A ← T", "A ← V", "V ← T", "V ← A"]

    fig, axes = plt.subplots(2, 6, figsize=(17.5, 6.2))
    for r, (sid, want) in enumerate(picks):
        b = int(sid) - 1
        Xb = {m: Xn[m][b:b+1] for m in C.MODS}
        Mb = {m: Mn[m][b:b+1] for m in C.MODS}
        att = cross_attention(models, Xb, Mb, dev)
        for c, (pr, cn) in enumerate(zip(pairs, pair_cn)):
            ax = axes[r, c]
            W = att[pr]
            im = ax.imshow(W, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
            ax.set_title(cn, fontsize=9)
            if r == 1:
                ax.set_xlabel("源位置", fontsize=7.5)
            if c == 0:
                ax.set_ylabel(f"a4_{sid}\n目标位置", fontsize=7.5)
            ax.tick_params(labelsize=6)
            ax.grid(False)
    fig.suptitle("跨模态注意力权重热力图（每格 = 目标位置对源位置的注意力，三种子平均）", fontsize=12.5, color=C_TITLE)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015).set_label("注意力权重", fontsize=8)
    save(fig, "qx12_cross_attention")


# ================================================================= t-SNE
def fusion_z(models, Xn, Mn, dev):
    """返回三种子平均融合向量 z，(N, H)。"""
    Zs = []
    for ck, m in models:
        Xt = {k: torch.tensor(Xn[k], device=dev) for k in C.MODS}
        Mt = {k: torch.tensor(Mn[k], device=dev) for k in C.MODS}
        m.eval()
        with torch.no_grad():
            z = m(Xt, Mt)["z"].cpu().numpy()
        Zs.append(z)
    return np.mean(Zs, 0)


def qx13_tsne():
    models, dt = load_models()
    dev = T2.device_of()
    Xv = dt["X"]["valid"]
    Mv = dt["M"]["valid"]
    yc = dt["items"]["valid"]["y_cls"]
    Z = fusion_z(models, Xv, Mv, dev)
    print("z shape", Z.shape, flush=True)
    ts = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    E = ts.fit_transform(Z)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, title in zip(axes, ["按真实类别", "按预测类别"]):
        if title == "按真实类别":
            labels = yc
        else:
            P, R, A, V = ens_predict(models, Xv, Mv, dev)
            labels = T2.decision(P, R, float(np.mean([ck["tau"] for ck, _ in models])), "band")
        for c in (0, 1, 2):
            m = labels == c
            ax.scatter(E[m, 0], E[m, 1], s=9, alpha=0.55, color=CLS_COLOR[c], label=CLS_NAME[c], edgecolors="none")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, markerscale=2.2, framealpha=0.9)
        ax.grid(alpha=0.2)
    fig.suptitle("融合特征 t-SNE 嵌入（验证集 728 条，128 维融合向量 z）", fontsize=12.5, color=C_TITLE)
    save(fig, "qx13_tsne")


# ================================================================= 多模态证据时间轴对齐图
def qx14_evidence_timeline():
    """证据时间轴对齐图（每个样本单独一张，字号放大便于正文阅读）。"""
    expl = json.load(open(os.path.join(C.Q3_OUT, "explainable_explanations.json"), encoding="utf-8"))
    df = pd.read_csv(os.path.join(C.Q3_OUT, "explainable_predictions.csv"), dtype={"id": str})
    picks = [("09", "Negative", "neg"), ("17", "Positive", "pos"), ("01", "Neutral", "neu")]
    cmap_txt = plt.get_cmap("YlOrRd")
    for k, (sid, want, tag) in enumerate(picks):
        e = expl[sid]
        row = df[df["id"] == sid].iloc[0]
        D = float(e["evidence_audio"]["duration"])
        fig = plt.figure(figsize=(11.2, 5.6))
        gs = fig.add_gridspec(3, 1, height_ratios=[0.80, 0.30, 1.05], hspace=0.80,
                              left=0.05, right=0.975, top=0.865, bottom=0.095)
        axT = fig.add_subplot(gs[0])
        axA = fig.add_subplot(gs[1], sharex=axT)
        axV = fig.add_subplot(gs[2], sharex=axT)

        fig.suptitle(
            f"样本 a4_{sid} · 预测 {row['predicted_class_name']} · ŝ={row['predicted_score']:+.2f} · "
            f"主导模态 {MOD_CN[row['dominant_modality']]} · "
            f"模态作用 α=[{row['alpha_text']:.2f}, {row['alpha_audio']:.2f}, {row['alpha_vision']:.2f}]",
            fontsize=13.5, color=C_TITLE, y=0.975)

        # ---- 文本轨（两行）----
        words = e["raw_text"].strip().split()
        n = max(len(words), 1)
        half = (n + 1) // 2
        occ_t = e["local_importance_occlusion"]["text"]
        ev_val = {}
        for w, p in zip(e["evidence_text"]["words"], e["evidence_text"]["positions"]):
            ev_val[w] = float(occ_t[min(max(int(p), 0), 49)])
        mx_t = max([abs(v) for v in ev_val.values()] + [1e-9])
        m = 0.045 * D
        span = max(D - 2 * m, 1e-6)
        for i in range(n):
            w = words[i]
            line = 0 if i < half else 1
            j = i if line == 0 else i - half
            cnt = half if line == 0 else (n - half)
            x = m + (j + 0.5) / max(cnt, 1) * span
            y = 0.74 if line == 0 else 0.28
            if w in ev_val:
                v = abs(ev_val[w]) / mx_t
                val_c = 0.25 + 0.75 * v
                axT.text(x, y, w, ha="center", va="center", fontsize=12, clip_on=False,
                         color=("white" if val_c > 0.58 else "#111111"),
                         bbox=dict(boxstyle="round,pad=0.30", facecolor=cmap_txt(val_c),
                                   edgecolor="#B03A2E", lw=0.9))
                axT.text(x, y + 0.205, f"{ev_val[w]:+.2f}", ha="center", va="center",
                         fontsize=9.5, color="#B03A2E" if ev_val[w] < 0 else "#1F6E43",
                         fontweight="bold")
            else:
                axT.text(x, y, w, ha="center", va="center", fontsize=12, color="#333333", clip_on=False)
        axT.set_ylim(0, 1); axT.set_yticks([]); axT.tick_params(labelbottom=False)
        for sp in ("top", "right", "left"):
            axT.spines[sp].set_visible(False)
        axT.text(0.0, 1.05, "文本轨（句级证据；色深 ∝ |Δlogit|）",
                 transform=axT.transAxes, fontsize=10, color=C_GRAY, va="bottom")

        # ---- 语音轨 ----
        occ_a = e["local_importance_occlusion"]["audio"]
        axA.add_patch(Rectangle((0, 0.30), D, 0.40, facecolor="#EDEDED", edgecolor="none"))
        mx_a = max([abs(float(occ_a[min(max(int(p), 0), 49)]))
                    for p in e["evidence_audio"]["positions"]] + [1e-9])
        for (s0, s1), p in zip(e["evidence_audio"]["time_windows"], e["evidence_audio"]["positions"]):
            v = abs(float(occ_a[min(max(int(p), 0), 49)])) / mx_a
            axA.add_patch(Rectangle((s0, 0.30), max(s1 - s0, 0.03), 0.40, facecolor=C_AUDIO,
                                    alpha=0.25 + 0.75 * v, edgecolor="white", lw=0.8))
        axA.set_ylim(0, 1); axA.set_yticks([]); axA.tick_params(labelbottom=False)
        for sp in ("top", "right", "left"):
            axA.spines[sp].set_visible(False)
        axA.text(0.0, 1.07, "语音轨（关键时段，色深 ∝ |Δlogit|）",
                 transform=axA.transAxes, fontsize=10, color=C_GRAY, va="bottom")

        # ---- 视觉轨 ----
        frames = e["evidence_vision"]["key_frames"]
        vpos = e["evidence_vision"]["positions"]
        occ_v = e["local_importance_occlusion"]["vision"]
        items = []
        for f, p in zip(frames, vpos):
            if f and os.path.exists(f):
                items.append((float((int(p) + 0.5) * D / 50.0), int(p), f,
                              abs(float(occ_v[min(max(int(p), 0), 49)]))))
        items.sort()
        mx_v = max([it[3] for it in items] + [1e-9])
        min_gap = D * 0.175
        placed, last = [], -1e9
        for t, p, f, v in items:
            x = min(max(t, last + min_gap), D - min_gap * 0.45)
            placed.append(x); last = x
        hw = min_gap * 0.40
        for (t, p, f, v), x in zip(items, placed):
            try:
                axV.imshow(mpimg.imread(f), extent=[x - hw, x + hw, 0.36, 1.04], aspect="auto", zorder=3)
            except Exception:
                axV.add_patch(Rectangle((x - hw, 0.36), 2 * hw, 0.68, facecolor="#DDDDDD", zorder=3))
            axV.add_patch(Rectangle((x - hw, 0.36), 2 * hw, 0.68, fill=False, edgecolor=C_VIS,
                                    lw=0.8 + 1.8 * v / mx_v, zorder=4))
            axV.plot([x, t], [0.36, 0.15], ls=":", color=C_GRAY, lw=0.9, zorder=5)
            axV.plot([t], [0.15], marker="o", ms=3.6, color=C_VIS, zorder=5)
            axV.text(x, 0.02, f"t{p:02d} · {t:.1f}s", ha="center", va="bottom",
                     fontsize=9, color="#555555")
        axV.axhline(0.15, color="#BBBBBB", lw=0.9, zorder=1)
        axV.set_ylim(0, 1.10); axV.set_xlim(0, D); axV.set_yticks([])
        axV.set_xlabel("时间 (s)", fontsize=11.5)
        axV.tick_params(axis="x", labelsize=10.5)
        for sp in ("top", "right", "left"):
            axV.spines[sp].set_visible(False)
        axV.text(0.0, 1.04, "视觉轨（关键帧按真实时间落位，虚线 = 真实时刻）",
                 transform=axV.transAxes, fontsize=10, color=C_GRAY, va="bottom")
        save(fig, f"qx14{'abc'[k]}_evidence_timeline_{tag}")


def main():
    for fn in (qx11_word_heat, qx12_cross_attention, qx13_tsne, qx14_evidence_timeline):
        try:
            fn()
        except Exception as ex:
            import traceback
            print(f"[ERR] {fn.__name__}: {ex}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

