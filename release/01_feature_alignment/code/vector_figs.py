# -*- coding: utf-8 -*-
"""模块一 文档展示图：全部以 PDF 输出到 q1_output/vector_figs/

用法: python vector_figs.py
输出: q1_output/vector_figs/*.pdf
"""
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import pipeline as P

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42          # 嵌入 TrueType, 保证 PDF 中文正常
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12

OUTDIR = os.path.join(C.OUT, "vector_figs")
os.makedirs(OUTDIR, exist_ok=True)


def save(fig, name):
    p = os.path.join(OUTDIR, name + ".pdf")
    fig.savefig(p, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("PDF ->", p, flush=True)


def load_tb():
    with open(C.PKL_TIMEBIN, "rb") as f:
        return pickle.load(f)["test"]


def load_recs():
    return json.load(open(os.path.join(C.OUT, "align_records.json"), encoding="utf-8"))


# ---------- 图1 视频时长分布 ----------
def fig01():
    recs = load_recs()
    dur = np.array([r["duration"] for r in recs], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(dur, bins=20, color="#2b6cb0", edgecolor="white")
    ax.axvline(dur.mean(), color="#c53030", ls="--", lw=1.6,
               label="均值 %.2f s" % dur.mean())
    ax.axvline(np.median(dur), color="#2f855a", ls=":", lw=1.6,
               label="中位 %.2f s" % np.median(dur))
    ax.set_xlabel("视频时长 / s"); ax.set_ylabel("样本数")
    ax.set_title("原始视频 100 条视频时长分布")
    ax.legend(); ax.grid(alpha=0.25)
    save(fig, "fig01_duration_hist")


# ---------- 图2 情感极性 + 强度分布 ----------
def fig02():
    df = pd.read_excel(C.LABEL_XLSX)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    order = ["Negative", "Neutral", "Positive"]
    cnt = df["annotation"].value_counts().reindex(order).fillna(0).astype(int)
    axes[0].bar(order, cnt.values, color=["#c53030", "#718096", "#2f855a"])
    for i, v in enumerate(cnt.values):
        axes[0].text(i, v + 0.6, str(int(v)), ha="center", fontsize=11)
    axes[0].set_ylabel("样本数"); axes[0].set_ylim(0, cnt.max() * 1.18)
    axes[0].set_title("(a) 情感极性分布"); axes[0].grid(alpha=0.25, axis="y")
    lab = df["label"].astype(float)
    axes[1].hist(lab, bins=20, color="#6b46c1", edgecolor="white")
    axes[1].axvline(0, color="#c53030", ls="--", lw=1.4)
    axes[1].set_xlabel("连续情感强度"); axes[1].set_ylabel("样本数")
    axes[1].set_title("(b) 情感强度分布"); axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    save(fig, "fig02_label_dist")


# ---------- 图3 三模态有效覆盖率 ----------
def fig03():
    recs = load_recs()
    cov = {
        "文本": [r["timebin_text_valid"] / C.N_BINS for r in recs],
        "语音": [r["timebin_audio_valid"] / C.N_BINS for r in recs],
        "视觉": [r["timebin_vision_valid"] / C.N_BINS for r in recs],
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bp = ax.boxplot([cov["文本"], cov["语音"], cov["视觉"]], patch_artist=True,
                    labels=["文本", "语音", "视觉"], widths=0.5)
    for patch, c in zip(bp["boxes"], ["#4299e1", "#48bb78", "#ed8936"]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for i, k in enumerate(["文本", "语音", "视觉"], start=1):
        v = np.array(cov[k])
        ax.scatter(np.random.normal(i, 0.045, len(v)), v, s=8, color="black", alpha=0.45)
        ax.text(i, 1.02, "均值 %.3f" % v.mean(), ha="center", fontsize=10)
    ax.set_ylabel("50 时间箱有效覆盖率"); ax.set_ylim(-0.03, 1.10)
    ax.set_title("三模态时间箱有效覆盖率（100 样本）")
    ax.grid(alpha=0.25, axis="y")
    save(fig, "fig03_coverage")


# ---------- 图4 词级对齐质量 ----------
def fig04():
    recs = load_recs()
    ratio = np.array([r.get("word_match_ratio", 0.0) for r in recs], dtype=float)
    mode = pd.Series([r.get("text_align", "?") for r in recs]).value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    axes[0].hist(ratio, bins=20, range=(0, 1), color="#3182ce", edgecolor="white")
    axes[0].axvline(0.5, color="#c53030", ls="--", lw=1.5, label="置信阈值 0.5")
    axes[0].set_xlabel("词匹配率 word_match_ratio"); axes[0].set_ylabel("样本数")
    axes[0].set_title("(a) 词级对齐置信度分布"); axes[0].legend(); axes[0].grid(alpha=0.25, axis="y")
    axes[1].bar(mode.index, mode.values,
                color=["#2f855a" if "whisper" in m else "#dd6b20" for m in mode.index])
    for i, v in enumerate(mode.values):
        axes[1].text(i, v + 0.8, str(int(v)), ha="center")
    axes[1].set_ylabel("样本数"); axes[1].set_ylim(0, mode.max() * 1.2)
    axes[1].set_title("(b) 对齐策略分布"); axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    save(fig, "fig04_wordmatch")


# ---------- 图5 典型样本多模态对齐总览 ----------
def _pick_typical(tb):
    recs = load_recs()
    hi = [r for r in recs if r.get("word_match_ratio", 0) >= 0.8]
    pool = hi if hi else recs
    pool = sorted(pool, key=lambda r: (-(r["timebin_audio_valid"] + r["timebin_vision_valid"]),
                                       abs(r["n_words"] - 15)))
    rec = pool[0]
    return rec, list(tb["id"]).index(rec["id"])


def fig05():
    import soundfile as sf
    tb = load_tb()
    rec, i = _pick_typical(tb)
    sid, T, words = rec["id"], float(rec["duration"]), rec["words"]
    a = np.load(os.path.join(C.FEAT_AUDIO, sid + ".npy"))
    v = np.load(os.path.join(C.FEAT_VISION, sid + ".npy"))
    y, sr = sf.read(os.path.join(C.AUDIO_DIR, sid + ".wav"), dtype="float32")
    t_wave = np.arange(len(y)) / sr
    edges = np.arange(C.N_BINS + 1) * T / C.N_BINS

    fig, axes = plt.subplots(5, 1, figsize=(13, 14),
                             gridspec_kw={"height_ratios": [1.0, 0.7, 1.6, 1.2, 1.6]})
    ax = axes[0]
    ax.plot(t_wave, y, lw=0.35, color="#2b6cb0")
    ax.set_xlim(0, T); ax.set_ylabel("幅度")
    ax.set_title("典型样本 %s (时长 %.2fs, %d 词, 对齐 %s, 词匹配率 %.2f)"
                 % (sid, T, len(words), rec["text_align"], rec.get("word_match_ratio", 0)))
    ax = axes[1]
    for k, w in enumerate(words):
        ax.barh(0, w["end"] - w["start"], left=w["start"], height=0.5,
                color=plt.cm.tab20(k % 20), edgecolor="k", linewidth=0.4)
        if len(words) <= 25 or k % 2 == 0:
            ax.text((w["start"] + w["end"]) / 2, 0.04, w["token"],
                    ha="center", va="bottom", fontsize=7)
    ax.set_yticks([]); ax.set_xlim(0, T); ax.set_ylabel("词")
    ax.set_xlabel("时间 / s")
    ax = axes[2]
    im = ax.imshow(a.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, T, 0, a.shape[1]])
    ax.set_ylabel("语音 74 维")
    for e in edges[::5]:
        ax.axvline(e, color="w", alpha=0.15, lw=0.5)
    plt.colorbar(im, ax=ax, fraction=0.02)
    ax = axes[3]
    im = ax.imshow(v.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, T, 0, v.shape[1]])
    ax.set_ylabel("视觉 35 维"); plt.colorbar(im, ax=ax, fraction=0.02)
    ax = axes[4]
    comb = np.hstack([tb["text_norm"][i], tb["audio_norm"][i], tb["vision_norm"][i]])
    im = ax.imshow(comb.T, aspect="auto", origin="lower", cmap="viridis")
    n = C.N_BINS
    ax.axvline(n - 0.5, color="white", lw=2); ax.axvline(2 * n - 0.5, color="white", lw=2)
    ax.set_xticks([n / 2 - 0.5, 1.5 * n - 0.5, 2.5 * n - 0.5])
    ax.set_xticklabels(["文本 768维", "语音 74维", "视觉 35维"])
    ax.set_xlabel("时间箱 k (1..50)"); ax.set_ylabel("标准化特征")
    plt.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    save(fig, "fig05_typical_sample")


# ---------- 图6 时间箱对齐机制示意 ----------
def fig06():
    tb = load_tb()
    rec, i = _pick_typical(tb)
    sid, T = rec["id"], float(rec["duration"])
    v_times = np.load(os.path.join(C.FEAT_VISION, sid + ".times.npy")).astype(float)
    a_times = (np.arange(tb["n_audio_frames"][i]) + 0.5) / C.AUDIO_FPS
    edges = np.arange(C.N_BINS + 1) * T / C.N_BINS
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.0),
                             gridspec_kw={"height_ratios": [1.0, 1.3]})
    ax = axes[0]
    ax.vlines(v_times, 0.55, 0.95, color="#ed8936", lw=0.8, alpha=0.8)
    ax.vlines(a_times[::5], 0.05, 0.45, color="#4299e1", lw=0.8, alpha=0.6)
    for e in edges:
        ax.axvline(e, color="gray", lw=0.35, alpha=0.5)
    ax.set_yticks([0.25, 0.75]); ax.set_yticklabels(["语音帧 100Hz\n(每5帧示1)", "视觉帧\n原生fps"])
    ax.set_xlim(0, T); ax.set_ylim(0, 1.0)
    ax.set_title("时间箱对齐机制: I_k=[kT/50,(k+1)T/50), 按真实时间戳落箱 (样本 %s)" % sid)
    ax = axes[1]
    im = ax.imshow(tb["audio_norm"][i].T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, C.N_BINS, 0, C.AUDIO_DIM])
    ax.set_xlabel("时间箱索引 k (1..50)"); ax.set_ylabel("语音 74 维")
    plt.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    save(fig, "fig06_timebin_scheme")


# ---------- 图7 跨模态相关性热力图 ----------
def fig07():
    tb = load_tb()
    mat, _ = P.compute_cross_modal_corr(tb)
    labels = ["文本", "语音", "视觉"]
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            val = mat[i, j]
            ax.text(j, i, "%.3f" % val, ha="center", va="center", fontsize=14,
                    fontweight="bold", color="black" if val < 0.65 else "white")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_title("跨模态平均绝对皮尔逊相关 |ρ|")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|r|")
    save(fig, "fig07_corr_heatmap")


# ---------- 图8 音视互相关曲线 ----------
def fig08():
    tb = load_tb()
    sync_df = P.compute_av_sync(tb)
    curves = np.load(os.path.join(C.ANALYSIS_DIR, "sync_curves.npy"))
    taus = np.load(os.path.join(C.ANALYSIS_DIR, "sync_taus.npy"))
    cand = sync_df[sync_df["valid_ratio"] >= 0.5]
    if len(cand) == 0:
        cand = sync_df
    idx = int(cand["sync_score"].idxmax())
    row = sync_df.iloc[idx]
    dt = float(row.get("frame_dt", 1.0 / C.VISION_FPS))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(taus * dt, curves[idx], "o-", color="#2b6cb0", lw=1.6, ms=5)
    ax.axvline(float(row["peak_lag_sec"]), color="#c53030", ls="--", lw=1.4,
               label="peak_lag = %.3f s" % float(row["peak_lag_sec"]))
    ax.set_xlabel("时移 τ / s"); ax.set_ylabel("归一化互相关 C(τ)")
    ax.set_title("音视同步互相关曲线 (样本 %s, sync_score=%.3f)"
                 % (row["id"], float(row["sync_score"])))
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, "fig08_ncc_curve")


# ---------- 图9 音视同步滞后分布 ----------
def fig09():
    tb = load_tb()
    sync_df = P.compute_av_sync(tb)
    lag = sync_df["peak_lag_sec"].values.astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(lag, bins=24, color="#2b6cb0", edgecolor="white", alpha=0.92)
    ax.axvline(float(np.median(lag)), color="#c53030", ls="--", lw=1.6,
               label="中位数 %.3f s" % float(np.median(lag)))
    within = float(np.mean(np.abs(lag) <= 0.2))
    ax.set_xlabel("peak_lag / s"); ax.set_ylabel("样本数")
    ax.set_title("100 样本音视同步滞后分布 (|lag|≤0.2s 占比 %.1f%%)" % (within * 100))
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, "fig09_lag_histogram")


def main():
    for fn in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09]:
        try:
            fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            print("[FAIL]", fn.__name__, e, flush=True)
    print("== 矢量图( PDF )已输出到", OUTDIR, "==", flush=True)


if __name__ == "__main__":
    main()
