# -*- coding: utf-8 -*-
"""模块一 全套可视化：相关性热力图 + 互相关曲线 + 滞后直方图 + 增强典型样本图

用法: python figures.py
输出: q1_output/figs/
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

LABELS = ["文本 Text", "语音 Audio", "视觉 Vision"]


def load_tb():
    with open(C.PKL_TIMEBIN, "rb") as f:
        return pickle.load(f)["test"]


def _pick_typical(tb):
    recs = json.load(open(os.path.join(C.OUT, "align_records.json"), encoding="utf-8"))
    hi = [r for r in recs if r.get("word_match_ratio", 0) >= 0.8]
    pool = hi if hi else recs
    pool = sorted(pool, key=lambda r: (-(r["timebin_audio_valid"] + r["timebin_vision_valid"]),
                                       abs(r["n_words"] - 15)))
    rec = pool[0]
    i = list(tb["id"]).index(rec["id"])
    return rec, i


# ---------------- Step10 相关性热力图 ----------------
def fig_corr_heatmap(tb):
    mat, corr_df = P.compute_cross_modal_corr(tb)
    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    for i in range(3):
        for j in range(3):
            val = mat[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="black" if val < 0.65 else "white")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
    ax.set_title("跨模态平均绝对皮尔逊相关矩阵  |ρ|  (100 样本聚合)", fontsize=12, pad=12)
    for i in range(3):
        for j in range(3):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="white", lw=1.5))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|r|")
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.tight_layout()
    out = os.path.join(C.FIGS_DIR, "corr_heatmap.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已保存:", out)


# ---------------- Step11 互相关曲线 ----------------
def fig_ncc_curve(tb):
    sync_df = P.compute_av_sync(tb)
    curves = np.load(os.path.join(C.ANALYSIS_DIR, "sync_curves.npy"))
    taus = np.load(os.path.join(C.ANALYSIS_DIR, "sync_taus.npy"))
    cand = sync_df[sync_df["valid_ratio"] >= 0.5]
    if len(cand) == 0:
        cand = sync_df
    idx = int(cand["sync_score"].idxmax())
    row = sync_df.iloc[idx]
    sid = row["id"]
    lag_sec = float(row["peak_lag_sec"])
    dt = float(row.get("frame_dt", 1.0 / C.VISION_FPS))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(taus * dt, curves[idx], "o-", color="#2b6cb0", lw=1.6, ms=5)
    ax.axvline(lag_sec, color="red", ls="--", lw=1.3,
               label=f"峰值滞后 peak_lag = {lag_sec:.3f} s")
    ax.axhline(float(row["sync_score"]), color="gray", ls=":", lw=1.0)
    ax.set_xlabel("时移 τ / 秒")
    ax.set_ylabel("归一化互相关 C(τ)")
    ax.set_title(f"音视同步互相关曲线  |  样本 {sid}  |  "
                 f"sync_score={row['sync_score']:.3f}  sharpness={row['sharpness']:.1f}")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(C.FIGS_DIR, "ncc_curve_typical.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已保存:", out)


# ---------------- Step11 滞后直方图 ----------------
def fig_lag_histogram(tb):
    sync_df = P.compute_av_sync(tb)
    lag = sync_df["peak_lag_sec"].values.astype(float)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(lag, bins=24, color="#2b6cb0", edgecolor="white", alpha=0.9)
    med = float(np.median(lag))
    ax.axvline(med, color="red", ls="--", lw=1.6, label=f"中位数 = {med:.3f} s")
    within = float(np.mean(np.abs(lag) <= 0.2))
    ax.set_xlabel("peak_lag / 秒")
    ax.set_ylabel("样本数")
    ax.set_title(f"100 样本音视同步滞后分布  |  |lag|≤0.2s 占比 = {within:.1%}")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(C.FIGS_DIR, "lag_histogram.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已保存:", out)


# ---------------- 增强典型样本图(波形+词+特征+时间箱) ----------------
def fig_typical_sample(tb):
    import soundfile as sf
    rec, i = _pick_typical(tb)
    sid = rec["id"]; T = float(rec["duration"]); words = rec["words"]
    a = np.load(os.path.join(C.FEAT_AUDIO, sid + ".npy"))
    v = np.load(os.path.join(C.FEAT_VISION, sid + ".npy"))
    y, sr = sf.read(os.path.join(C.AUDIO_DIR, sid + ".wav"), dtype="float32")
    t_wave = np.arange(len(y)) / sr

    bin_edges = np.arange(C.N_BINS + 1) * T / C.N_BINS

    fig, axes = plt.subplots(5, 1, figsize=(14, 15),
                             gridspec_kw={"height_ratios": [1.0, 0.7, 1.6, 1.2, 1.6]})

    ax = axes[0]
    ax.plot(t_wave, y, lw=0.35, color="#2b6cb0")
    ax.set_xlim(0, T); ax.set_ylabel("幅度")
    ax.set_title(f"典型样本 {sid}  |  时长 {T:.2f}s  |  {len(words)} 词  |  "
                 f"对齐 {rec['text_align']}  |  词匹配率 {rec.get('word_match_ratio',0):.2f}", fontsize=11)

    ax = axes[1]
    for k, w in enumerate(words):
        ax.barh(0, w["end"] - w["start"], left=w["start"], height=0.5,
                color=plt.cm.tab20(k % 20), edgecolor="k", linewidth=0.4)
        cx = (w["start"] + w["end"]) / 2
        if len(words) <= 25 or k % 2 == 0:
            ax.text(cx, 0.02, w["token"], ha="center", va="bottom",
                    fontsize=7, rotation=0)
    ax.set_yticks([]); ax.set_xlim(0, T)
    ax.set_ylabel("词"); ax.set_xlabel("时间 / 秒")

    ax = axes[2]
    im = ax.imshow(a.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, T, 0, a.shape[1]])
    ax.set_ylabel("语音 74 维")
    for e in bin_edges[::5]:
        ax.axvline(e, color="w", alpha=0.15, linewidth=0.5)
    plt.colorbar(im, ax=ax, fraction=0.02)

    ax = axes[3]
    im = ax.imshow(v.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, T, 0, v.shape[1]])
    ax.set_ylabel("视觉 35 维")
    plt.colorbar(im, ax=ax, fraction=0.02)

    ax = axes[4]
    n = C.N_BINS
    tn = tb["text_norm"][i]; an = tb["audio_norm"][i]; vn = tb["vision_norm"][i]
    comb = np.hstack([tn, an, vn])
    im = ax.imshow(comb.T, aspect="auto", origin="lower", cmap="viridis")
    ax.axvline(n - 0.5, color="white", lw=2)
    ax.axvline(2 * n - 0.5, color="white", lw=2)
    ax.set_xticks([n / 2 - 0.5, 1.5 * n - 0.5, 2.5 * n - 0.5])
    ax.set_xticklabels(["文本 768维", "语音 74维", "视觉 35维"], fontsize=10)
    ax.set_xlabel("时间箱 k (1..50) × 三模态")
    ax.set_ylabel("特征维度(标准化)")
    plt.colorbar(im, ax=ax, fraction=0.02)

    plt.tight_layout()
    out = os.path.join(C.FIGS_DIR, "typical_sample_full.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已保存:", out)


def main():
    os.makedirs(C.FIGS_DIR, exist_ok=True)
    tb = load_tb()
    fig_corr_heatmap(tb)
    fig_ncc_curve(tb)
    fig_lag_histogram(tb)
    fig_typical_sample(tb)
    print("== 全部图已生成到", C.FIGS_DIR, "==")


if __name__ == "__main__":
    main()
