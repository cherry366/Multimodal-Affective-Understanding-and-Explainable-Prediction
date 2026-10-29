# -*- coding: utf-8 -*-
"""典型样本对齐可视化：波形/词时间戳 + 原帧特征 + 对齐后时间箱特征

用法: python visualize.py
输出: q1_output/typical_sample.png
"""
import os
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def pick_sample():
    recs = json.load(open(os.path.join(C.OUT, "align_records.json"), encoding="utf-8"))
    # 选一条音视有效帧都多、词数接近中位的样本作为典型
    # 优先展示"词级对齐置信度高"的样本，避免典型图落在数据本身对不上的片段上
    hi = [r for r in recs if r.get("word_match_ratio", 0) >= 0.8]
    pool = hi if hi else recs
    pool = sorted(pool, key=lambda r: (-(r["timebin_audio_valid"] + r["timebin_vision_valid"]),
                                       abs(r["n_words"] - 15)))
    return pool[0]


def main():
    rec = pick_sample()
    sid, T = rec["id"], float(rec["duration"])
    words = rec["words"]

    a = np.load(os.path.join(C.FEAT_AUDIO, sid + ".npy"))      # (T_a,74)
    v = np.load(os.path.join(C.FEAT_VISION, sid + ".npy"))     # (T_v,35)
    with open(C.PKL_TIMEBIN, "rb") as f:
        tb = pickle.load(f)["test"]
    i = tb["id"].index(sid)

    bin_edges = np.arange(C.N_BINS + 1) * T / C.N_BINS

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # 1 文本词时间戳
    ax = axes[0]
    for k, w in enumerate(words):
        ax.barh(0, w["end"] - w["start"], left=w["start"], height=0.5,
                color=plt.cm.tab20(k % 20), edgecolor="k", linewidth=0.4)
        if k < 20:
            ax.text((w["start"] + w["end"]) / 2, 0.35, str(k + 1),
                    ha="center", fontsize=7)
    ax.set_yticks([]); ax.set_xlim(0, T)
    ax.set_title(f"典型样本 {sid}  (时长 {T:.2f}s, {len(words)} 词, 对齐方式={rec['text_align']})")
    ax.set_ylabel("文本")

    # 2 语音原始帧特征
    ax = axes[1]
    im = ax.imshow(a.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, T, 0, a.shape[1]])
    ax.set_ylabel("语音74维")
    for e in bin_edges[::5]:
        ax.axvline(e, color="w", alpha=0.15, linewidth=0.5)
    plt.colorbar(im, ax=ax, fraction=0.02)

    # 3 视觉原始帧特征
    ax = axes[2]
    im = ax.imshow(v.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, T, 0, v.shape[1]])
    ax.set_ylabel("视觉35维")
    for e in bin_edges[::5]:
        ax.axvline(e, color="w", alpha=0.15, linewidth=0.5)
    plt.colorbar(im, ax=ax, fraction=0.02)

    # 4 对齐后的时间箱特征
    ax = axes[3]
    im = ax.imshow(tb["audio"][i].T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, C.N_BINS, 0, C.AUDIO_DIM])
    ax.set_ylabel("对齐后 50x74")
    ax.set_xlabel("时间箱索引 k (1..50)")
    plt.colorbar(im, ax=ax, fraction=0.02)

    plt.tight_layout()
    out = os.path.join(C.OUT, "typical_sample.png")
    plt.savefig(out, dpi=130)
    print("已保存:", out)


if __name__ == "__main__":
    main()
