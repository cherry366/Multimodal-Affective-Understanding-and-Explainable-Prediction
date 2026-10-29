# -*- coding: utf-8 -*-
"""单视频：每帧 ↔ 对齐音频波形 ↔ 对齐单词 的对应关系图（全分辨率 + 所有人脸框）

用法: python visualize_aligned.py <id>
输出: q1_output/single/<id>/aligned_frames.png
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config as C
import pipeline as P

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def detect_all_faces(gray):
    import cv2
    det = cv2.CascadeClassifier(os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    # 全分辨率、抓所有脸
    return det.detectMultiScale(gray, 1.08, 5, minSize=(30, 30))


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "-3g5yACwYnA$2"
    import cv2, soundfile as sf

    base = os.path.join(C.OUT, "single", sid)
    frames_dir = os.path.join(base, "frames")
    frames_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    n_frames = len(frames_files)

    # 对齐后的词时间戳
    align = json.load(open(os.path.join(C.ALIGN_DIR, sid + ".json"), encoding="utf-8"))
    words = align["words"]

    # 音频
    y, sr = sf.read(os.path.join(C.AUDIO_DIR, sid + ".wav"), dtype="float32")

    N = len(words)
    # 为每个词选"词中点"对应的帧，并把全音频波形切片画出来
    print("词数", N, "帧数", n_frames)

    fig = plt.figure(figsize=(2.6 * N, 7.2))
    gs = gridspec.GridSpec(3, N, height_ratios=[2.6, 1.4, 0.8],
                           hspace=0.22, wspace=0.06,
                           left=0.02, right=0.98, top=0.93, bottom=0.06)
    fig.suptitle(f"单视频：每帧 ↔ 对齐音频波形 ↔ 对齐单词（全分辨率人脸框）\n样本 {sid}",
                 fontsize=15, fontweight="bold")

    for j, w in enumerate(words):
        t_mid = (w["start"] + w["end"]) / 2.0
        fi = int(round(t_mid * C.VISION_FPS))
        fi = max(0, min(n_frames - 1, fi))

        # 帧图（全分辨率 + 所有人脸框）
        img = cv2.imdecode(np.fromfile(os.path.join(frames_dir, frames_files[fi]),
                                       dtype=np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_all_faces(gray)
        for nf, (x, yy, ww, hh) in enumerate(faces):
            cv2.rectangle(img, (x, yy), (x + ww, yy + hh), (0, 200, 0), 3)
            cv2.putText(img, f"face{nf+1}", (x, max(0, yy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax = fig.add_subplot(gs[0, j])
        ax.imshow(rgb); ax.axis("off")
        ax.set_title(f"t={t_mid:.2f}s  [{fi}]", fontsize=9)

        # 该词对应的音频波形切片（词区间高亮 + 帧时刻红竖线）
        pad = 0.25
        t0 = max(0.0, w["start"] - pad)
        t1 = min(len(y) / sr, w["end"] + pad)
        i0, i1 = int(t0 * sr), int(t1 * sr)
        tt = np.arange(i0, i1) / sr
        ax = fig.add_subplot(gs[1, j])
        ax.plot(tt, y[i0:i1], lw=0.5, color="#1f5fbf")
        ax.axvspan(w["start"], w["end"], color="#ffd54f", alpha=0.55)
        ax.axvline(t_mid, color="red", lw=1.2)
        ax.set_xlim(t0, t1)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("音频波形", fontsize=9)
        ax.set_xlabel(f"[{w['start']:.2f}, {w['end']:.2f}]s", fontsize=8)

        # 对齐的单词
        ax = fig.add_subplot(gs[2, j])
        ax.axis("off")
        ax.text(0.5, 0.5, f"'{w['token']}'", ha="center", va="center",
                fontsize=11, fontweight="bold", color="#8b0000",
                transform=ax.transAxes)
        ax.set_title(f"词{j+1}", fontsize=8)

    out = os.path.join(base, "aligned_frames.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print("已保存:", out)


if __name__ == "__main__":
    main()
