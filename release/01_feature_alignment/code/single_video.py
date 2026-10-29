# -*- coding: utf-8 -*-
"""单视频全分辨率处理 + 结果图（不做"缩小后再检测"优化）

用法: python single_video.py [-3g5yACwYnA$2]
输出: q1_output/single/<id>/ 下
  frames/xxx.jpg       全分辨率(1280x720)抽帧
  face_boxes.json      每帧人脸框(全分辨率检测)
  feat_vision.npy      视觉35维(全分辨率检测)
  result.png           结果图
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import config as C
import pipeline as P

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def extract_frames_fullres(sid):
    import cv2
    outdir = os.path.join(C.OUT, "single", sid, "frames")
    os.makedirs(outdir, exist_ok=True)
    if os.listdir(outdir):
        return outdir
    df = P.read_samples()
    video = P.video_path(df[df["id"] == sid].iloc[0])
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or C.VISION_FPS
    interval = max(1, int(round(fps / C.VISION_FPS)))
    n = t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % interval == 0:
            # 全分辨率，不做任何缩放
            _, buf = cv2.imencode(".jpg", frame,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            buf.tofile(os.path.join(outdir, f"{t:06d}.jpg"))
            t += 1
        n += 1
    cap.release()
    return outdir


def vision_features_fullres(sid, frames_dir):
    import cv2
    out = os.path.join(C.OUT, "single", sid, "feat_vision.npy")
    boxes_path = os.path.join(C.OUT, "single", sid, "face_boxes.json")
    det = cv2.CascadeClassifier(os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg"))
    feats, boxes = [], []
    t0 = time.time()
    for idx, fn in enumerate(files):
        img = cv2.imdecode(np.fromfile(os.path.join(frames_dir, fn),
                                       dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            feats.append(np.zeros(C.VISION_DIM, np.float32)); boxes.append(None); continue
        # 关键：直接在全分辨率(1280x720)灰度图上检测，不缩小
        faces = det.detectMultiScale(img, 1.2, 5, minSize=(24, 24))
        if len(faces) == 0:
            feats.append(np.zeros(C.VISION_DIM, np.float32)); boxes.append(None); continue
        x, y, fw, fh = faces[0]
        face = cv2.resize(img[y:y + fh, x:x + fw], (7, 5)).astype(np.float32)
        feats.append((face.reshape(-1) / 255.0).astype(np.float32))
        boxes.append([int(x), int(y), int(fw), int(fh)])
    arr = np.stack(feats).astype(np.float32) if feats else np.zeros((1, C.VISION_DIM), np.float32)
    np.save(out, arr)
    with open(boxes_path, "w", encoding="utf-8") as f:
        json.dump({"id": sid, "frame_count": len(files),
                   "detect_time_s": round(time.time() - t0, 2),
                   "faces": boxes}, f, ensure_ascii=False)
    return arr, boxes


def load_audio_wave(sid):
    import soundfile as sf
    wav = os.path.join(C.AUDIO_DIR, sid + ".wav")
    y, sr = sf.read(wav, dtype="float32")
    return y, sr


def make_figure(sid):
    import cv2
    row = P.read_samples()[P.read_samples()["id"] == sid].iloc[0]
    text = str(row["text"])
    meta = P._read_meta(sid)
    T = float(meta["duration"])
    wav = os.path.join(C.AUDIO_DIR, sid + ".wav")

    # 复用已有的音频/文本特征与词时间戳
    a = np.load(os.path.join(C.FEAT_AUDIO, sid + ".npy"))       # (Ta,74)
    temb, words = P.text_and_align(row, wav)                    # (n_words,768), words[]
    v = np.load(os.path.join(C.OUT, "single", sid, "feat_vision.npy"))  # (Tv,35)
    frames_dir = os.path.join(C.OUT, "single", sid, "frames")
    frames_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    boxes = json.load(open(os.path.join(C.OUT, "single", sid, "face_boxes.json"),
                           encoding="utf-8"))["faces"]

    t_a = (np.arange(a.shape[0]) + 0.5) / C.AUDIO_FPS
    t_v = (np.arange(v.shape[0]) + 0.5) / C.VISION_FPS
    t_t = np.array([(w["start"] + w["end"]) / 2.0 for w in words]) if words else np.zeros(0)
    at, am = P.align_timebin(a, t_a, T, C.AUDIO_DIM)
    vt, vm = P.align_timebin(v, t_v, T, C.VISION_DIM)
    tt, tm = P.align_timebin(temb, t_t, T, C.TEXT_DIM)

    y, sr = load_audio_wave(sid)
    t_wave = np.arange(len(y)) / sr

    # ---- 采样 8 帧（带人脸框，全分辨率） ----
    n_frames = len(frames_files)
    picks = np.linspace(0, n_frames - 1, 8).astype(int)
    frame_imgs = []
    for pi in picks:
        img = cv2.imdecode(np.fromfile(os.path.join(frames_dir, frames_files[pi]),
                                       dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            if boxes[pi]:
                x, yy, w, h = boxes[pi]
                cv2.rectangle(img, (x, yy), (x + w, yy + h), (0, 220, 0), 3)
            frame_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            frame_imgs.append(np.full((720, 1280, 3), 255, np.uint8))

    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(6, 1, height_ratios=[2.2, 0.8, 1.0, 1.6, 1.4, 2.4],
                           hspace=0.55)
    fig.suptitle(f"单视频多模态特征与时序对齐  |  样本 {sid}  |  时长 {T:.2f}s  |  "
                 f"标签 {row['annotation']} ({row['label']:.1f})  |  "
                 f"词对齐 {meta['text_align']}  r={meta['word_match_ratio']:.2f}",
                 fontsize=14, fontweight="bold", y=0.985)

    # A 视频帧（全分辨率人脸检测），用一行 OffsetImage 排布
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    axA = fig.add_subplot(gs[0])
    axA.set_xlim(0, 8); axA.set_ylim(-0.16, 1.0)
    axA.set_xticks([]); axA.set_yticks([])
    for i, (img, pi) in enumerate(zip(frame_imgs, picks)):
        im = OffsetImage(img, zoom=0.55)
        ab = AnnotationBbox(im, (i + 0.5, 0.5), frameon=True,
                            bboxprops=dict(edgecolor="gray", lw=1))
        axA.add_artist(ab)
        axA.text(i + 0.5, -0.06, f"t={pi/30:.1f}s", ha="center", va="top", fontsize=8)
    axA.set_title("① 视觉：全分辨率(1280×720)抽帧 + 人脸检测（绿框，未做缩小优化）", fontsize=11)
    axA.axis("off")

    # B 文本词时间轴
    axB = fig.add_subplot(gs[1])
    for k, w in enumerate(words):
        axB.barh(0, w["end"] - w["start"], left=w["start"], height=0.55,
                 color=plt.cm.tab20(k % 20), edgecolor="k", linewidth=0.4)
        axB.text((w["start"] + w["end"]) / 2, 0.32, w["token"],
                 ha="center", va="bottom", fontsize=8, rotation=0)
    axB.set_xlim(0, T); axB.set_yticks([])
    axB.set_title(f"② 文本：{len(words)} 个词及其时间戳（Whisper 对齐，语种=en）", fontsize=11)
    axB.set_xlabel("时间 / 秒")

    # C 音频波形
    axC = fig.add_subplot(gs[2])
    axC.plot(t_wave, y, lw=0.4, color="#2b6cb0")
    axC.set_xlim(0, T); axC.set_xlabel("时间 / 秒")
    axC.set_ylabel("幅度")
    axC.set_title("③ 音频波形（16 kHz 单声道）", fontsize=11)

    # D 音频 74 维特征
    axD = fig.add_subplot(gs[3])
    im = axD.imshow(a.T, aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, T, 0, a.shape[1]])
    axD.set_ylabel("语音特征维度 (74)")
    axD.set_title("④ 语音特征：24 MFCC + 24 Δ + 24 ΔΔ + 1 F0 + 1 RMS = 74 维", fontsize=11)
    plt.colorbar(im, ax=axD, fraction=0.02)

    # E 视觉 35 维特征
    axE = fig.add_subplot(gs[4])
    im = axE.imshow(v.T, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, T, 0, v.shape[1]])
    axE.set_xlabel("时间 / 秒")
    axE.set_ylabel("视觉特征维度 (35)")
    axE.set_title("⑤ 视觉特征：人脸 5×7 灰度网格 = 35 维（全分辨率检测）", fontsize=11)
    plt.colorbar(im, ax=axE, fraction=0.02)

    # F 对齐后的 50 时间箱（三模态并排）
    axF = fig.add_subplot(gs[5])
    n = C.N_BINS
    comb = np.hstack([tt / (np.abs(tt).max() + 1e-9),
                      at / (np.abs(at).max() + 1e-9),
                      vt / (np.abs(vt).max() + 1e-9)])
    axF.imshow(comb.T, aspect="auto", origin="lower", cmap="viridis")
    axF.axvline(n - 0.5, color="white", lw=2)
    axF.axvline(2 * n - 0.5, color="white", lw=2)
    axF.set_xticks([n / 2 - 0.5, 1.5 * n - 0.5, 2.5 * n - 0.5])
    axF.set_xticklabels(["文本 768维", "语音 74维", "视觉 35维"], fontsize=10)
    axF.set_ylabel("特征维度（各自归一化）")
    axF.set_title(f"⑥ 对齐到 50 个等长时间箱 Iₖ=[(k−1)T/50, kT/50)  （含 mask）", fontsize=11)

    outdir = os.path.join(C.OUT, "single", sid)
    out = os.path.join(outdir, "result.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print("已保存:", out)
    print("帧数=%d 视觉帧=%d 语音帧=%d 词数=%d" % (n_frames, v.shape[0], a.shape[0], len(words)))
    print("检测耗时(秒):", json.load(open(os.path.join(outdir,"face_boxes.json"),encoding="utf-8"))["detect_time_s"])


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "-3g5yACwYnA$2"
    os.makedirs(os.path.join(C.OUT, "single", sid), exist_ok=True)
    t0 = time.time()
    fd = extract_frames_fullres(sid)
    v, boxes = vision_features_fullres(sid, fd)
    make_figure(sid)
    print("总耗时 %.1f 秒" % (time.time() - t0))
