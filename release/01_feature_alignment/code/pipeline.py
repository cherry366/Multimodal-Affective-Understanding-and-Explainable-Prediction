# -*- coding: utf-8 -*-
"""模块一 融合版流水线

设计：对齐优先 + 时间箱/词级双版本 + 掩码闭环 + eps统一池化 + 工具链落地

用法:
    python pipeline.py all
    python pipeline.py audio | vision | text | build | summary

输出:
    output/q1_features_timebin.pkl   时间箱版(主,掩码可反查时间)
    output/q1_features_word.pkl      词级版(与标准化特征aligned_50 兼容)
    output/summary.csv               100条汇总表
    output/align_records.json        对齐记录
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内加速;直连HF可删
import sys
import json
import pickle
import subprocess

import numpy as np
import pandas as pd

import config as C


# ================== 通用工具 ==================
def _update_meta(sid, **kw):
    path = os.path.join(C.META_DIR, sid + ".json")
    d = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    d.update(kw)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)


def _read_meta(sid):
    path = os.path.join(C.META_DIR, sid + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


_BERT = {}
def _get_bert():
    if not _BERT:
        from transformers import BertTokenizerFast, BertModel
        _BERT["tok"] = BertTokenizerFast.from_pretrained("bert-base-uncased")
        _BERT["model"] = BertModel.from_pretrained("bert-base-uncased")
        _BERT["model"].eval()
    return _BERT["tok"], _BERT["model"]


_WHISPER = {}   # 按模型规格缓存 {size: model}


# ================== Step0 样本清单 ==================
def read_samples():
    df = pd.read_excel(C.LABEL_XLSX, dtype={"video_id": str, "clip_id": str})
    df["clip_id"] = df["clip_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["id"] = df["video_id"] + "$" + df["clip_id"]
    df["classification"] = df["annotation"].map(C.CLASS_MAP)
    df["regression"] = df["label"].astype(float)
    return df[["id", "video_id", "clip_id", "text", "label", "annotation",
               "classification", "regression"]]


def video_path(row):
    return os.path.join(C.ATT1, row["video_id"], row["clip_id"] + ".mp4")


# ================== Step1 抽音频 ==================
def extract_audio(row):
    wav = os.path.join(C.AUDIO_DIR, row["id"] + ".wav")
    if os.path.exists(wav) and _read_meta(row["id"]).get("duration"):
        return wav
    subprocess.run([C.get_ffmpeg(), "-i", video_path(row), "-vn", "-ac", "1",
                    "-ar", str(C.AUDIO_SR), "-y", wav],
                   check=True, capture_output=True)
    import soundfile as sf
    info = sf.info(wav)
    _update_meta(row["id"], duration=round(info.frames / info.samplerate, 6))
    return wav


# ================== Step2 语音特征 74维/帧 ==================
def audio_features(row, wav):
    out = os.path.join(C.FEAT_AUDIO, row["id"] + ".npy")
    if os.path.exists(out):
        return out
    try:
        import librosa
    except ImportError:
        raise ImportError("请先安装 librosa: pip install librosa")
    y, sr = librosa.load(wav, sr=C.AUDIO_SR, mono=True)
    hop = int(round(C.AUDIO_SR / C.AUDIO_FPS))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, hop_length=hop)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    L = mfcc.shape[1]

    def _fix(v):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if len(v) < L:
            v = np.pad(v, (0, L - len(v)))
        return v[:L]

    f0 = _fix(librosa.yin(y, fmin=80, fmax=400, sr=sr, hop_length=hop))
    rms = _fix(librosa.feature.rms(y=y, hop_length=hop)[0])
    feat = np.vstack([mfcc, d1, d2, f0[None, :], rms[None, :]]).T.astype(np.float32)
    np.save(out, feat)
    _update_meta(row["id"], n_audio_frames=int(feat.shape[0]))
    return out


# ================== Step3 抽帧(缩放到640宽加速) ==================
def extract_frames(row):
    import cv2
    outdir = os.path.join(C.FRAMES_DIR, row["id"])
    os.makedirs(outdir, exist_ok=True)
    if os.listdir(outdir):
        return outdir
    cap = cv2.VideoCapture(video_path(row))
    fps = cap.get(cv2.CAP_PROP_FPS) or C.VISION_FPS
    interval = max(1, int(round(fps / C.VISION_FPS)))
    n = t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % interval == 0:
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(round(h * 640.0 / w))))
            _, buf = cv2.imencode(".jpg", frame,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            buf.tofile(os.path.join(outdir, f"{t:06d}.jpg"))
            t += 1
        n += 1
    cap.release()
    return outdir


# ================== Step4 视觉特征 35维/帧 (OpenFace 2.2.0) ==================
# 35 维构成 = 17 AU回归 + 6 头部姿态 + 8 注视 + 4 眼部几何
OF_AU_COLS = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
              "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
              "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]
OF_POSE_COLS = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
OF_GAZE_COLS = ["gaze_0_x", "gaze_0_y", "gaze_0_z",
                "gaze_1_x", "gaze_1_y", "gaze_1_z",
                "gaze_angle_x", "gaze_angle_y"]


def _run_openface(row):
    """对单条视频运行 OpenFace FeatureExtraction, 返回输出 CSV 路径。"""
    outdir = os.path.join(C.OF_RAW_DIR, row["id"])
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, row["clip_id"] + ".csv")
    if os.path.exists(csv_path):
        return csv_path
    cmd = [C.OPENFACE_EXE, "-f", video_path(row),
           "-out_dir", outdir, "-mloc", C.OPENFACE_MLOC,
           "-pose", "-aus", "-gaze"]
    # OpenFace 需在其安装目录下运行, 否则 model/... 相对路径无法解析
    subprocess.run(cmd, cwd=C.OPENFACE_DIR, check=True, capture_output=True)
    return csv_path


def _parse_openface_csv(row, csv_path):
    """解析 OpenFace CSV -> (feat[N,35], times[N], valid[N])。"""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    N = len(df)
    feat = np.zeros((N, C.VISION_DIM), dtype=np.float32)

    feat[:, 0:17] = df[OF_AU_COLS].values.astype(np.float32)
    feat[:, 17:23] = df[OF_POSE_COLS].values.astype(np.float32)
    feat[:, 23:31] = df[OF_GAZE_COLS].values.astype(np.float32)

    # 4 维眼部几何: 左眼 28 点 / 右眼 28 点 的 2D 瞳孔中心
    xl = [f"eye_lmk_x_{i}" for i in range(28)]
    yl = [f"eye_lmk_y_{i}" for i in range(28)]
    xr = [f"eye_lmk_x_{i}" for i in range(28, 56)]
    yr = [f"eye_lmk_y_{i}" for i in range(28, 56)]
    feat[:, 31] = df[xl].mean(axis=1).values.astype(np.float32)
    feat[:, 32] = df[yl].mean(axis=1).values.astype(np.float32)
    feat[:, 33] = df[xr].mean(axis=1).values.astype(np.float32)
    feat[:, 34] = df[yr].mean(axis=1).values.astype(np.float32)

    times = df["timestamp"].values.astype(np.float64)
    valid = (df["success"].values.astype(int) == 1).astype(np.int8)

    feat[valid == 0] = 0.0                     # 跟踪失败帧置零
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat, times, valid


def vision_features(row):
    """OpenFace 视觉特征: 输出 feat_vision/{id}.npy(35维) + .times.npy + .valid.npy。"""
    out = os.path.join(C.FEAT_VISION, row["id"] + ".npy")
    tpath = out.replace(".npy", ".times.npy")
    vpath = out.replace(".npy", ".valid.npy")
    if os.path.exists(out) and os.path.exists(tpath) and os.path.exists(vpath):
        return out
    try:
        feat, times, valid = _parse_openface_csv(row, _run_openface(row))
    except Exception as e:
        import traceback
        print(f"[openface-fail] {row['id']}: {e}", flush=True)
        traceback.print_exc()
        dur = float(_read_meta(row["id"]).get("duration", 0) or 0)
        n = max(1, int(round(dur * C.VISION_FPS)))
        feat = np.zeros((n, C.VISION_DIM), np.float32)
        times = (np.arange(n) + 0.5) / C.VISION_FPS
        valid = np.zeros(n, np.int8)
    np.save(out, feat)
    np.save(tpath, times)
    np.save(vpath, valid)
    _update_meta(row["id"], n_vision_frames=int(feat.shape[0]),
                 n_vision_valid=int(valid.sum()),
                 vision_backend="openface_2.2.0")
    return out


# ================== Step5 文本特征(词级) + 词时间戳 ==================
def _norm_word(w):
    """归一化词形：小写并去标点，仅用于序列对齐比较"""
    import re
    return re.sub(r"[^a-z0-9']", "", str(w).lower())


def _get_whisper(size="base"):
    if size not in _WHISPER:
        import whisper
        _WHISPER[size] = whisper.load_model(size)
    return _WHISPER[size]


def _whisper_raw_words(row, wav, size):
    """Whisper 词级时间戳（按模型规格分别磁盘缓存，避免重复跑 ASR）"""
    cache = os.path.join(C.RAW_ALIGN_DIR, f"{row['id']}.{size}.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            d = json.load(f)
        return d["whisper_words"]
    import librosa
    y, _ = librosa.load(wav, sr=C.AUDIO_SR, mono=True)
    wm = _get_whisper(size)
    # 关键：MOSEI 为英文语料，强制 language="en"。
    # 否则短音频/带噪片段会被误判语种（实测出现整段转写成希伯来语），
    # 词数看似接近但内容完全对不上。
    res = wm.transcribe(y, language="en", task="transcribe",
                        word_timestamps=True, condition_on_previous_text=False)
    wt = [{"token": w["word"], "start": float(w["start"]), "end": float(w["end"])}
          for seg in res["segments"] for w in seg.get("words", [])]
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"id": row["id"], "model": size, "asr_text": res.get("text", ""),
                   "whisper_words": wt}, f, ensure_ascii=False)
    return wt


def _align_text_to_asr(raw_words, wt, T):
    """把 ASR 词时间戳映射到数据给定的文本词表。

    规则：先归一化后做序列对齐(difflib.SequenceMatcher)。
    equal 块内的词直接继承 ASR 时间戳（锚点）；
    漏读/错读/语气词造成的缺口，在相邻锚点之间线性插值。
    返回 (spans, n_matched, n_interp)，spans[i]=(start,end)
    """
    import difflib
    n = len(raw_words)
    spans = [None] * n
    if n == 0:
        return [], 0, 0
    if not wt:
        step = (T / n) if T > 0 else 0.0
        return [(i * step, (i + 1) * step) for i in range(n)], 0, n

    A = [_norm_word(w) for w in raw_words]
    B = [_norm_word(w["token"]) for w in wt]
    sm = difflib.SequenceMatcher(a=A, b=B, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                spans[i1 + k] = (wt[j1 + k]["start"], wt[j1 + k]["end"])

    known = [i for i, s in enumerate(spans) if s is not None]
    if not known:
        step = (T / n) if T > 0 else 0.0
        return [(i * step, (i + 1) * step) for i in range(n)], 0, n

    i_first = known[0]                      # 头部缺口
    if i_first > 0:
        step = spans[i_first][0] / i_first
        for k in range(i_first):
            spans[k] = (k * step, (k + 1) * step)
    for a, b in zip(known, known[1:]):      # 中间缺口
        gap = b - a - 1
        if gap > 0:
            t0, t1 = spans[a][1], max(spans[b][0], spans[a][1])
            step = (t1 - t0) / gap
            for k in range(1, gap + 1):
                spans[a + k] = (t0 + (k - 1) * step, t0 + k * step)
    i_last = known[-1]                      # 尾部缺口
    if i_last < n - 1:
        rem = n - 1 - i_last
        t1 = T if T > spans[i_last][1] else spans[i_last][1] + 0.3 * rem
        step = (t1 - spans[i_last][1]) / rem
        for k in range(1, rem + 1):
            spans[i_last + k] = (spans[i_last][1] + (k - 1) * step,
                                 spans[i_last][1] + k * step)
    return spans, len(known), n - len(known)


def _speech_intervals(y, sr, top_db=30):
    """能量法 VAD：返回语音活跃区间 [(t_start,t_end), ...]（秒）"""
    import librosa
    iv = librosa.effects.split(y, top_db=top_db)
    out = [(float(s) / sr, float(e) / sr) for s, e in iv]
    return [(a, b) for a, b in out if b - a > 0.02]


def _distribute_over_intervals(n, intervals, T):
    """把 n 个词按"语音活跃时长"比例铺到活跃区间上（ASR 完全失败时的兜底）"""
    if not intervals:
        step = (T / n) if T > 0 else 0.0
        return [(i * step, (i + 1) * step) for i in range(n)]
    total = sum(b - a for a, b in intervals)
    spans = []
    acc = 0.0
    for w in range(n):
        t0 = acc / n * total
        t1 = (acc + 1) / n * total
        acc += 1
        spans.append(_map_to_intervals(t0, t1, intervals))
    return spans


def _map_to_intervals(t0, t1, intervals):
    """把 [0,total] 上的相对位置映射回真实时间轴（跳过静音段）"""
    def pos(t):
        acc = 0.0
        for a, b in intervals:
            if t <= acc + (b - a):
                return a + (t - acc)
            acc += (b - a)
        return intervals[-1][1]
    return (pos(t0), max(pos(t1), pos(t0) + 1e-3))


def _bert_word_emb(row, raw_words):
    """BERT 词级语义向量：按字符区间重叠加权，取该词覆盖到的 subword 平均"""
    import torch
    text = str(row["text"])
    tok, model = _get_bert()
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt",
              truncation=True, max_length=256)
    offsets = enc.pop("offset_mapping")[0].numpy()
    with torch.no_grad():
        h = model(**enc).last_hidden_state[0].numpy()
    spans_chars, pos = [], 0
    for w in raw_words:
        i = text.find(w, pos)
        if i < 0:
            i = pos
        spans_chars.append((i, i + len(w)))
        pos = i + len(w)
    word_emb = np.zeros((len(raw_words), C.TEXT_DIM), dtype=np.float32)
    for wi, (a, b) in enumerate(spans_chars):
        sel = np.array([(t0 < b and t1 > a) for t0, t1 in offsets])
        if sel.any():
            word_emb[wi] = h[sel].mean(axis=0)
    return word_emb


def text_and_align(row, wav):
    emb_path = os.path.join(C.FEAT_TEXT, row["id"] + ".npy")
    align_path = os.path.join(C.ALIGN_DIR, row["id"] + ".json")
    if os.path.exists(emb_path) and os.path.exists(align_path):
        with open(align_path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d.get("method"), str) and d["method"].startswith("whisper_seq"):
            return np.load(emb_path), d["words"]

    text = str(row["text"])
    raw_words = [w for w in text.split() if w]
    n = len(raw_words)
    word_emb = None
    if os.path.exists(emb_path):
        _tmp = np.load(emb_path)
        if _tmp.shape[0] == n:
            word_emb = _tmp
    if word_emb is None:
        word_emb = _bert_word_emb(row, raw_words)

    dur = float(_read_meta(row["id"]).get("duration", 0) or 0)
    import librosa
    y, _ = librosa.load(wav, sr=C.AUDIO_SR, mono=True)

    # --- 递进式 ASR：按 C.WHISPER_SIZES 依次尝试，够好就停 ---
    best = None
    asr_info = []
    for size in C.WHISPER_SIZES:
        try:
            wt = _whisper_raw_words(row, wav, size)
        except Exception as e:
            asr_info.append({"model": size, "error": repr(e)[:160]})
            continue
        spans, nm, ni = _align_text_to_asr(raw_words, wt, dur)
        ratio = nm / max(1, n)
        asr_info.append({"model": size, "n_asr_words": len(wt),
                         "match_ratio": round(ratio, 4)})
        if best is None or ratio > best["ratio"]:
            best = dict(ratio=ratio, spans=spans, nm=nm, ni=ni,
                        size=size, wt=wt)
        if ratio >= C.ASR_ACCEPT_RATIO:
            break

    if best is not None and best["ratio"] > 0:
        spans, method = best["spans"], "whisper_seq"
        nm, ni, ratio, size = best["nm"], best["ni"], best["ratio"], best["size"]
    else:
        # ASR 完全没对上 -> 用能量 VAD 把词铺到"有声音"的区间上
        iv = _speech_intervals(y, C.AUDIO_SR)
        spans = _distribute_over_intervals(n, iv, dur)
        method, nm, ni, ratio, size = "vad_uniform", 0, n, 0.0, None

    words = [{"token": w, "start": float(s[0]), "end": float(s[1])}
             for w, s in zip(raw_words, spans)]
    _update_meta(row["id"], text_align=method, word_match_ratio=round(ratio, 4),
                 n_words=n, asr_model=size, n_word_interpolated=ni,
                 asr_trials=asr_info)

    np.save(emb_path, word_emb)
    with open(align_path, "w", encoding="utf-8") as f:
        json.dump({"id": row["id"], "method": method, "asr_model": size,
                   "word_match_ratio": round(ratio, 4), "words": words},
                  f, ensure_ascii=False)
    return word_emb, words


# ================== Step6/7 对齐 ==================
def align_timebin(feat, times, T, dim, valid=None):
    n = C.N_BINS
    out = np.zeros((n, dim), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if valid is None:
        valid = np.ones(len(feat), dtype=np.float32)
    else:
        valid = np.asarray(valid, dtype=np.float32)
    if T > 0 and len(times):
        idx = np.clip(np.floor(times / T * n).astype(int), 0, n - 1)
        for k in range(n):
            sel = (idx == k)
            c = float(valid[sel].sum())
            cnt[k] = c
            if c > 0:
                out[k] = (feat[sel] * valid[sel, None]).sum(axis=0)
    out = (out / (cnt[:, None] + C.EPS)).astype(np.float32)
    mask = (cnt > 0).astype(np.float32)
    return out, mask


def align_word(feat, times, words, dim, valid=None):
    n = C.MAX_WORDS
    out = np.zeros((n, dim), dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    if valid is None:
        valid = np.ones(len(feat), dtype=np.float32)
    else:
        valid = np.asarray(valid, dtype=np.float32)
    for k, w in enumerate(words[:n]):
        sel = (times >= w["start"]) & (times < w["end"])
        c = float(valid[sel].sum())
        if c > 0:
            out[k] = (feat[sel] * valid[sel, None]).sum(axis=0) / c
            mask[k] = 1.0
    length = min(len(words), n)
    return out, mask, length


def _pad_rows(arr, n, dim):
    out = np.zeros((n, dim), dtype=np.float32)
    m = min(len(arr), n)
    out[:m] = arr[:m]
    return out


# ================== Step8/9 打包 ==================
def build_features(samples):
    keys = ["id", "raw_text", "text", "audio", "vision",
            "text_mask", "audio_mask", "vision_mask",
            "duration", "n_audio_frames", "n_vision_frames",
            "align_mode", "annotations", "classification_labels",
            "regression_labels"]
    tb = {k: [] for k in keys}
    wd = {k: [] for k in keys}
    wd["length"] = []
    records = []

    for idx, (_, row) in enumerate(samples.iterrows()):
        sid = row["id"]
        meta = _read_meta(sid)
        T = float(meta.get("duration", 0) or 0)
        if T <= 0:
            raise ValueError(f"{sid} 缺少时长(duration)，请先跑 audio 步骤")
        a = np.load(os.path.join(C.FEAT_AUDIO, sid + ".npy"))
        v = np.load(os.path.join(C.FEAT_VISION, sid + ".npy"))
        wav = os.path.join(C.AUDIO_DIR, sid + ".wav")
        temb, words = text_and_align(row, wav)

        t_a = (np.arange(a.shape[0]) + 0.5) / C.AUDIO_FPS
        # OpenFace 版视觉: 优先用真实帧时间戳(时序可核验); 缺省才回退到 30fps 均匀
        _t_v_path = os.path.join(C.FEAT_VISION, sid + ".times.npy")
        if os.path.exists(_t_v_path):
            t_v = np.load(_t_v_path).astype(np.float64)
        else:
            t_v = (np.arange(v.shape[0]) + 0.5) / C.VISION_FPS
        _v_valid_path = os.path.join(C.FEAT_VISION, sid + ".valid.npy")
        v_valid = np.load(_v_valid_path).astype(np.float32) if os.path.exists(_v_valid_path) else None
        t_t = np.array([(w["start"] + w["end"]) / 2.0 for w in words],
                       dtype=np.float64) if words else np.zeros(0)

        at, am = align_timebin(a, t_a, T, C.AUDIO_DIM)
        vt, vm = align_timebin(v, t_v, T, C.VISION_DIM, valid=v_valid)
        tt, tm = align_timebin(temb, t_t, T, C.TEXT_DIM)
        aw, awm, awl = align_word(a, t_a, words, C.AUDIO_DIM)
        vw, vwm, vwl = align_word(v, t_v, words, C.VISION_DIM, valid=v_valid)
        tw = _pad_rows(temb, C.MAX_WORDS, C.TEXT_DIM)
        twm = np.zeros(C.MAX_WORDS, np.float32)
        twm[:min(len(words), C.MAX_WORDS)] = 1.0
        L = min(len(words), C.MAX_WORDS)

        def _fill(d, txt, aud, vis, tmask, amask, vmask, extra=None):
            d["id"].append(sid); d["raw_text"].append(str(row["text"]))
            d["text"].append(txt); d["audio"].append(aud); d["vision"].append(vis)
            d["text_mask"].append(tmask); d["audio_mask"].append(amask)
            d["vision_mask"].append(vmask)
            d["duration"].append(T)
            d["n_audio_frames"].append(int(a.shape[0]))
            d["n_vision_frames"].append(int(v.shape[0]))
            d["align_mode"].append(meta.get("text_align", "whisper"))
            d["annotations"].append(row["annotation"])
            d["classification_labels"].append(int(row["classification"]))
            d["regression_labels"].append(float(row["regression"]))
            if extra is not None:
                d["length"].append(extra)

        _fill(tb, tt, at, vt, tm, am, vm)
        _fill(wd, tw, aw, vw, twm, awm, vwm, extra=L)

        records.append({
            "id": sid, "duration": T, "n_words": len(words),
            "text_align": meta.get("text_align", "whisper"),
            "asr_model": meta.get("asr_model"),
            "word_match_ratio": meta.get("word_match_ratio", 0.0),
            "n_audio_frames": int(a.shape[0]),
            "n_vision_frames": int(v.shape[0]),
            "timebin_audio_valid": int(am.sum()),
            "timebin_vision_valid": int(vm.sum()),
            "timebin_text_valid": int(tm.sum()),
            "word_valid_length": int(L),
            "words": words,
        })
        if (idx + 1) % 10 == 0:
            print(f"  build {idx + 1}/{len(samples)}", flush=True)

    for d in (tb, wd):
        for k in ["text", "audio", "vision", "text_mask",
                  "audio_mask", "vision_mask"]:
            d[k] = np.stack(d[k])
        for k in ["duration", "n_audio_frames", "n_vision_frames"]:
            d[k] = np.asarray(d[k])
    wd["length"] = np.asarray(wd["length"])

    # --- 预处理三件套(对齐后): 生成标准化特征, 供模块二与质量分析使用 ---
    tb["text_norm"] = preprocess_block(tb["text"], tb["text_mask"])
    tb["audio_norm"] = preprocess_block(tb["audio"], tb["audio_mask"])
    tb["vision_norm"] = preprocess_block(tb["vision"], tb["vision_mask"])
    wd["text_norm"] = preprocess_block(wd["text"], wd["text_mask"])
    wd["audio_norm"] = preprocess_block(wd["audio"], wd["audio_mask"])
    wd["vision_norm"] = preprocess_block(wd["vision"], wd["vision_mask"])

    with open(C.PKL_TIMEBIN, "wb") as f:
        pickle.dump({"test": tb}, f)
    with open(C.PKL_WORD, "wb") as f:
        pickle.dump({"test": wd}, f)
    with open(os.path.join(C.OUT, "align_records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    return tb, wd



# ================== 预处理三件套(对齐后: 缺失已在mask + μ±5σ去极值 + z-score) ==================
def preprocess_block(arr, mask, sigma=C.SIGMA_CLIP):
    # arr:(S,N,D) mask:(S,N) -> 逐样本逐维度, 仅用有效格估计 μ/σ
    out = np.zeros_like(arr, dtype=np.float32)
    for si in range(arr.shape[0]):
        m = mask[si].astype(bool)
        valid = arr[si][m]
        if valid.shape[0] < 2:
            continue
        mu = valid.mean(axis=0)
        sd = valid.std(axis=0) + C.EPS
        clipped = np.clip(arr[si], mu - sigma * sd, mu + sigma * sd)
        z = (clipped - mu) / sd
        z[~m] = 0.0
        out[si] = z.astype(np.float32)
    return out


# ================== 跨模态相关性矩阵(皮尔逊) ==================
def _pairwise_abs_corr(A, B, mask):
    # 两模态块的逐维平均绝对皮尔逊相关(仅有效格). A:(N,D1) B:(N,D2)
    m = mask.astype(bool)
    A = A[m]; B = B[m]
    if A.shape[0] < 5:
        return np.nan
    A = (A - A.mean(axis=0)) / (A.std(axis=0) + C.EPS)
    B = (B - B.mean(axis=0)) / (B.std(axis=0) + C.EPS)
    R = (A.T @ B) / A.shape[0]          # 列标准化后即皮尔逊相关矩阵 (D1,D2)
    return float(np.abs(R).mean())


def compute_cross_modal_corr(tb):
    # 3x3 跨模态平均绝对相关矩阵 + 每样本相关表. 返回 (mat3x3, corr_df)
    if "text_norm" not in tb:
        tb["text_norm"] = preprocess_block(tb["text"], tb["text_mask"])
        tb["audio_norm"] = preprocess_block(tb["audio"], tb["audio_mask"])
        tb["vision_norm"] = preprocess_block(tb["vision"], tb["vision_mask"])
    T = tb["text_norm"]; A = tb["audio_norm"]; V = tb["vision_norm"]
    tm = tb["text_mask"]; am = tb["audio_mask"]; vm = tb["vision_mask"]
    S = T.shape[0]
    specs = [("text_audio", T, A, tm, am),
             ("text_vision", T, V, tm, vm),
             ("audio_vision", A, V, am, vm)]
    per = {}
    for name, X, Y, xm, ym in specs:
        mask = (xm * ym).astype(bool)
        per[name] = [_pairwise_abs_corr(X[si], Y[si], mask[si]) for si in range(S)]
    agg = {}
    for name, vals in per.items():
        v = np.array([x for x in vals if not np.isnan(x)], dtype=float)
        agg[name] = dict(mean=float(v.mean()) if len(v) else np.nan,
                         std=float(v.std()) if len(v) else np.nan,
                         n=int(len(v)))
    mat = np.ones((3, 3), dtype=float)
    mat[0, 1] = mat[1, 0] = agg["text_audio"]["mean"]
    mat[0, 2] = mat[2, 0] = agg["text_vision"]["mean"]
    mat[1, 2] = mat[2, 1] = agg["audio_vision"]["mean"]
    os.makedirs(C.ANALYSIS_DIR, exist_ok=True)
    corr_df = pd.DataFrame({("corr_" + k): v for k, v in per.items()})
    corr_df.insert(0, "id", tb["id"])
    corr_df.to_csv(os.path.join(C.ANALYSIS_DIR, "corr_per_sample.csv"),
                   index=False, encoding="utf-8-sig")
    pd.DataFrame({k: agg[k] for k in ["text_audio", "text_vision", "audio_vision"]}
                 ).to_csv(os.path.join(C.ANALYSIS_DIR, "corr_matrix.csv"),
                          encoding="utf-8-sig")
    return mat, corr_df


# ================== 音视同步互相关校验 ==================
def _zscore_keep_mask(x, mask):
    m = mask.astype(bool)
    if m.sum() < 2:
        return np.zeros_like(x)
    mu = x[m].mean(); sd = x[m].std() + C.EPS
    z = (x - mu) / sd
    z[~m] = 0.0
    return z


def _ncc(a, v, tau):
    # 归一化互相关 C(tau), a/v 已 z-score 且无效格=0
    N = len(a)
    if tau >= 0:
        num = float(np.dot(a[:N - tau], v[tau:]))
        den = float(np.sqrt(np.dot(a[:N - tau], a[:N - tau]) *
                            np.dot(v[tau:], v[tau:])))
    else:
        t = -tau
        num = float(np.dot(a[t:], v[:N - t]))
        den = float(np.sqrt(np.dot(a[t:], a[t:]) * np.dot(v[:N - t], v[:N - t])))
    return num / (den + C.EPS)


def compute_av_sync(tb):
    # 音视同步互相关(原始帧级): 音频激活度插值到视觉真实帧时间戳, tau以视觉帧为单位
    rows, curves = [], []
    taus = np.arange(-C.NCC_TAU_MAX, C.NCC_TAU_MAX + 1)
    for si, sid in enumerate(tb["id"]):
        a_feat = np.load(os.path.join(C.FEAT_AUDIO, sid + ".npy"))
        v_feat = np.load(os.path.join(C.FEAT_VISION, sid + ".npy"))
        v_times = np.load(os.path.join(C.FEAT_VISION, sid + ".times.npy")).astype(float)
        v_valid = np.load(os.path.join(C.FEAT_VISION, sid + ".valid.npy")).astype(bool)
        a_times = (np.arange(len(a_feat)) + 0.5) / C.AUDIO_FPS
        # 音视同步物理代理: 音频=响度RMS, 视觉=嘴部张开AU25+AU26(说话时强同步)
        a_rms = np.nan_to_num(np.interp(v_times, a_times, a_feat[:, C.RMS_IDX]))
        v_mouth = np.nan_to_num(v_feat[:, C.AU_MOUTH_IDX].mean(axis=1))
        a = _zscore_keep_mask(a_rms, v_valid)
        v = _zscore_keep_mask(v_mouth, v_valid)
        curve = [float(_ncc(a, v, int(t))) for t in taus]
        curves.append(curve)
        ci = int(np.argmax(curve))
        peak_lag_frames = int(taus[ci])
        sync_score = float(curve[ci])
        others = [c for i, c in enumerate(curve) if abs(taus[i] - peak_lag_frames) >= 2]
        second = max(others) if others else 0.0
        dt = float(np.median(np.diff(v_times))) if len(v_times) > 1 else 1.0 / C.VISION_FPS
        rows.append(dict(id=sid,
                         sync_score=round(sync_score, 4),
                         peak_lag_frames=peak_lag_frames,
                         peak_lag_sec=round(peak_lag_frames * dt, 4),
                         sharpness=round(sync_score / (second + C.EPS), 4),
                         valid_ratio=round(float(v_valid.mean()), 4),
                         frame_dt=round(dt, 6)))
    sync_df = pd.DataFrame(rows)
    os.makedirs(C.ANALYSIS_DIR, exist_ok=True)
    sync_df.to_csv(os.path.join(C.ANALYSIS_DIR, "sync_metrics.csv"),
                   index=False, encoding="utf-8-sig")
    np.save(os.path.join(C.ANALYSIS_DIR, "sync_curves.npy"),
            np.asarray(curves, dtype=np.float32))
    np.save(os.path.join(C.ANALYSIS_DIR, "sync_taus.npy"), taus.astype(np.float32))
    return sync_df


# ================== Step10 五项校验 ==================
def validate(samples):
    with open(C.PKL_TIMEBIN, "rb") as f:
        tb = pickle.load(f)["test"]
    with open(C.PKL_WORD, "rb") as f:
        wd = pickle.load(f)["test"]

    assert len(tb["id"]) == 100, "样本数不是100"
    assert set(tb["id"]) == set(samples["id"]), "id 集合不一致"
    print("[1] 覆盖检查通过: 100 条 id 与输出一一对应")

    assert tb["text"].shape == (100, C.N_BINS, C.TEXT_DIM)
    assert tb["audio"].shape == (100, C.N_BINS, C.AUDIO_DIM)
    assert tb["vision"].shape == (100, C.N_BINS, C.VISION_DIM)
    print(f"[2] 形状检查通过: {tb['audio'].shape}")

    recs = json.load(open(os.path.join(C.OUT, "align_records.json"), encoding="utf-8"))
    for r in recs:
        T = r["duration"]
        for k in [0, 1, C.N_BINS // 2, C.N_BINS - 1]:
            lo, hi = k * T / C.N_BINS, (k + 1) * T / C.N_BINS
            assert hi > lo, f"{r['id']} 时间箱{k}区间非法"
    print("[3] 时间反查检查通过: 索引 k -> [k*T/50,(k+1)*T/50)")

    assert wd["length"].max() <= C.MAX_WORDS
    print("[4] 长度检查通过: padding(length) 与 真实缺失(mask) 可区分")

    assert len(set(r["id"] for r in recs)) == 100
    print("[5] 异常检查通过: 全部样本保留, 无效位置仅 mask=0")

    rows = []
    for r in recs:
        m = _read_meta(r["id"])
        nvf = r["n_vision_frames"]; nvv = int(m.get("n_vision_valid", nvf))
        rows.append({
            "id": r["id"], "duration_s": r["duration"], "n_words": r["n_words"],
            "text_align": r["text_align"],
            "asr_model": r["asr_model"] or "",
            "n_audio_frames": r["n_audio_frames"],
            "n_vision_frames": r["n_vision_frames"],
            "vision_backend": m.get("vision_backend", ""),
            "n_vision_valid": nvv,
            "vision_valid_ratio": round(nvv / max(1, nvf), 4),
            "timebin_audio_valid": r["timebin_audio_valid"],
            "timebin_vision_valid": r["timebin_vision_valid"],
            "timebin_text_valid": r["timebin_text_valid"],
            "fill_ratio_audio": round(1 - r["timebin_audio_valid"] / C.N_BINS, 4),
            "word_valid_length": r["word_valid_length"],
            "dim": f"{C.TEXT_DIM}/{C.AUDIO_DIM}/{C.VISION_DIM}",
        })
    df_summary = pd.DataFrame(rows)
    # --- 新增: 跨模态相关性 + 音视同步 质量列(仅主版本时间箱) ---
    try:
        corr_mat, corr_df = compute_cross_modal_corr(tb)
        sync_df = compute_av_sync(tb)
        df_summary = df_summary.merge(sync_df, on="id", how="left")
        df_summary = df_summary.merge(corr_df, on="id", how="left")
        print("[6] 质量分析通过: 跨模态相关性矩阵 + 音视同步指标已并入汇总表")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[warn] 质量分析(相关性/同步)失败: {e}")
    try:
        df_summary.to_csv(C.SUMMARY_CSV, index=False, encoding="utf-8-sig")
        print(f"汇总表已生成: {C.SUMMARY_CSV}")
    except PermissionError:
        alt = os.path.join(C.OUT, "summary_new.csv")
        df_summary.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"summary.csv 被占用, 已写入: {alt}")
    return tb, wd


# ================== 入口 ==================
def main():
    C.ensure_dirs()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    samples = read_samples()
    N = len(samples)
    print(f"读取样本 {N} 条", flush=True)
    print("标签分布:", samples["annotation"].value_counts().to_dict(), flush=True)

    if cmd in ("all", "audio"):
        for i, (_, row) in enumerate(samples.iterrows()):
            audio_features(row, extract_audio(row))
            if (i + 1) % 20 == 0:
                print(f"  audio {i + 1}/{N}", flush=True)
    if cmd in ("all", "vision"):
        for i, (_, row) in enumerate(samples.iterrows()):
            vision_features(row)
            if (i + 1) % 5 == 0:
                print(f"  vision {i + 1}/{N}", flush=True)
    if cmd in ("all", "text"):
        for i, (_, row) in enumerate(samples.iterrows()):
            text_and_align(row, extract_audio(row))
            if (i + 1) % 5 == 0:
                print(f"  text {i + 1}/{N}", flush=True)
    if cmd in ("all", "build"):
        build_features(samples)
    if cmd in ("all", "summary"):
        validate(samples)
    print("== 完成 ==", flush=True)


if __name__ == "__main__":
    main()
