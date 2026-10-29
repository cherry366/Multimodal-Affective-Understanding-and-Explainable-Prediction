# -*- coding: utf-8 -*-
"""模块三 数据层：标准化特征（复用） + 可解释测试集（numpy2 兼容读取 + 证据落回辅助）。"""
import os
import glob
import pickle
import numpy as np
import cv2

import config_q3 as C


class CompatUnpickler(pickle.Unpickler):
    """可解释测试集的 pkl 由 numpy 2.x 保存，当前 numpy 1.x 需把 numpy._core 重映射到 numpy.core。"""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_a4_aligned(idx):
    """读取可解释测试集对齐版第 idx 条（idx 从 1 开始）。返回样本字典。"""
    p = os.path.join(C.A4_DIR, f"{idx:02d}.pkl")
    with open(p, "rb") as f:
        return CompatUnpickler(f).load()


def load_all_a4():
    """读取可解释测试集全部 20 条对齐版，打包成与标准化特征一致的格式。

    文本 mask 复用标准化特征规则：attention_mask × (token != [UNK]=100)；
    音频/视觉 mask 用"整行是否全零"判定。
    """
    ids, text, audio, vision, raw, tb = [], [], [], [], [], []
    for i in range(1, 21):
        s = load_a4_aligned(i)
        ids.append(s["id"])
        raw.append(s["raw_text"])
        text.append(np.asarray(s["text"], dtype=np.float32))
        audio.append(np.asarray(s["audio"], dtype=np.float32))
        vision.append(np.asarray(s["vision"], dtype=np.float32))
        tb.append(np.asarray(s["text_bert"]).astype(np.int64).reshape(3, C.L))
    text = np.stack(text)
    audio = np.stack(audio)
    vision = np.stack(vision)
    tb = np.stack(tb)
    att = tb[:, 1, :].astype(np.float32)
    mask = {
        "text": att * (tb[:, 0, :] != 100).astype(np.float32),
        "audio": (np.abs(audio).sum(-1) > 0).astype(np.float32),
        "vision": (np.abs(vision).sum(-1) > 0).astype(np.float32),
    }
    return {"id": ids, "raw_text": raw, "text": text, "audio": audio, "vision": vision,
            "text_bert": tb, "mask": mask}


def words_of(raw_text):
    """把 raw_text 拆成词列表，供文本证据落回。"""
    return raw_text.strip().split()


def word_for_position(raw_text, pos):
    """把 aligned 位置 pos(0..49) 近似映射到原始文本中的词。

    对齐版 text 是 50 个词级槽位，但 raw_text 分词数可能小于 50；
    采用比例映射 pos/L -> 词下标，保证证据始终能落回原文。
    """
    words = words_of(raw_text)
    if not words:
        return None
    if len(words) <= C.L:
        idx = int(pos)
    else:
        idx = int(pos * len(words) / C.L)
    idx = min(max(idx, 0), len(words) - 1)
    return words[idx]


def video_meta(idx):
    """读取可解释测试集第 idx 条原始视频的时长(秒)、fps、总帧数。"""
    p = os.path.join(C.A4_DIR, "videos", f"{idx:02d}.mp4")
    cap = cv2.VideoCapture(p)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = nframes / fps if fps > 0 else 0.0
    cap.release()
    return dur, fps, nframes


def _write_img(path, frame):
    """用 imencode + python 写文件，避免 OpenCV 对中文路径写失败。"""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def extract_frame(idx, pos, out_path):
    """抽取第 idx 条视频在 aligned 位置 pos(0..49) 对应的关键帧，保存为 jpg。

    对 seek 失败做 ±1、±2 帧的邻近回退，保证关键帧稳定可读。
    """
    dur, fps, nframes = video_meta(idx)
    if nframes <= 0:
        return None
    t = (pos + 0.5) * (dur / 50.0)
    fid0 = int(t * fps)
    p = os.path.join(C.A4_DIR, "videos", f"{idx:02d}.mp4")
    cands = [fid0, fid0-1, fid0+1, fid0-2, fid0+2, fid0-3, fid0+3]
    frame = None
    for fid in cands:
        fid = min(max(fid, 0), nframes - 1)
        cap = cv2.VideoCapture(p)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, fr = cap.read()
        cap.release()
        if ok and fr is not None:
            frame = fr
            break
    if frame is None:
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path if _write_img(out_path, frame) else None
