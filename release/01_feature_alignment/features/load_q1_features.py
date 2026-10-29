# -*- coding: utf-8 -*-
"""模块一 特征文件读取示例：同时支持 .pkl 与 .pkl.gz。

用法：
    from load_q1_features import load_q1_features
    d = load_q1_features("q1_features_timebin.pkl.gz")["test"]
    print(d["text"].shape, d["text_mask"].shape)   # (100, 50, 768) (100, 50)

字段（两版一致，word 版多一个 "length"）：
    id, raw_text, text, audio, vision,
    text_mask, audio_mask, vision_mask,
    duration, n_audio_frames, n_vision_frames,
    align_mode, annotations, classification_labels, regression_labels,
    text_norm, audio_norm, vision_norm   （归一化版本，可直接训练）
"""
import gzip
import pickle


def load_q1_features(path):
    """读取模块一自生成特征文件（.pkl 或 .pkl.gz），返回原始 dict。"""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "q1_features_timebin.pkl.gz"
    obj = load_q1_features(p)
    d = obj["test"]
    print("文件:", p, "| 样本数:", len(d["id"]), "| 字段数:", len(d))
    for k in ("text", "audio", "vision", "text_mask"):
        v = d[k]
        print("  %-10s %s %s" % (k, getattr(v, "shape", type(v)), getattr(v, "dtype", "")))
