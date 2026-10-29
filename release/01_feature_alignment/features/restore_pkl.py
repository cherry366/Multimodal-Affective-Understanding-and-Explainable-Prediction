# -*- coding: utf-8 -*-
"""把 .pkl.gz 还原成原始 .pkl（解压后与流水线 pipeline.py 直接产出的文件逐字节一致）。

用法：
    python restore_pkl.py                 # 还原当前目录下两个文件
    python restore_pkl.py a.pkl.gz b.pkl.gz
"""
import gzip
import os
import shutil
import sys


def restore(gz_path):
    assert gz_path.endswith(".gz"), gz_path
    out = gz_path[:-3]
    with gzip.open(gz_path, "rb") as fi, open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    return out


if __name__ == "__main__":
    args = sys.argv[1:] or ["q1_features_timebin.pkl.gz", "q1_features_word.pkl.gz"]
    for a in args:
        if os.path.exists(a):
            p = restore(a)
            print("%s -> %s  (%.2f MB)" % (a, p, os.path.getsize(p) / 1048576))
        else:
            print("[跳过] 找不到", a)
