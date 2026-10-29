# -*- coding: utf-8 -*-
"""模块二 数据层：标准化特征/ 缺失测试集读取、mask 构造、全局标准化、缺失注入

核心约定
--------
1) mask：音频/视觉用"整行是否全零"判定；文本用 attention_mask 且排除 [UNK](id=100)
2) 有效区间：leff = 最后一个有效位置 + 1（避开尾部填充）
3) 缺失注入只发生在 [lo, leff) 内，且不覆盖原有填充位
4) 标准化统计量只用标准化特征train 的有效格估计（避免逐样本统计被缺失污染）
"""
import os
import pickle
import glob
import numpy as np

import config_q2 as C


# ---------------------------------------------------------------- 基础工具
def row_valid(X):
    """(N,50,d) -> (N,50) 行全零判定（1=有效）"""
    return (np.abs(X).sum(-1) > 0).astype(np.float32)


def leff_of(M):
    """(N,50) -> (N,) 最后一个有效位置+1"""
    idx = np.where(M > 0, np.arange(M.shape[1])[None, :], -1)
    return (idx.max(1) + 1).astype(np.int64)


# ---------------------------------------------------------------- 标准化特征
def load_aligned():
    d = pickle.load(open(C.P2_ALIGNED, "rb"))
    out = {}
    for sp in ["train", "valid", "test"]:
        s = d[sp]
        TB = np.asarray(s["text_bert"]).astype(np.int64)
        att = TB[:, 1, :].astype(np.float32)
        item = {
            "id": list(s["id"]),
            "text": np.asarray(s["text"], dtype=np.float32),
            "audio": np.asarray(s["audio"], dtype=np.float32),
            "vision": np.asarray(s["vision"], dtype=np.float32),
            "text_bert": TB,
            "y_cls": np.asarray(s["classification_labels"]).astype(np.int64),
            "y_reg": np.asarray(s["regression_labels"]).astype(np.float32),
        }
        M = {
            "text": att * (TB[:, 0, :] != 100).astype(np.float32),
            "audio": row_valid(item["audio"]),
            "vision": row_valid(item["vision"]),
        }
        item["mask"] = M
        item["leff"] = {m: leff_of(M[m]) for m in C.MODS}
        out[sp] = item
    return out


# ---------------------------------------------------------------- 缺失测试集
def load_a3_aligned(bert=None, device="cpu", use_bert=True):
    """读取缺失测试集对齐版 30 条：文本特征用 text_bert 经 BERT 复现（与标准化特征同一条链路）"""
    files = sorted(glob.glob(os.path.join(C.A3_DIR, "*.pkl")))
    ids, Ys, Ms, TBs = [], {m: [] for m in C.MODS}, {m: [] for m in C.MODS}, []
    for f in files:
        t = pickle.load(open(f, "rb"))
        key = "test" if "test" in t else list(t.keys())[0]
        s = t[key]
        ids.append(os.path.splitext(os.path.basename(f))[0])
        A = np.asarray(s["audio"], dtype=np.float32)[0]
        V = np.asarray(s["vision"], dtype=np.float32)[0]
        TB = np.asarray(s["text_bert"]).astype(np.int64).reshape(3, C.L)
        TBs.append(TB)
        Ys["audio"].append(A)
        Ys["vision"].append(V)
        Ms["audio"].append(row_valid(A[None])[0])
        Ms["vision"].append(row_valid(V[None])[0])
        att = TB[1].astype(np.float32)
        Ms["text"].append(att * (TB[0] != 100).astype(np.float32) * (TB[0] != 0).astype(np.float32))
    for m in ("audio", "vision"):
        Ys[m] = np.stack(Ys[m]).astype(np.float32)
        Ms[m] = np.stack(Ms[m]).astype(np.float32)
    Ms["text"] = np.stack(Ms["text"]).astype(np.float32)
    TBs = np.stack(TBs)

    if use_bert:
        import torch
        from transformers import BertModel
        if bert is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            bert = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
        with torch.no_grad():
            inp = torch.tensor(TBs[:, 0, :], device=device)
            am = torch.tensor(TBs[:, 1, :], device=device)
            tt = torch.tensor(TBs[:, 2, :], device=device)
            Ys["text"] = bert(input_ids=inp, attention_mask=am, token_type_ids=tt).last_hidden_state.cpu().numpy()
    else:
        raise RuntimeError("附属3 对齐版不含 768 维文本，必须用 BERT 复现")
    Ys["text"] = Ys["text"].astype(np.float32)
    out = {"id": ids, "text_bert": TBs}
    for m in C.MODS:
        out[m] = Ys[m]
    out["mask"] = Ms
    out["leff"] = {m: leff_of(Ms[m]) for m in C.MODS}
    return out


# ---------------------------------------------------------------- 标准化
def fit_stats(items, split="train", eps=1e-6):
    st = {}
    for m in C.MODS:
        X, M = items[split][m], items[split]["mask"][m]
        sel = M.reshape(-1) > 0
        V = X.reshape(-1, X.shape[-1])[sel]
        mu, sd = V.mean(0), V.std(0)
        sd = np.where(sd < eps, 1.0, sd)
        st[m] = (mu.astype(np.float32), sd.astype(np.float32))
    return st


def apply_norm(X, M, mu, sd):
    Z = (X - mu) / sd
    return (Z * M[..., None]).astype(np.float32)


def norm_pack(items, stats, split):
    out = {}
    for m in C.MODS:
        mu, sd = stats[m]
        out[m] = apply_norm(items[split][m], items[split]["mask"][m], mu, sd)
    return out


# ---------------------------------------------------------------- 缺失注入
def _place(rng, used, lo, hi, ln, tries=30):
    if ln <= 0 or hi - lo < ln:
        return False
    for _ in range(tries):
        s = int(rng.integers(lo, hi - ln + 1))
        if not used[s:s + ln].any():
            used[s:s + ln] = True
            return True
    for s in range(lo, hi - ln + 1):
        if not used[s:s + ln].any():
            used[s:s + ln] = True
            return True
    return False


def sample_positions(rng, lo, hi, r, K, segmax, position=None):
    """在 [lo,hi) 内采样缺失位置，返回索引数组"""
    span = hi - lo
    if span <= 1:
        return np.array([], dtype=int)
    total = int(round(r * span))
    total = int(np.clip(total, 0, span))
    if total == 0:
        return np.array([], dtype=int)
    used = np.zeros(hi, dtype=bool)
    if position is not None:
        ln = min(total, span)
        s = lo if position == "front" else (lo + (span - ln) // 2 if position == "middle" else hi - ln)
        used[s:s + ln] = True
        return np.where(used)[0]
    k = int(max(1, min(K, total)))
    rem = total
    while rem > 0 and k > 0:
        ln = int(rng.integers(1, min(segmax, rem) + 1))
        if _place(rng, used, lo, hi, ln):
            rem -= ln
            k -= 1
        else:
            break
    while rem > 0:
        if not _place(rng, used, lo, hi, 1):
            break
        rem -= 1
    return np.where(used)[0]


def inject(Xs, Ms, Leffs, rng, modes=None, r=None, K=None, segmax=None,
           position=None, indep=False, r_range=None, k_choices=None, k_prob=None,
           clean_ratio=0.0, scatter=False):
    """对一批样本注入缺失（原地修改副本）

    参数
    ----
    Xs, Ms : dict m -> (B,50[,d]) / (B,50)  标准化后的特征与掩码
    Leffs  : dict m -> (B,)                 各模态有效长度
    modes  : 指定缺失模态集合；None 表示按 P_SUBSET 随机
    indep  : True=各模态独立采样缺失区间（未对齐版形态）；False=同位置缺失（缺失测试集形态）
    返回
    ----
    Xn, Mn : 注入后的副本
    inj    : dict m -> (B,50) bool，仅"原本有效、后被置 0"的位置
    """
    B = Ms[C.MODS[0]].shape[0]
    Xn = {m: Xs[m].copy() for m in C.MODS}
    Mn = {m: Ms[m].copy() for m in C.MODS}
    inj = {m: np.zeros((B, C.L), dtype=bool) for m in C.MODS}
    nclean = int(round(B * clean_ratio))
    skip = set(rng.choice(B, size=nclean, replace=False).tolist()) if nclean > 0 else set()

    for b in range(B):
        if b in skip:
            continue
        if modes is None:
            nmod = int(rng.choice(list(C.P_SUBSET.keys()), p=list(C.P_SUBSET.values())))
            if nmod == 0:
                continue
            sel = list(rng.choice(C.MODS, size=nmod, replace=False))
        else:
            sel = list(modes)
        if len(sel) == 0:
            continue
        ind = bool(indep)
        pos0 = None
        for i, m in enumerate(sel):
            le = int(Leffs[m][b])
            if le <= 2:
                continue
            lo = 1
            hi = le - 1 if m == "text" else le      # 文本留出 [SEP]
            if hi - lo <= 1:
                continue
            if ind or i == 0:
                rb = float(r if r is not None else rng.uniform(*C.AUG_R_RANGE))
                kb = int(K if K is not None else rng.choice(k_choices or C.AUG_K_CHOICES,
                                                            p=None if k_choices else C.AUG_K_PROB))
                sm = int(segmax or C.AUG_SEG_MAX)
                if scatter:
                    cand = np.arange(lo, hi)
                    pos = cand[rng.random(len(cand)) < rb]
                else:
                    pos = sample_positions(rng, lo, hi, rb, kb, sm, position=position)
            else:
                pos = pos0
            if not ind and i == 0:
                pos0 = pos
            if pos is None or len(pos) == 0:
                continue
            was = Mn[m][b, pos] > 0
            if was.sum() == 0:
                continue
            Mn[m][b, pos[was]] = 0.0
            inj[m][b, pos[was]] = True
            if Xn[m].ndim == 3:
                Xn[m][b, pos[was], :] = 0.0
    return Xn, Mn, inj


def a3_like_inject(Xs, Ms, Leffs, rng, r=None):
    """缺失测试集同分布缺失：三模态同位置、r≈0.195、段长1~4、段数1~4"""
    return inject(Xs, Ms, Leffs, rng, modes=list(C.MODS),
                  r=C.A3_R if r is None else r, K=None,
                  segmax=C.A3_SEG_MAX, indep=C.A3_INDEP,
                  k_choices=C.A3_K_CHOICES, clean_ratio=0.0)


def missing_stats(M, leff):
    """每条样本的缺失统计（有效区内的缺失格数 / 缺失段数 / 缺失率）"""
    B = M.shape[0]
    out = np.zeros((B, 3), dtype=np.float32)
    for b in range(B):
        le = int(leff[b])
        if le <= 2:
            continue
        seg = M[b, 1:le]
        n_miss = int((seg == 0).sum())
        z = (seg == 0).astype(int)
        n_seg = int(((z[1:] == 1) & (z[:-1] == 0)).sum() + (1 if z[0] == 1 else 0))
        out[b] = (n_miss, n_seg, n_miss / max(le - 1, 1))
    return out