# -*- coding: utf-8 -*-
"""模块三 可解释性核心：模型集成、Shapley 模态贡献、局部证据（遮挡 LOPO + IG）。"""
import os
import math
import numpy as np
import torch

import config_q3 as C
import data_q3 as D3
import data_q2 as D
from model_q2 import Q2Net
from train_q2 import get_data, device_of, load_ckpt, predict


def build_backbone(dev):
    # 模块三 骨干与模块二 final_v2 一致（含覆盖率先验；完整数据下 log(c_eff)=0 不影响门控）
    return Q2Net(use_key_mask=True, use_gate=True, use_cross=True, use_recon=True,
                 use_moe=True, use_nll=True, use_cov_prior=True, use_moe_feats=True,
                 fusion_dropout=0.0).to(dev)


def load_models():
    """加载模块二 final_v2 三种子作为 Q3 骨干；返回 (models, dt)。"""
    dev = device_of()
    dt = get_data()
    models = []
    for s in C.SEEDS:
        ck = load_ckpt(os.path.join(C.Q2_CKPT, f"final_v2_s{s}.pt"))
        m = build_backbone(dev)
        m.load_state_dict(ck["state"])
        m.eval()
        models.append((ck, m))
    return models, dt


@torch.no_grad()
def ens_predict(models, Xn, Mn, dev):
    Ps, Rs, As, Vs = [], [], [], []
    for ck, m in models:
        p, r, a, lv = predict(m, Xn, Mn, dev)
        Ps.append(p); Rs.append(r); As.append(a)
        if lv is not None:
            Vs.append(np.exp(lv))           # exp(log_var) = 方差 sigma^2
    P = np.mean(Ps, 0); R = np.mean(Rs, 0); A = np.mean(As, 0)
    V = np.mean(Vs, 0) if Vs else None       # 平均方差
    return P, R, A, V


# ---------------- Shapley 模态贡献 ----------------
def _subsets():
    return [[], ["text"], ["audio"], ["vision"],
            ["text", "audio"], ["text", "vision"],
            ["audio", "vision"], ["text", "audio", "vision"]]


def _logit_c_ensemble(models, Xt, Mt, c, dev):
    """三个模型对第 c 类 logit 的平均。Xt/Mt 为 torch tensor dict。"""
    tot = 0.0
    for _, m in models:
        out = m(Xt, Mt)
        tot = tot + out["logits"][:, c]
    return tot / len(models)


@torch.no_grad()
def shapley_values(models, Xn, Mn, c, dev):
    """三模态精确 Shapley。Xn/Mn 为单样本 numpy 字典（各 1×50×d）。

    v(S) = 平均 logit（只保留 S 中模态，其余特征置 0 且 mask 置 0）
    phi_m = sum_{S subseteq N\\{m}}  |S|!(|N|-|S|-1)! / |N|!  * (v(S∪{m}) - v(S))
    """
    x = {m: torch.tensor(Xn[m], device=dev) for m in C.MODS}
    msk = {m: torch.tensor(Mn[m], device=dev) for m in C.MODS}
    zero = {m: torch.zeros_like(x[m]) for m in C.MODS}
    zmsk = {m: torch.zeros_like(msk[m]) for m in C.MODS}
    key = lambda s: tuple(sorted(s))

    def v(subset):
        Xt, Mt = {}, {}
        for m in C.MODS:
            if m in subset:
                Xt[m], Mt[m] = x[m], msk[m]
            else:
                Xt[m], Mt[m] = zero[m], zmsk[m]
        return float(_logit_c_ensemble(models, Xt, Mt, c, dev).item())

    vals = {key(s): v(s) for s in _subsets()}
    Mset = list(C.MODS)
    n = len(Mset)
    phi = {m: 0.0 for m in Mset}
    for m in Mset:
        for S in _subsets():
            if m in S:
                continue
            k = len(S)
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            Su = key(list(S) + [m])
            phi[m] += w * (vals[Su] - vals[key(S)])
    pi = softmax_ratio(phi, C.SHAPLEY_TAU)
    return phi, pi


def softmax_ratio(phi, tau=1.0):
    v = np.array([phi[m] for m in C.MODS], dtype=np.float64)
    v = (v - v.mean()) / (v.std() + 1e-8)     # 温度前标准化，避免尺度影响
    e = np.exp(v / tau)
    return {m: float(e[i] / e.sum()) for i, m in enumerate(C.MODS)}


# ---------------- 遮挡 (Leave-One-Position-Out) 局部证据 ----------------
@torch.no_grad()
def occlusion_importance(models, Xn, Mn, c, dev):
    """逐位置遮挡（LOPO）重要性。

    I_m(t) = logit_c(full) - logit_c(把模态 m 的第 t 个位置置 0)
    返回 {m: (50,) float32}，正值表示"去掉该位置会降低目标类置信度"（即证据）。
    用批处理：把 50 个"各遮一个位置"的副本一次送入模型。
    """
    B = 1
    L = C.L
    # 基值 logit
    Xf = {k: torch.tensor(Xn[k], device=dev, dtype=torch.float32) for k in C.MODS}
    Mf = {k: torch.tensor(Mn[k], device=dev) for k in C.MODS}
    base = float(_logit_c_ensemble(models, Xf, Mf, c, dev).item())

    out = {}
    for m in C.MODS:
        x = Xf[m].repeat(L, 1, 1)            # (L,50,d)
        rows = torch.arange(L, device=dev)
        x[rows, rows, :] = 0.0
        Xb = {k: (x if k == m else Xf[k].repeat(L, 1, 1)) for k in C.MODS}
        Mb = {k: Mf[k].repeat(L, 1) for k in C.MODS}
        Mb[m][rows, rows] = 0.0
        tot = 0.0
        for _, md in models:
            tot = tot + md(Xb, Mb)["logits"][:, c]
        logit = tot / len(models)            # (L,)
        imp = base - logit.detach().cpu().numpy().astype(np.float32)
        out[m] = imp
    return out


def normalize_importance(imp):
    """把重要性归一化到 [0,1]（按最大值），保持相对差异。"""
    out = {}
    for m, w in imp.items():
        w = np.asarray(w, dtype=np.float32)
        mx = w.max()
        out[m] = (w / mx) if mx > 1e-12 else w
    return out


# ---------------- Integrated Gradients（梯度补充视角） ----------------
def _ig_modality(models, Xn, Mn, m, c, dev, steps=C.IG_STEPS):
    x = torch.tensor(Xn[m], device=dev, dtype=torch.float32)      # (1,50,d)
    base = torch.zeros_like(x)
    Mbase = {k: torch.tensor(Mn[k], device=dev) for k in C.MODS}
    Xbase = {k: torch.tensor(Xn[k], device=dev) for k in C.MODS}
    grads = []
    for k in range(1, steps + 1):
        alpha = k / steps
        xi = base + alpha * (x - base)
        xi = xi.detach().requires_grad_(True)
        Xt = dict(Xbase)
        Xt[m] = xi
        Mt = dict(Mbase)
        logit = _logit_c_ensemble(models, Xt, Mt, c, dev)
        g = torch.autograd.grad(logit, xi)[0]
        grads.append(g.detach())
    gmean = torch.stack(grads).mean(0)          # (1,50,d)
    ig = (x - base) * gmean                      # (1,50,d)
    return ig.detach().cpu().numpy()[0]          # (50,d)


def local_importance(models, Xn, Mn, c, dev, steps=C.IG_STEPS):
    """Integrated Gradients 局部重要性（梯度视角），L2 聚合。"""
    out = {}
    for m in C.MODS:
        ig = _ig_modality(models, Xn, Mn, m, c, dev, steps)
        w = np.linalg.norm(ig, axis=1)
        mx = w.max()
        if mx > 1e-12:
            w = w / mx
        out[m] = w.astype(np.float32)
    return out


def topk_positions(w, k=C.TOP_K, mask=None):
    """返回重要性最高的 k 个位置（过滤 mask=0 的填充位）。"""
    w = np.asarray(w)
    order = np.argsort(-w)
    if mask is not None:
        order = [i for i in order if mask[i] > 0]
    return [int(i) for i in order[:k]]


# ---------------- 证据落回 ----------------
def text_evidence(raw_text, positions):
    picked = [D3.word_for_position(raw_text, i) for i in positions]
    return {"words": [w for w in picked if w], "positions": [int(i) for i in positions]}


def audio_evidence(idx, positions):
    dur, fps, nf = D3.video_meta(idx)
    wins = [[round(p * dur / 50.0, 2), round((p + 1) * dur / 50.0, 2)] for p in positions]
    return {"time_windows": wins, "positions": [int(i) for i in positions], "duration": round(dur, 2)}


def vision_evidence(idx, positions):
    frames = []
    for p in positions:
        op = os.path.join(C.Q3_FRAMES, f"a4_{idx:02d}_frame_{p:02d}.jpg")
        ok = D3.extract_frame(idx, p, op)
        frames.append(ok or None)
    return {"key_frames": frames, "positions": [int(i) for i in positions]}
