# -*- coding: utf-8 -*-
"""模块二 训练与验证（MoDE-Former+ v2）

用法:
    python train_q2.py final_v2 42          # 训练单个变体单个种子
    python train_q2.py final_v2             # 训练单个变体全部种子
    python train_q2.py all                  # 训练全部变体 × 全部种子

变体（v2 及组件消融）:
    final_v2   : 覆盖率先验门控 + InfoNCE + MoE(缺失特征路由) + KD + NLL + EMA
    v2_nocov   : 去掉覆盖率先验门控
    v2_noctr   : 去掉 InfoNCE 不变性损失
    v2_nomoe   : 去掉模态内专家路由
    v2_nokd    : 去掉知识蒸馏
    v2_nonll   : 去掉异方差不确定性
    （旧变体 full / final 等保留用于对比与教师模型）
"""
import argparse
import io
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr

import config_q2 as C
import data_q2 as D
from model_q2 import Q2Net

VARIANT_SPECS = {
    "full":      dict(use_key_mask=True,  use_gate=True,  use_aug=True,  use_recon=True),
    "norecon":   dict(use_key_mask=True,  use_gate=True,  use_aug=True,  use_recon=False),
    "mask_only": dict(use_key_mask=True,  use_gate=False, use_aug=True,  use_recon=False),
    "gate_only": dict(use_key_mask=False, use_gate=True,  use_aug=True,  use_recon=False),
    "nomask":    dict(use_key_mask=False, use_gate=False, use_aug=True,  use_recon=False),
    "noaug":     dict(use_key_mask=True,  use_gate=True,  use_aug=False, use_recon=False),
    "final":     dict(use_key_mask=True,  use_gate=True,  use_aug=True,  use_recon=True,
                      use_mod_emb=True, use_moe=True, use_kd=True, use_inv=True, use_nll=True),
    # ---- v2.1 最终方案（cov-prior 门控；对照实验证明 InfoNCE/EMA/KD退火/融合dropout 有害或无效）----
    "final_v2":  dict(use_key_mask=True, use_gate=True, use_aug=True, use_recon=True,
                      use_moe=True, use_kd=True, use_nll=True,
                      use_ctr=False, use_cov_prior=True, use_moe_feats=True),
    "v2_nocov":  dict(use_key_mask=True, use_gate=True, use_aug=True, use_recon=True,
                      use_moe=True, use_kd=True, use_nll=True,
                      use_ctr=False, use_cov_prior=False, use_moe_feats=True),
    "v2_nomoe":  dict(use_key_mask=True, use_gate=True, use_aug=True, use_recon=True,
                      use_moe=False, use_kd=True, use_nll=True,
                      use_ctr=False, use_cov_prior=True, use_moe_feats=True),
    "v2_nokd":   dict(use_key_mask=True, use_gate=True, use_aug=True, use_recon=True,
                      use_moe=True, use_kd=False, use_nll=True,
                      use_ctr=False, use_cov_prior=True, use_moe_feats=True),
    "v2_nonll":  dict(use_key_mask=True, use_gate=True, use_aug=True, use_recon=True,
                      use_moe=True, use_kd=True, use_nll=False,
                      use_ctr=False, use_cov_prior=True, use_moe_feats=True),
}

_C = {}


# ------------------------------------------------------------------ 数据
def get_data():
    if "items" not in _C:
        items = D.load_aligned()
        stats = D.fit_stats(items, "train")
        _C["items"] = items
        _C["stats"] = stats
        _C["X"] = {sp: D.norm_pack(items, stats, sp) for sp in items}
        _C["M"] = {sp: {m: items[sp]["mask"][m] for m in C.MODS} for sp in items}
        _C["L"] = {sp: items[sp]["leff"] for sp in items}
    return _C


def device_of():
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_batch(dt, split, idx, rng, augment):
    X, M, Lf = dt["X"][split], dt["M"][split], dt["L"][split]
    Xo = {m: X[m][idx] for m in C.MODS}
    Xb = {m: X[m][idx] for m in C.MODS}
    Mo = {m: M[m][idx] for m in C.MODS}
    Mb = {m: M[m][idx] for m in C.MODS}
    Lb = {m: Lf[m][idx] for m in C.MODS}
    if augment:
        Xb, Mb, inj = D.inject(Xb, Mb, Lb, rng, clean_ratio=C.CLEAN_RATIO)
    else:
        inj = {m: np.zeros(Mb[m].shape, dtype=bool) for m in C.MODS}
    return Xo, Xb, Mb, Mo, inj


def to_t(T, device):
    return {m: torch.tensor(T[m], device=device) for m in C.MODS}


# ------------------------------------------------------------------ 指标
def decision(probs, reg, tau, mode="band"):
    am = probs.argmax(1)
    if tau <= 0:
        return am
    if mode == "band":
        return np.where(np.abs(reg) < tau, 1, am)
    s = np.where(reg > 0, 2, 0)
    return np.where(np.abs(reg) < tau, 1, s)


def metrics_of(probs, reg, y_cls, y_reg, tau=0.0, mode="band"):
    pred = decision(probs, reg, tau, mode)
    out = {
        "acc": float(accuracy_score(y_cls, pred)),
        "f1_macro": float(f1_score(y_cls, pred, average="macro")),
        "f1_weighted": float(f1_score(y_cls, pred, average="weighted")),
        "mae": float(np.abs(reg - y_reg).mean()),
        "pearson": float(pearsonr(reg, y_reg)[0]) if np.std(reg) > 1e-9 else 0.0,
    }
    return out


def best_tau(probs, reg, y_cls):
    best = (-1.0, 0.0, "band")
    for tau in [0.0] + C.TAU_GRID:
        for mode in ("band", "sign"):
            pred = decision(probs, reg, tau, mode)
            f = float(f1_score(y_cls, pred, average="macro"))
            if f > best[0]:
                best = (f, tau, mode)
    return best


# ------------------------------------------------------------------ 推理
@torch.no_grad()
def predict(model, Xb, Mb, device):
    model.eval()
    out = model(to_t(Xb, device), to_t(Mb, device))
    lv = out["log_var"].cpu().numpy() if "log_var" in out else None
    return (torch.softmax(out["logits"], 1).cpu().numpy(),
            out["reg"].cpu().numpy(),
            out["alpha"].cpu().numpy(),
            lv)


def eval_split(model, dt, split, device, mode="clean", draws=1, seed0=1234, idx=None):
    X, M, Lf = dt["X"][split], dt["M"][split], dt["L"][split]
    N = X["text"].shape[0]
    idx = np.arange(N) if idx is None else np.asarray(idx)
    y_cls = dt["items"][split]["y_cls"][idx]
    y_reg = dt["items"][split]["y_reg"][idx]
    Ps, Rs = [], []
    for k in range(max(1, draws)):
        if mode == "clean":
            Xb = {m: X[m][idx] for m in C.MODS}
            Mb = {m: M[m][idx] for m in C.MODS}
        else:
            rng = np.random.default_rng(seed0 + 100 * k)
            Xb = {m: X[m][idx] for m in C.MODS}
            Mb = {m: M[m][idx] for m in C.MODS}
            Lb = {m: Lf[m][idx] for m in C.MODS}
            Xb, Mb, _ = D.a3_like_inject(Xb, Mb, Lb, rng)
        p, r, _, _ = predict(model, Xb, Mb, device)
        Ps.append(p)
        Rs.append(r)
    P = np.mean(Ps, 0)
    R = np.mean(Rs, 0)
    return P, R, y_cls, y_reg


# ------------------------------------------------------------------ 损失
def info_nce(zc, zm, tau=0.1):
    """对比式不变性损失：clean 与 missing 两个视图的 InfoNCE。"""
    zc = F.normalize(zc, dim=1)
    zm = F.normalize(zm, dim=1)
    z = torch.cat([zc, zm], dim=0)                 # (2B, H)
    logits = (z @ z.t()) / tau                     # (2B, 2B)
    B = zc.shape[0]
    labels = torch.cat([torch.arange(B) + B, torch.arange(B)], dim=0).to(z.device)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(mask, -1e9)
    return F.cross_entropy(logits, labels)


def lr_schedule(step, warm_steps, total_steps, base=C.LR):
    if step < warm_steps:
        return base * (step + 1) / max(warm_steps, 1)
    p = (step - warm_steps) / max(total_steps - warm_steps, 1)
    return base * 0.5 * (1 + math.cos(math.pi * p))


# ------------------------------------------------------------------ 训练
def train_one(name, seed, epochs=None, verbose=True, overrides=None):
    spec = VARIANT_SPECS[name]
    dt = get_data()
    device = device_of()
    epochs = epochs or C.EPOCHS
    torch.manual_seed(seed)
    np.random.seed(seed)

    ov = overrides or {}
    lam_reg = ov.get("reg", C.LAMBDA_REG)
    lam_rec = ov.get("rec", C.MU_RECON)
    lam_kd0 = ov.get("kd", C.LAMBDA_KD)
    lam_ctr = ov.get("ctr", C.LAMBDA_CTR)
    lam_nll = ov.get("nll", C.LAMBDA_NLL)

    model = Q2Net(use_key_mask=spec["use_key_mask"], use_gate=spec["use_gate"],
                  use_cross=True, use_recon=spec["use_recon"],
                  use_mod_emb=spec.get("use_mod_emb", False),
                  use_moe=spec.get("use_moe", False),
                  use_nll=spec.get("use_nll", False),
                  use_cov_prior=spec.get("use_cov_prior", False),
                  cov_prior_init=C.COV_PRIOR_INIT,
                  use_moe_feats=spec.get("use_moe_feats", False),
                  fusion_dropout=C.FUSION_DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WD)

    teacher = None
    if spec.get("use_kd", False):
        tpath = os.path.join(C.CKPT, f"full_s{seed}.pt")
        if not os.path.exists(tpath):
            raise FileNotFoundError(f"KD teacher 需要先训练: {tpath}")
        teacher = Q2Net(use_key_mask=True, use_gate=True, use_cross=True,
                        use_recon=True).to(device)
        teacher.load_state_dict(load_ckpt(tpath)["state"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    y_tr = dt["items"]["train"]["y_cls"]
    if C.CLS_WEIGHTED:
        cnt = np.bincount(y_tr, minlength=3).astype(np.float64)
        w = (cnt.sum() / (3.0 * np.maximum(cnt, 1)))
        cls_w = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        cls_w = None

    rng = np.random.default_rng(seed)
    N = dt["X"]["train"]["text"].shape[0]
    hist, best = [], {"score": -9, "state": None, "epoch": -1, "tau": 0.0, "mode": "band"}

    steps_per_epoch = int(math.ceil(N / C.BATCH))
    total_steps = epochs * steps_per_epoch
    warm_steps = C.WARMUP_EPOCHS * steps_per_epoch
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    global_step = 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(N)
        tot, nb = 0.0, 0
        t0 = time.time()
        kd_weight = lam_kd0 * max(0.0, 1.0 - ep / max(epochs, 1)) if C.KD_ANNEAL else lam_kd0
        for i in range(0, N, C.BATCH):
            for g in opt.param_groups:
                g["lr"] = lr_schedule(global_step, warm_steps, total_steps)
            idx = perm[i:i + C.BATCH]
            Xo, Xb, Mb, Mo, inj = make_batch(dt, "train", idx, rng, spec["use_aug"])
            Xo_t, X_t, M_t = to_t(Xo, device), to_t(Xb, device), to_t(Mb, device)
            Mo_t = to_t(Mo, device)
            yc = torch.tensor(y_tr[idx], device=device)
            yr = torch.tensor(dt["items"]["train"]["y_reg"][idx], device=device)
            out = model(X_t, M_t)
            loss = F.cross_entropy(out["logits"], yc, weight=cls_w)
            loss = loss + lam_reg * F.smooth_l1_loss(out["reg"], yr, beta=1.0)
            if "recon" in out and lam_rec > 0:
                rec, nrec = 0.0, 0
                for m in C.MODS:
                    sel = torch.tensor(inj[m], device=device)
                    if sel.sum() == 0:
                        continue
                    rec = rec + F.mse_loss(out["recon"][m][sel], Xo_t[m][sel])
                    nrec += 1
                if nrec:
                    loss = loss + lam_rec * rec / nrec
            if teacher is not None and kd_weight > 0:
                with torch.no_grad():
                    zT = teacher(Xo_t, Mo_t)["logits"]
                kd = F.kl_div(F.log_softmax(out["logits"] / C.KD_TAU, 1),
                              F.softmax(zT / C.KD_TAU, 1), reduction="batchmean")
                loss = loss + kd_weight * (C.KD_TAU ** 2) * kd
            if spec.get("use_ctr", False) and lam_ctr > 0:
                zc = model(Xo_t, Mo_t)["z"]
                loss = loss + lam_ctr * info_nce(zc, out["z"], C.CTR_TAU)
            if spec.get("use_inv", False):
                zc = model(Xo_t, Mo_t)["z"]
                inv = ((zc.mean(0) - out["z"].mean(0)) ** 2).sum()
                loss = loss + C.LAMBDA_INV * inv
            if spec.get("use_nll", False) and "log_var" in out:
                lv = out["log_var"]
                nll = 0.5 * (((yr - out["reg"].detach()) ** 2) * torch.exp(-lv) + lv).mean()
                loss = loss + lam_nll * nll
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if C.EMA_DECAY > 0:
                d = C.EMA_DECAY
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema[k].mul_(d).add_(v, alpha=1 - d)
            global_step += 1
            tot += float(loss.item())
            nb += 1

        live_state = None
        if C.EMA_DECAY > 0:
            live_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema)
        P, R, yc_v, yr_v = eval_split(model, dt, "valid", device, mode="a3", draws=3)
        f1_a, tau, tmode = best_tau(P, R, yc_v)
        m_a = metrics_of(P, R, yc_v, yr_v, tau, tmode)
        Pc, Rc, _, _ = eval_split(model, dt, "valid", device, mode="clean")
        m_c = metrics_of(Pc, Rc, yc_v, yr_v, tau, tmode)
        score = m_a["f1_macro"] + 0.2 * max(0.0, m_a["pearson"]) - C.SCORE_MAE_W * m_a["mae"]
        if live_state is not None:
            model.load_state_dict(live_state)
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), "val_a3": m_a, "val_clean": m_c,
                     "tau": tau, "tau_mode": tmode, "score": score, "sec": time.time() - t0})
        if verbose:
            print(f"[{name} s{seed}] ep{ep:02d} loss={tot/max(nb,1):.4f} "
                  f"a3:F1={m_a['f1_macro']:.4f} Acc={m_a['acc']:.4f} MAE={m_a['mae']:.4f} r={m_a['pearson']:.4f} | "
                  f"clean:F1={m_c['f1_macro']:.4f} Acc={m_c['acc']:.4f} | tau={tau:.2f} {time.time()-t0:.1f}s",
                  flush=True)
        ref_state = ema if C.EMA_DECAY > 0 else model.state_dict()
        if ep >= C.MIN_EPOCHS and score > best["score"]:
            best = {"score": score, "epoch": ep, "tau": tau, "mode": tmode,
                    "state": {k: v.detach().cpu().clone() for k, v in ref_state.items()}}
        elif ep >= C.MIN_EPOCHS and ep - best["epoch"] >= C.PATIENCE:
            if verbose:
                print(f"[{name} s{seed}] early stop @ ep{ep} (best ep{best['epoch']})", flush=True)
            break

    if best["state"] is None:
        best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best["state"])
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    res = {"variant": name, "seed": seed, "best_epoch": best["epoch"], "tau": best["tau"],
           "tau_mode": best["mode"], "spec": spec, "hist": hist}
    for split, draws in (("valid", C.N_EVAL_DRAWS), ("test", C.N_EVAL_DRAWS)):
        for mode in ("clean", "a3"):
            P, R, yc_, yr_ = eval_split(model, dt, split, device, mode=mode, draws=draws)
            res[f"{split}_{mode}"] = metrics_of(P, R, yc_, yr_, best["tau"], best["mode"])
    res["ckpt"] = os.path.join(C.CKPT, f"{name}_s{seed}.pt")
    save_ckpt(res["ckpt"], {"state": final_state, "spec": spec, "seed": seed,
                            "tau": best["tau"], "tau_mode": best["mode"],
                            "variant": name, "metrics": res})
    if verbose:
        print(f"[{name} s{seed}] DONE  valid_a3={json.dumps(res['valid_a3'])}", flush=True)
        print(f"[{name} s{seed}]       test_a3={json.dumps(res['test_a3'])}", flush=True)
    return res


def save_ckpt(path, obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    with open(path, "wb") as f:
        f.write(buf.getvalue())


def load_ckpt(path):
    with open(path, "rb") as f:
        ck = torch.load(io.BytesIO(f.read()), map_location="cpu")
    # 权重以 fp16 存储以控制数据体积，加载时统一转回 fp32 再使用
    if "state" in ck:
        ck["state"] = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                       for k, v in ck["state"].items()}
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("seed", nargs="?", default=None, type=int)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    jobs = []
    if args.variant == "all":
        for v in VARIANT_SPECS:
            for s in C.SEEDS:
                jobs.append((v, s))
    else:
        seeds = [args.seed] if args.seed is not None else C.SEEDS
        for s in seeds:
            jobs.append((args.variant, s))
    rows = []
    for v, s in jobs:
        r = train_one(v, s, epochs=args.epochs)
        rows.append({k: r[k] for k in ("variant", "seed", "best_epoch", "tau", "tau_mode")} |
                    {f"valid_clean_{k}": val for k, val in r["valid_clean"].items()} |
                    {f"valid_a3_{k}": val for k, val in r["valid_a3"].items()} |
                    {f"test_clean_{k}": val for k, val in r["test_clean"].items()} |
                    {f"test_a3_{k}": val for k, val in r["test_a3"].items()})
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "train_summary.csv"), index=False, encoding="utf-8-sig")
    print("all jobs done:", len(rows), flush=True)


if __name__ == "__main__":
    main()
