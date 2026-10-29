# -*- coding: utf-8 -*-
"""集成评估：final(v1) vs final_v2，验证/测试 clean+a3，3 种子平均预测。"""
import os
import json
import numpy as np
import torch

import config_q2 as C
import data_q2 as D
import train_q2 as T
from model_q2 import Q2Net


def build(name):
    if name == "final_v2":
        return Q2Net(use_key_mask=True, use_gate=True, use_cross=True, use_recon=True,
                     use_moe=True, use_nll=True, use_cov_prior=True,
                     use_moe_feats=True, fusion_dropout=C.FUSION_DROPOUT)
    if name == "final":
        return Q2Net(use_key_mask=True, use_gate=True, use_cross=True, use_recon=True,
                     use_mod_emb=True, use_moe=True, use_nll=True)
    raise ValueError(name)


def load_models(name, dev):
    dt = T.get_data()
    models = []
    for s in C.SEEDS:
        ck = T.load_ckpt(os.path.join(C.CKPT, f"{name}_s{s}.pt"))
        m = build(name).to(dev)
        m.load_state_dict(ck["state"])
        m.eval()
        models.append((ck, m))
    return dt, models


@torch.no_grad()
def ensemble_eval(models, dt, split, dev, mode, draws=5, seed0=1234):
    X, M, Lf = dt["X"][split], dt["M"][split], dt["L"][split]
    N = X["text"].shape[0]
    y_cls = dt["items"][split]["y_cls"]
    y_reg = dt["items"][split]["y_reg"]
    Ps, Rs = [], []
    for k in range(max(1, draws)):
        if mode == "clean":
            Xb = {m: X[m] for m in C.MODS}
            Mb = {m: M[m] for m in C.MODS}
        else:
            rng = np.random.default_rng(seed0 + 100 * k)
            Xb = {m: X[m] for m in C.MODS}
            Mb = {m: M[m] for m in C.MODS}
            Lb = {m: Lf[m] for m in C.MODS}
            Xb, Mb, _ = D.a3_like_inject(Xb, Mb, Lb, rng)
        pa, ra = [], []
        for ck, m in models:
            p, r, _, _ = T.predict(m, Xb, Mb, dev)
            pa.append(p)
            ra.append(r)
        Ps.append(np.mean(pa, 0))
        Rs.append(np.mean(ra, 0))
    return np.mean(Ps, 0), np.mean(Rs, 0), y_cls, y_reg


def main():
    dev = T.device_of()
    out = {}
    for name in ["final", "final_v2"]:
        dt, models = load_models(name, dev)
        # 在 valid_a3 上选 tau
        P, R, yc, yr = ensemble_eval(models, dt, "valid", dev, "a3", draws=3)
        f1, tau, tmode = T.best_tau(P, R, yc)
        res = {"tau": tau, "mode": tmode}
        for split in ["valid", "test"]:
            for mode in ["clean", "a3"]:
                P, R, yc, yr = ensemble_eval(models, dt, split, dev, mode, draws=5)
                res[f"{split}_{mode}"] = T.metrics_of(P, R, yc, yr, tau, tmode)
        out[name] = res
        print(f"== {name} (tau={tau}, {tmode}) ==", flush=True)
        for k, v in res.items():
            if isinstance(v, dict):
                print(f"  {k}: " + " ".join(f"{kk}={vv:.4f}" for kk, vv in v.items()), flush=True)
    with open(os.path.join(C.OUT, "ensemble_v1_v2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved ensemble_v1_v2.json", flush=True)


if __name__ == "__main__":
    main()
