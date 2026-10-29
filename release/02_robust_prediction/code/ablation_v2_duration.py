# -*- coding: utf-8 -*-
"""缺失时长专项（v2.1）：连续缺失段长度对性能的影响
输出: ablation_duration_v2.csv
"""
import os
import numpy as np
import pandas as pd
import torch

import config_q2 as C
import data_q2 as D
import train_q2 as T
from model_q2 import Q2Net


def build_v2(dev):
    return Q2Net(use_key_mask=True, use_gate=True, use_cross=True, use_recon=True,
                 use_moe=True, use_nll=True, use_cov_prior=True,
                 use_moe_feats=True, fusion_dropout=C.FUSION_DROPOUT).to(dev)


def load_v2(dev):
    dt = T.get_data()
    models = []
    for s in C.SEEDS:
        ck = T.load_ckpt(os.path.join(C.CKPT, f"final_v2_s{s}.pt"))
        mdl = build_v2(dev)
        mdl.load_state_dict(ck["state"])
        mdl.eval()
        models.append(mdl)
    return dt, models


@torch.no_grad()
def eval_duration(models, dt, dev, L, mods, pos="middle", draws=3, seed0=555):
    X, M, Lf = dt["X"]["valid"], dt["M"]["valid"], dt["L"]["valid"]
    idx = np.arange(X["text"].shape[0])
    yc = dt["items"]["valid"]["y_cls"][idx]
    yr = dt["items"]["valid"]["y_reg"][idx]
    Ps, Rs = [], []
    for k in range(draws):
        Xb = {m: X[m][idx].copy() for m in C.MODS}
        Mb = {m: M[m][idx].copy() for m in C.MODS}
        for m in mods:
            le_all = Lf[m][idx]
            for b in range(len(idx)):
                le = int(le_all[b])
                lo = 1
                hi = le - 1 if m == "text" else le
                span = hi - lo
                if span <= 1:
                    continue
                ln = min(int(L), span)
                if pos == "front":
                    s = lo
                elif pos == "back":
                    s = hi - ln
                else:
                    s = lo + (span - ln) // 2
                Mb[m][b, s:s + ln] = 0.0
                Xb[m][b, s:s + ln, :] = 0.0
        pa, ra = [], []
        for mdl in models:
            p, r, _, _ = T.predict(mdl, Xb, Mb, dev)
            pa.append(p)
            ra.append(r)
        Ps.append(np.mean(pa, 0))
        Rs.append(np.mean(ra, 0))
    return np.mean(Ps, 0), np.mean(Rs, 0), yc, yr


def main():
    dev = T.device_of()
    dt, models = load_v2(dev)
    taus = [T.load_ckpt(os.path.join(C.CKPT, f"final_v2_s{s}.pt"))["tau"] for s in C.SEEDS]
    tau = float(np.mean(taus))
    Ls = [1, 2, 4, 6, 8, 12, 16, 24, 32, 40]
    groups = [("三模态", list(C.MODS)), ("仅文本", ["text"]),
              ("仅语音", ["audio"]), ("仅视觉", ["vision"])]
    rows = []
    for gname, mods in groups:
        for L in Ls:
            P, R, yc, yr = eval_duration(models, dt, dev, L, mods, pos="middle")
            mt = T.metrics_of(P, R, yc, yr, tau, "band")
            row = {"group": gname, "duration": L, "split": "valid", "tau": tau}
            row.update(mt)
            rows.append(row)
            print(f"[{gname}] L={L:2d} F1={mt['f1_macro']:.4f} Acc={mt['acc']:.4f} "
                  f"MAE={mt['mae']:.4f} r={mt['pearson']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "ablation_duration_v2.csv"),
                                  index=False, encoding="utf-8-sig")
    print("v2 duration ablation done", flush=True)


if __name__ == "__main__":
    main()
