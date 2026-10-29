# -*- coding: utf-8 -*-
"""模块二 连续 vs 分散缺失对照（final_v2，验证集，3 种子集成）

同样缺失格数（r × 有效长度）下，比较:
  continuous : 单段连续缺失（K=1，segmax=整段）
  dispersed  : 多段短缺失（K=6，segmax=4）
输出: q2_output\ablation_contig_dispersed_v2.csv
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
        m = build_v2(dev)
        m.load_state_dict(ck["state"])
        m.eval()
        models.append((ck, m))
    return dt, models


@torch.no_grad()
def eval_pattern(models, dt, dev, r, pattern, draws=10, seed0=2024):
    X, M, Lf = dt["X"]["valid"], dt["M"]["valid"], dt["L"]["valid"]
    idx = np.arange(X["text"].shape[0])
    y_cls = dt["items"]["valid"]["y_cls"][idx]
    y_reg = dt["items"]["valid"]["y_reg"][idx]
    Ps, Rs = [], []
    for k in range(draws):
        rng = np.random.default_rng(seed0 + 61 * k)
        Xb = {m: X[m][idx] for m in C.MODS}
        Mb = {m: M[m][idx] for m in C.MODS}
        Lb = {m: Lf[m][idx] for m in C.MODS}
        if pattern == "continuous":
            Xb, Mb, _ = D.inject(Xb, Mb, Lb, rng, modes=list(C.MODS), r=r,
                                 K=1, segmax=100, indep=False, clean_ratio=0.0)
        elif pattern == "dispersed":
            Xb, Mb, _ = D.inject(Xb, Mb, Lb, rng, modes=list(C.MODS), r=r,
                                 K=C.CVD_SEG_COUNT, segmax=C.CVD_SEG_MAX,
                                 indep=False, clean_ratio=0.0)
        else:
            raise ValueError(pattern)
        pa, ra = [], []
        for ck, m in models:
            p, r_, _, _ = T.predict(m, Xb, Mb, dev)
            pa.append(p)
            ra.append(r_)
        Ps.append(np.mean(pa, 0))
        Rs.append(np.mean(ra, 0))
    return Ps, Rs, y_cls, y_reg


def main():
    dev = T.device_of()
    dt, models = load_v2(dev)
    tau = float(np.mean([ck["tau"] for ck, _ in models]))
    print("final_v2 ensemble loaded, tau=", tau, flush=True)

    rows = []
    for pattern, label in [("continuous", "连续单段"), ("dispersed", "多段分散")]:
        for r in C.CVD_RATES:
            Ps, Rs, yc, yr = eval_pattern(models, dt, dev, r, pattern)
            ms = [T.metrics_of(P, R, yc, yr, tau, "band") for P, R in zip(Ps, Rs)]
            row = {"pattern": pattern, "label": label, "rate": r, "tau": tau}
            colmap = {"f1_macro": "f1", "acc": "acc", "mae": "mae", "pearson": "pearson"}
            for ksrc, kdst in colmap.items():
                vals = [m[ksrc] for m in ms]
                row[kdst] = float(np.mean(vals))
                row[kdst + "_std"] = float(np.std(vals))
            rows.append(row)
            print(f"[{label} r={r:.2f}] F1={row['f1']:.4f}±{row['f1_std']:.4f} "
                  f"Acc={row['acc']:.4f} MAE={row['mae']:.4f} Pearson={row['pearson']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "ablation_contig_dispersed_v2.csv"),
                              index=False, encoding="utf-8-sig")
    print("contig vs dispersed done", flush=True)


if __name__ == "__main__":
    main()
