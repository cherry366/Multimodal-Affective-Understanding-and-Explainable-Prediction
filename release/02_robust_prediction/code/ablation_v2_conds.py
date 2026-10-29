# -*- coding: utf-8 -*-
"""最终方案 v2.1 在验证集上的缺失消融评估：类型 / 位置 / 缺失率 / 极端形态
只跑 final_v2 3 种子集成，输出 ablation_*_v2.csv，供第二问最终图集使用。
"""
import os
import numpy as np
import pandas as pd
import torch

import config_q2 as C
import data_q2 as D
import train_q2 as T
from model_q2 import Q2Net
from ablation_q2 import apply_cond


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
def eval_cond(models, dt, dev, cond, draws=10, seed0=777):
    X, M, Lf = dt["X"]["valid"], dt["M"]["valid"], dt["L"]["valid"]
    idx = np.arange(X["text"].shape[0])
    y_cls = dt["items"]["valid"]["y_cls"][idx]
    y_reg = dt["items"]["valid"]["y_reg"][idx]
    Ps, Rs = [], []
    for k in range(draws):
        rng = np.random.default_rng(seed0 + 97 * k)
        Xb = {m: X[m][idx] for m in C.MODS}
        Mb = {m: M[m][idx] for m in C.MODS}
        Lb = {m: Lf[m][idx] for m in C.MODS}
        Xb, Mb = apply_cond(Xb, Mb, Lb, cond, rng)
        pa, ra = [], []
        for ck, m in models:
            p, r, _, _ = T.predict(m, Xb, Mb, dev)
            pa.append(p)
            ra.append(r)
        Ps.append(np.mean(pa, 0))
        Rs.append(np.mean(ra, 0))
    return Ps, Rs, y_cls, y_reg


def run_grid(models, dt, dev, grid, tag):
    rows = []
    tau = float(np.mean([ck["tau"] for ck, _ in models]))
    for cond in grid:
        Ps, Rs, yc, yr = eval_cond(models, dt, dev, cond)
        ms = [T.metrics_of(P, R, yc, yr, tau, "band") for P, R in zip(Ps, Rs)]
        row = {"family": tag, "condition": cond["name"], "split": "valid", "tau": tau}
        std_name = {"acc": "acc_std", "f1_macro": "f1_std", "mae": "mae_std", "pearson": "pearson_std"}
        for key in ("acc", "f1_macro", "mae", "pearson"):
            vals = [m[key] for m in ms]
            row[key] = float(np.mean(vals))
            row[std_name[key]] = float(np.std(vals))
        rows.append(row)
        print(f"[{tag}] {cond['name']:16s} F1={row['f1_macro']:.4f}±{row['f1_std']:.4f} "
              f"Acc={row['acc']:.4f} MAE={row['mae']:.4f} r={row['pearson']:.4f}", flush=True)
    return pd.DataFrame(rows)


def main():
    dev = T.device_of()
    dt, models = load_v2(dev)
    print("final_v2 ensemble loaded, seeds:", C.SEEDS, flush=True)

    # A. 缺失模态类型（r=0.2 固定）
    sets = {"无缺失": None, "仅文本": ["text"], "仅语音": ["audio"], "仅视觉": ["vision"],
            "文本+语音": ["text", "audio"], "文本+视觉": ["text", "vision"],
            "语音+视觉": ["audio", "vision"], "三模态": ["text", "audio", "vision"]}
    gridA = [{"kind": "clean", "name": n} if mods is None
             else {"kind": "type", "name": n, "modes": mods, "r": 0.2}
             for n, mods in sets.items()]
    run_grid(models, dt, dev, gridA, "type").to_csv(
        os.path.join(C.OUT, "ablation_type_v2.csv"), index=False, encoding="utf-8-sig")

    # B. 缺失位置（r=0.3 单段）
    gridB = [{"kind": "pos", "name": f"{mn}-{pn}", "modes": mods, "r": 0.3, "position": pos}
             for mn, mods in [("文本", ["text"]), ("语音", ["audio"]),
                              ("视觉", ["vision"]), ("三模态", ["text", "audio", "vision"])]
             for pos, pn in [("front", "前段"), ("middle", "中段"), ("back", "后段")]]
    run_grid(models, dt, dev, gridB, "position").to_csv(
        os.path.join(C.OUT, "ablation_position_v2.csv"), index=False, encoding="utf-8-sig")

    # C. 缺失率
    gridC = [{"kind": "rate", "name": "无缺失（r=0）", "modes": None, "r": 0.0}]
    gridC += [{"kind": "rate", "name": f"r={int(r*100)}%", "modes": None, "r": r}
              for r in [0.1, 0.2, 0.3, 0.4, 0.5]]
    run_grid(models, dt, dev, gridC, "rate").to_csv(
        os.path.join(C.OUT, "ablation_rate_v2.csv"), index=False, encoding="utf-8-sig")

    # D. 极端缺失
    gridE = [{"kind": "type", "name": n, "modes": mods, "r": 1.0}
             for n, mods in [("仅文本全缺失", ["text"]), ("仅语音全缺失", ["audio"]),
                             ("仅视觉全缺失", ["vision"])]]
    gridE += [{"kind": "type", "name": "三模态各50%", "modes": list(C.MODS), "r": 0.5},
              {"kind": "type", "name": "三模态各70%", "modes": list(C.MODS), "r": 0.7},
              {"kind": "type", "name": "散点缺失20%", "modes": list(C.MODS), "r": 0.2, "scatter": True},
              {"kind": "type", "name": "散点缺失40%", "modes": list(C.MODS), "r": 0.4, "scatter": True}]
    run_grid(models, dt, dev, gridE, "extreme").to_csv(
        os.path.join(C.OUT, "ablation_extreme_v2.csv"), index=False, encoding="utf-8-sig")
    print("v2 condition ablation done", flush=True)


if __name__ == "__main__":
    main()
