# -*- coding: utf-8 -*-
"""模块二 消融实验：缺失模态类型 / 缺失位置 / 缺失率 + 组件消融

用法: python ablation_q2.py
输出: q2_output\ablation_type.csv / ablation_position.csv / ablation_rate.csv / ablation_component.csv
"""
import os
import numpy as np
import pandas as pd
import torch

import config_q2 as C
import data_q2 as D
import train_q2 as T
from model_q2 import Q2Net


def build_model(spec):
    return Q2Net(use_key_mask=spec["use_key_mask"], use_gate=spec["use_gate"],
                 use_cross=True, use_recon=spec["use_recon"])


def load_all_ckpt():
    dt, dev = T.get_data(), T.device_of()
    cks = {}
    for f in sorted(os.listdir(C.CKPT)):
        if not f.endswith(".pt"):
            continue
        ck = T.load_ckpt(os.path.join(C.CKPT, f))
        model = build_model(ck["spec"]).to(dev)
        model.load_state_dict(ck["state"])
        model.eval()
        cks[(ck["variant"], ck["seed"])] = (ck, model)
    return dt, dev, cks


def apply_cond(Xb, Mb, Lb, cond, rng):
    kind = cond["kind"]
    if kind == "clean":
        return Xb, Mb
    if kind == "a3":
        Xn, Mn, _ = D.a3_like_inject(Xb, Mb, Lb, rng)
        return Xn, Mn
    Xn, Mn, _ = D.inject(Xb, Mb, Lb, rng, modes=cond.get("modes"), r=cond.get("r"),
                         K=cond.get("K"), segmax=cond.get("segmax", C.AUG_SEG_MAX),
                         position=cond.get("position"), indep=cond.get("indep", False),
                         clean_ratio=0.0, scatter=cond.get("scatter", False))
    return Xn, Mn


@torch.no_grad()
def eval_cond(model, dt, split, dev, cond, draws=3, seed0=777, idx=None):
    X, M, Lf = dt["X"][split], dt["M"][split], dt["L"][split]
    idx = np.arange(X["text"].shape[0]) if idx is None else np.asarray(idx)
    y_cls = dt["items"][split]["y_cls"][idx]
    y_reg = dt["items"][split]["y_reg"][idx]
    Ps, Rs = [], []
    for k in range(draws):
        rng = np.random.default_rng(seed0 + 97 * k)
        Xb = {m: X[m][idx] for m in C.MODS}
        Mb = {m: M[m][idx] for m in C.MODS}
        Lb = {m: Lf[m][idx] for m in C.MODS}
        Xb, Mb = apply_cond(Xb, Mb, Lb, cond, rng)
        p, r, _ = T.predict(model, Xb, Mb, dev)
        Ps.append(p)
        Rs.append(r)
    return np.mean(Ps, 0), np.mean(Rs, 0), y_cls, y_reg


CORPU = ["acc", "f1_macro", "f1_weighted", "mae", "pearson"]


def run_grid(cks, dt, dev, grid, splits=("valid",), draws=3, tag=""):
    rows = []
    for cond in grid:
        for (v, s), (ck, model) in cks.items():
            for split in splits:
                P, R, yc, yr = eval_cond(model, dt, split, dev, cond, draws=draws)
                mt = T.metrics_of(P, R, yc, yr, ck["tau"], ck["tau_mode"])
                row = {"family": tag, "condition": cond["name"], "variant": v, "seed": s,
                       "split": split, "tau": ck["tau"]}
                row.update(mt)
                rows.append(row)
        print(f"[{tag}] {cond['name']} done", flush=True)
    return pd.DataFrame(rows)


def main():
    dt, dev, cks = load_all_ckpt()
    print("loaded ckpts:", sorted(cks.keys()), flush=True)

    # ---- A. 缺失模态类型（r=0.2 固定）----
    sets = {"无缺失": None, "仅文本": ["text"], "仅语音": ["audio"], "仅视觉": ["vision"],
            "文本+语音": ["text", "audio"], "文本+视觉": ["text", "vision"],
            "语音+视觉": ["audio", "vision"], "三模态": ["text", "audio", "vision"]}
    gridA = []
    for name, mods in sets.items():
        if mods is None:
            gridA.append({"kind": "clean", "name": name})
        else:
            gridA.append({"kind": "type", "name": name, "modes": mods, "r": 0.2})
    dfA = run_grid(cks, dt, dev, gridA, splits=("valid", "test"), draws=3, tag="type")
    dfA.to_csv(os.path.join(C.OUT, "ablation_type.csv"), index=False, encoding="utf-8-sig")

    # ---- B. 缺失位置（r=0.3 单段）----
    gridB = []
    for mname, mods in [("文本", ["text"]), ("语音", ["audio"]), ("视觉", ["vision"]),
                        ("三模态", ["text", "audio", "vision"])]:
        for pos, pname in [("front", "前段"), ("middle", "中段"), ("back", "后段")]:
            gridB.append({"kind": "pos", "name": f"{mname}-{pname}", "modes": mods,
                          "r": 0.3, "position": pos})
    dfB = run_grid(cks, dt, dev, gridB, splits=("valid", "test"), draws=3, tag="position")
    dfB.to_csv(os.path.join(C.OUT, "ablation_position.csv"), index=False, encoding="utf-8-sig")

    # ---- C. 缺失率（模态集合按训练同分布随机）----
    gridC = [{"kind": "rate", "name": "无缺失（r=0）", "modes": None, "r": 0.0}]
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        gridC.append({"kind": "rate", "name": f"r={int(r*100)}%", "modes": None, "r": r})
    dfC = run_grid(cks, dt, dev, gridC, splits=("valid", "test"), draws=3, tag="rate")
    dfC.to_csv(os.path.join(C.OUT, "ablation_rate.csv"), index=False, encoding="utf-8-sig")

    # ---- D. 组件消融（统一条件下的汇总）----
    conds = [{"kind": "clean", "name": "无缺失"},
             {"kind": "a3", "name": "缺失测试集同分布(r≈19.5%)"},
             {"kind": "type", "name": "三模态r=20%", "modes": C.MODS, "r": 0.2},
             {"kind": "type", "name": "仅文本r=20%", "modes": ["text"], "r": 0.2},
             {"kind": "type", "name": "仅语音r=20%", "modes": ["audio"], "r": 0.2},
             {"kind": "type", "name": "仅视觉r=20%", "modes": ["vision"], "r": 0.2}]
    dfD = run_grid(cks, dt, dev, conds, splits=("valid", "test"), draws=3, tag="component")
    dfD.to_csv(os.path.join(C.OUT, "ablation_component.csv"), index=False, encoding="utf-8-sig")

    # ---- E. 极端与"未见形态"缺失（分布外压力测试）----
    gridE = []
    for name, mods in [("仅文本全缺失", ["text"]), ("仅语音全缺失", ["audio"]),
                       ("仅视觉全缺失", ["vision"])]:
        gridE.append({"kind": "type", "name": name, "modes": mods, "r": 1.0})
    gridE += [{"kind": "type", "name": "三模态各50%", "modes": list(C.MODS), "r": 0.5},
              {"kind": "type", "name": "三模态各70%", "modes": list(C.MODS), "r": 0.7},
              {"kind": "type", "name": "散点缺失20%", "modes": list(C.MODS), "r": 0.2, "scatter": True},
              {"kind": "type", "name": "散点缺失40%", "modes": list(C.MODS), "r": 0.4, "scatter": True}]
    dfE = run_grid(cks, dt, dev, gridE, splits=("valid", "test"), draws=3, tag="extreme")
    dfE.to_csv(os.path.join(C.OUT, "ablation_extreme.csv"), index=False, encoding="utf-8-sig")
    print("ablation done", flush=True)


if __name__ == "__main__":
    main()