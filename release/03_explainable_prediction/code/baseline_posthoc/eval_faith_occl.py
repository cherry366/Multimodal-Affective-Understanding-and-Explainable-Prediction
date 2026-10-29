# -*- coding: utf-8 -*-
"""模块三 遮挡(LOPO)局部证据忠实度评估（替代 IG 的弱局部归因）。"""
import json
import os
import numpy as np
import pandas as pd

import config_q3 as C
import data_q2 as D
import train_q2 as T2
from explain_q3 import load_models, ens_predict, occlusion_importance


def zero_positions(Xn, Mn, m, pos):
    Xo = {k: Xn[k].copy() for k in C.MODS}
    Mo = {k: Mn[k].copy() for k in C.MODS}
    if len(pos):
        Xo[m][0, pos, :] = 0.0
        Mo[m][0, pos] = 0.0
    return Xo, Mo


def keep_positions(Xn, Mn, m, pos):
    Xo = {k: Xn[k].copy() for k in C.MODS}
    Mo = {k: Mn[k].copy() for k in C.MODS}
    keep = np.zeros(50, dtype=bool)
    keep[list(pos)] = True
    for t in range(50):
        if not keep[t]:
            Xo[m][0, t, :] = 0.0
            Mo[m][0, t] = 0.0
    return Xo, Mo


def main():
    models, dt = load_models()
    dev = T2.device_of()
    Xv = dt["X"]["valid"]
    Mv = dt["M"]["valid"]
    yc = dt["items"]["valid"]["y_cls"]

    # 与 eval_q3 相同的 200 条分层样本
    rng = np.random.default_rng(2026)
    idx = []
    for c in (0, 1, 2):
        cand = np.where(yc == c)[0]
        n = int(round(200 * len(cand) / len(yc)))
        idx.extend(rng.choice(cand, size=max(n, 1), replace=False).tolist())
    idx = np.array(idx[:200])
    use = idx[:120]

    rows = []
    for i in use:
        Xb = {m: Xv[m][i:i+1] for m in C.MODS}
        Mb = {m: Mv[m][i:i+1] for m in C.MODS}
        Pb, Rb, Ab, _ = ens_predict(models, Xb, Mb, dev)
        c = int(np.argmax(Pb[0]))
        base = float(Pb[0, c])
        occl = occlusion_importance(models, Xb, Mb, c, dev)
        # 主导模态 = |Δlogit| 累计最大的模态
        dom = max(C.MODS, key=lambda m: np.abs(occl[m]).sum())
        w = np.abs(occl[dom])
        order = np.argsort(-w)
        valid = [t for t in order if Mb[dom][0, t] > 0]
        topk = valid[:C.TOP_K]
        eff = [t for t in range(50) if Mb[dom][0, t] > 0]
        randk = list(rng.choice(eff, size=min(len(eff), C.TOP_K), replace=False)) if len(eff) else []

        Xd, Md = zero_positions(Xb, Mb, dom, topk)
        Pd, _, _, _ = ens_predict(models, Xd, Md, dev)
        comp_topk = base - float(Pd[0, c])

        Xr, Mr = zero_positions(Xb, Mb, dom, randk)
        Pr, _, _, _ = ens_predict(models, Xr, Mr, dev)
        comp_randk = base - float(Pr[0, c])

        Xk, Mk = keep_positions(Xb, Mb, dom, topk)
        Pk, _, _, _ = ens_predict(models, Xk, Mk, dev)
        suff = base - float(Pk[0, c])

        rows.append({"sample_idx": int(i), "cls": c, "modality": dom,
                     "comp_topk": float(comp_topk), "comp_randk": float(comp_randk),
                     "suff_topk": float(suff)})

    df = pd.DataFrame(rows)
    summ = {
        "method": "occlusion_LOPO",
        "n_samples": int(len(df)),
        "top_k": C.TOP_K,
        "mean_comp_topk": float(df["comp_topk"].mean()),
        "mean_comp_randk": float(df["comp_randk"].mean()),
        "mean_suff_topk": float(df["suff_topk"].mean()),
        "comp_win_rate": float((df["comp_topk"] >= df["comp_randk"]).mean()),
        "mean_comp_topk_minus_randk": float((df["comp_topk"] - df["comp_randk"]).mean()),
    }
    df.to_csv(os.path.join(C.Q3_OUT, "occlusion_faithfulness.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(C.Q3_OUT, "occlusion_faithfulness.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
