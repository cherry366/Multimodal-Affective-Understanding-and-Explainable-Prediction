# -*- coding: utf-8 -*-
"""模块三 按模态分开的位置级遮挡(LOPO)忠实度检验。

目的：把原有的"位置级 top-k vs 随机"检验从"主导模态"扩展到
文本/语音/视觉三种模态分别检验，补齐语音/视觉局部证据的定量支撑。
预测与骨干均不改变，只做后验解释的忠实度评估。
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

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

    rows = []
    for i in idx:
        Xb = {m: Xv[m][i:i+1] for m in C.MODS}
        Mb = {m: Mv[m][i:i+1] for m in C.MODS}
        Pb, _, _, _ = ens_predict(models, Xb, Mb, dev)
        c = int(np.argmax(Pb[0]))
        base = float(Pb[0, c])
        occl = occlusion_importance(models, Xb, Mb, c, dev)

        for m in C.MODS:
            w = np.abs(occl[m])
            order = np.argsort(-w)
            valid = [t for t in order if Mb[m][0, t] > 0]
            eff = [t for t in range(50) if Mb[m][0, t] > 0]
            if not eff:
                continue
            topk = valid[:C.TOP_K]
            randk = list(rng.choice(eff, size=min(len(eff), C.TOP_K), replace=False))

            Xd, Md = zero_positions(Xb, Mb, m, topk)
            Pd, _, _, _ = ens_predict(models, Xd, Md, dev)
            comp_topk = base - float(Pd[0, c])

            Xr, Mr = zero_positions(Xb, Mb, m, randk)
            Pr, _, _, _ = ens_predict(models, Xr, Mr, dev)
            comp_randk = base - float(Pr[0, c])

            Xk, Mk = keep_positions(Xb, Mb, m, topk)
            Pk, _, _, _ = ens_predict(models, Xk, Mk, dev)
            suff = base - float(Pk[0, c])

            rows.append({"sample_idx": int(i), "cls": c, "modality": m,
                         "comp_topk": float(comp_topk), "comp_randk": float(comp_randk),
                         "suff_topk": float(suff)})

    df = pd.DataFrame(rows)
    summ = {"method": "occlusion_LOPO_per_modality",
            "n_samples": int(len(idx)),
            "top_k": C.TOP_K,
            "per_modality": {}}
    for m in C.MODS:
        g = df[df["modality"] == m]
        diffs = (g["comp_topk"] - g["comp_randk"]).values
        try:
            w = wilcoxon(g["comp_topk"], g["comp_randk"], alternative="greater")
            p = float(w.pvalue)
        except Exception:
            p = float("nan")
        summ["per_modality"][m] = {
            "n": int(len(g)),
            "mean_comp_topk": float(g["comp_topk"].mean()),
            "mean_comp_randk": float(g["comp_randk"].mean()),
            "mean_suff_topk": float(g["suff_topk"].mean()),
            "comp_win_rate": float((g["comp_topk"] >= g["comp_randk"]).mean()),
            "median_diff": float(np.median(diffs)),
            "wilcoxon_p_greater": p,
            "significant": bool(p < 0.05) if np.isfinite(p) else False,
        }

    df.to_csv(os.path.join(C.Q3_OUT, "occlusion_faithfulness_per_modality.csv"),
              index=False, encoding="utf-8-sig")
    with open(os.path.join(C.Q3_OUT, "occlusion_faithfulness_per_modality.json"),
              "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
