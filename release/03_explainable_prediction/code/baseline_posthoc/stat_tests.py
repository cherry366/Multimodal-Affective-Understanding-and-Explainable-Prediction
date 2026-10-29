# -*- coding: utf-8 -*-
"""模块三 可解释性忠实度的统计显著性检验（Wilcoxon 符号秩 + Spearman p 值）。"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

import config_q3 as C

OUT = C.Q3_OUT


def main():
    oc = pd.read_csv(os.path.join(OUT, "occlusion_faithfulness.csv"))
    fs = pd.read_csv(os.path.join(OUT, "faithfulness_samples.csv"))

    out = {}

    # 1) 遮挡 top-k vs 随机（位置级）
    w_topk = wilcoxon(oc["comp_topk"], oc["comp_randk"], alternative="greater")
    out["occlusion_topk_vs_rand"] = {
        "n": int(len(oc)),
        "median_diff": float(np.median(oc["comp_topk"] - oc["comp_randk"])),
        "statistic": float(w_topk.statistic),
        "p_value": float(w_topk.pvalue),
        "significant": bool(w_topk.pvalue < 0.05),
    }

    # 2) 主导模态 vs 随机模态（模态级）
    w_dom = wilcoxon(fs["comp_dom"], fs["comp_rand"], alternative="greater")
    out["lomo_dom_vs_rand"] = {
        "n": int(len(fs)),
        "median_diff": float(np.median(fs["comp_dom"] - fs["comp_rand"])),
        "statistic": float(w_dom.statistic),
        "p_value": float(w_dom.pvalue),
        "significant": bool(w_dom.pvalue < 0.05),
    }

    # 3) 门控 α 与 Shapley φ 的 Spearman 相关性（含 p 值）
    out["spearman_alpha_phi"] = {}
    for m in C.MODS:
        rho, p = spearmanr(fs[f"alpha_{m}"], fs[f"phi_{m}"])
        out["spearman_alpha_phi"][m] = {"rho": float(rho), "p_value": float(p)}

    with open(os.path.join(OUT, "stat_tests.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
