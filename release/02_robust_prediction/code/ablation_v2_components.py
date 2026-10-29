# -*- coding: utf-8 -*-
"""模块二 v2.1 组件消融：去掉 cov-prior / MoE / KD / NLL，3 种子。"""
import os
import numpy as np
import pandas as pd

import config_q2 as C
import train_q2 as T

VARIANTS = ["final_v2", "v2_nocov", "v2_nomoe", "v2_nokd", "v2_nonll"]


def main():
    rows = []
    for v in VARIANTS:
        for s in C.SEEDS:
            r = T.train_one(v, s)
            rows.append({
                "variant": v, "seed": s, "best_epoch": r["best_epoch"], "tau": r["tau"],
                "valid_a3_f1": r["valid_a3"]["f1_macro"], "valid_a3_acc": r["valid_a3"]["acc"],
                "valid_a3_mae": r["valid_a3"]["mae"], "valid_a3_pearson": r["valid_a3"]["pearson"],
                "valid_clean_f1": r["valid_clean"]["f1_macro"],
                "valid_clean_mae": r["valid_clean"]["mae"],
            })
            print(f"[{v} s{s}] a3F1={r['valid_a3']['f1_macro']:.4f} "
                  f"a3MAE={r['valid_a3']['mae']:.4f} a3r={r['valid_a3']['pearson']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT, "ablation_v2_components.csv"), index=False, encoding="utf-8-sig")
    # 汇总 mean±std
    g = df.groupby("variant")[["valid_a3_f1", "valid_a3_acc", "valid_a3_mae", "valid_a3_pearson",
                               "valid_clean_f1", "valid_clean_mae"]].agg(["mean", "std"])
    print(g.to_string(), flush=True)
    print("ablation v2 components done", flush=True)


if __name__ == "__main__":
    main()
