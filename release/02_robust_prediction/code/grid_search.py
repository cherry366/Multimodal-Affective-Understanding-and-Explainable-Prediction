# -*- coding: utf-8 -*-
"""模块二 λ 网格搜索（坐标下降，验证集选超参）

用法:
    python grid_search.py            # 默认 seed=42, epochs=15
    python grid_search.py --seeds 42,2024,7 --epochs 40
选择指标: score = F1_macro(valid_a3) + 0.2 * max(0, Pearson(valid_a3))
输出:   q2_output\grid_search_raw.csv / grid_search_summary.json
"""
import argparse
import json
import os
import numpy as np
import pandas as pd

import config_q2 as C
import train_q2 as T


def run_cfg(cfg, seeds):
    rows = []
    for s in seeds:
        r = T.train_one("final_v2", s, overrides=cfg)
        rows.append({
            "variant": r["variant"], "seed": r["seed"], "best_epoch": r["best_epoch"],
            "tau": r["tau"], "f1": r["valid_a3"]["f1_macro"],
            "acc": r["valid_a3"]["acc"], "mae": r["valid_a3"]["mae"],
            "pearson": r["valid_a3"]["pearson"],
            "f1_clean": r["valid_clean"]["f1_macro"],
        })
    return rows


def score_of(rows):
    f1 = float(np.mean([r["f1"] for r in rows]))
    p = float(np.mean([r["pearson"] for r in rows]))
    mae = float(np.mean([r["mae"] for r in rows]))
    return f1 + 0.2 * max(0.0, p) - C.SCORE_MAE_W * mae, f1, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--stage", default="all", choices=["all", "kd", "nll", "reg", "rec"])
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    stage = {"kd": C.LAMBDA_KD, "ctr": C.LAMBDA_CTR, "nll": C.LAMBDA_NLL,
             "reg": C.LAMBDA_REG, "rec": C.MU_RECON}
    all_rows = []
    summary = {"metric": "F1_a3 + 0.2*Pearson_a3 - 0.3*MAE_a3", "seeds": seeds, "epochs": args.epochs, "stages": {}}

    stages = []
    if args.stage in ("all", "kd"):
        stages.append(("kd", C.GRID_KD))
    if args.stage in ("all", "nll"):
        stages.append(("nll", C.GRID_NLL))
    if args.stage in ("all", "reg"):
        stages.append(("reg", C.GRID_REG))
    if args.stage in ("all", "rec"):
        stages.append(("rec", C.GRID_REC))

    for key, grid in stages:
        cands = [dict(stage, **{key: v}) for v in grid]
        best_sc, best_cfg = -9.0, None
        for cfg in cands:
            rows = run_cfg(cfg, seeds)
            for r in rows:
                r["param"] = key
                r["value"] = cfg[key] if key != "regrec" else f"reg{cfg['reg']}_rec{cfg['rec']}"
            all_rows += rows
            sc, f1, p = score_of(rows)
            print(f"[{key}] {r['value']}  score={sc:.4f} f1={f1:.4f} pearson={p:.4f}", flush=True)
            if sc > best_sc:
                best_sc, best_cfg = sc, cfg
        stage.update(best_cfg)
        summary["stages"][key] = {"best_cfg": best_cfg, "score": best_sc}
        print(f"== stage {key} 选择 {best_cfg} (score {best_sc:.4f}) ==", flush=True)

    summary["final_cfg"] = stage
    pd.DataFrame(all_rows).to_csv(os.path.join(C.OUT, "grid_search_raw.csv"),
                                  index=False, encoding="utf-8-sig")
    with open(os.path.join(C.OUT, "grid_search_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("grid search done; final_cfg=", stage, flush=True)


if __name__ == "__main__":
    main()
