# -*- coding: utf-8 -*-
"""模块二 最终方案 final_v2：混淆矩阵(clean/a3) + 不确定度校准，统一软集成口径。
与 figures_final_v3.py 的 fv07/fv14 及 ensemble_v1_v2.json 的 final_v2 主指标一致。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import config_q2 as C
import train_q2 as T
from eval_ensemble import load_models, ensemble_eval

dev = T.device_of()
dt, models = load_models("final_v2", dev)
tau = float(np.mean([ck["tau"] for ck, _ in models]))
tmode = "band"

# ---------- 混淆矩阵 ----------
CMs = {}
for mode, draws in [("clean", 1), ("a3", 5)]:
    P, R, yc, yr = ensemble_eval(models, dt, "valid", dev, mode, draws=draws)
    pred = T.decision(P, R, tau, tmode)
    CM = confusion_matrix(yc, pred, labels=[0, 1, 2])
    CMs[mode] = CM

lab = ["Neg", "Neu", "Pos"]
rows = []
for mode in ("clean", "a3"):
    cm = CMs[mode]
    print(f"== final_v2 混淆矩阵 [{mode}] (3种子软集成, N={int(cm.sum())}) ==")
    for i, name in enumerate(lab):
        row = cm[i]
        recall = row[i] / max(row.sum(), 1)
        print(f"  {name}: {row.tolist()}  召回={recall:.4f}")
        rows.append({"mode": mode, "true": name, "pred_Neg": int(row[0]),
                     "pred_Neu": int(row[1]), "pred_Pos": int(row[2]),
                     "recall": round(recall, 4)})
    # 预测列和
    pred_sum = cm.sum(0)
    true_sum = cm.sum(1)
    print(f"  预测列和: Neg={pred_sum[0]} Neu={pred_sum[1]} Pos={pred_sum[2]}")
    print(f"  真实行和: Neg={true_sum[0]} Neu={true_sum[1]} Pos={true_sum[2]}")
cm_df = pd.DataFrame(rows)
out_cm = os.path.join(C.OUT, "final_v2_valid_confusion.csv")
cm_df.to_csv(out_cm, index=False, encoding="utf-8-sig")
print("written:", out_cm)

# ---------- 不确定度校准（clean） ----------
Xv, Mv = dt["X"]["valid"], dt["M"]["valid"]
yv = dt["items"]["valid"]["y_reg"]
Rs, Vs = [], []
for ck, m in models:
    p, r, a, lv = T.predict(m, Xv, Mv, dev)
    Rs.append(r)
    if lv is not None:
        Vs.append(np.exp(lv))
R = np.mean(Rs, 0)
VAR = np.mean(Vs, 0) if Vs else np.zeros_like(R)
SIG = np.sqrt(np.maximum(VAR, 0.0))
err = np.abs(R - yv)
corr = float(np.corrcoef(SIG, err)[0, 1])
print(f"\n== final_v2 不确定度校准 (clean) sigma vs |error| Pearson = {corr:.4f} ==")
cal = pd.DataFrame({"true": yv, "pred": R, "sigma": SIG, "abs_err": err})
cal["bucket"] = pd.qcut(cal["sigma"], 5, labels=False)
g = cal.groupby("bucket").agg(n=("abs_err", "size"),
                              sigma_mean=("sigma", "mean"),
                              mae=("abs_err", "mean"))
print(g.round(4).to_string())
out_cal = os.path.join(C.OUT, "final_v2_uncertainty_calibration.csv")
cal.to_csv(out_cal, index=False, encoding="utf-8-sig")
print("written:", out_cal)
