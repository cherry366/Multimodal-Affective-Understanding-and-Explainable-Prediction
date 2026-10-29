# -*- coding: utf-8 -*-
"""导出验证集预测明细（供绘图使用）：prob/reg/alpha/labels。"""
import os
import numpy as np
import pandas as pd
import config_q3 as C
import data_q2 as D
import train_q2 as T2
from explain_q3 import load_models, ens_predict

CLS_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}

def main():
    models, dt = load_models()
    dev = T2.device_of()
    tau = float(np.mean([ck["tau"] for ck, _ in models]))
    Xv, Mv = dt["X"]["valid"], dt["M"]["valid"]
    yc = dt["items"]["valid"]["y_cls"]
    yr = dt["items"]["valid"]["y_reg"]
    P, R, A, V = ens_predict(models, Xv, Mv, dev)
    pred = T2.decision(P, R, tau, "band")
    df = pd.DataFrame({
        "idx": np.arange(len(yc)),
        "y_cls": yc, "y_cls_name": [CLS_NAME[c] for c in yc],
        "y_reg": yr, "pred_cls": pred,
        "pred_cls_name": [CLS_NAME[c] for c in pred],
        "pred_reg": R, "prob_neg": P[:,0], "prob_neu": P[:,1], "prob_pos": P[:,2],
        "alpha_text": A[:,0], "alpha_audio": A[:,1], "alpha_vision": A[:,2],
    })
    df.to_csv(os.path.join(C.Q3_OUT, "valid_predictions.csv"), index=False, encoding="utf-8-sig")
    print("saved valid_predictions.csv", df.shape, flush=True)

if __name__ == "__main__":
    main()
