# -*- coding: utf-8 -*-
"""模块二 最终方案 v2.1（cov-prior 门控）对缺失测试集全量推理。"""
import json
import os
import numpy as np
import pandas as pd
import torch

import config_q2 as C
import data_q2 as D
import train_q2 as T
from model_q2 import Q2Net

CLS_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}


def load_v2_ensemble(dev):
    models, taus, seeds = [], [], []
    for s in C.SEEDS:
        p = os.path.join(C.CKPT, f"final_v2_s{s}.pt")
        if not os.path.exists(p):
            continue
        ck = T.load_ckpt(p)
        m = Q2Net(use_key_mask=True, use_gate=True, use_cross=True, use_recon=True,
                  use_moe=True, use_nll=True, use_cov_prior=True,
                  use_moe_feats=True, fusion_dropout=C.FUSION_DROPOUT).to(dev)
        m.load_state_dict(ck["state"])
        m.eval()
        models.append(m)
        taus.append(ck["tau"])
        seeds.append(s)
    return models, taus, seeds


def main():
    dt = T.get_data()
    dev = T.device_of()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import BertModel
    bert = BertModel.from_pretrained("bert-base-uncased").to(dev).eval()
    a3 = D.load_a3_aligned(bert=bert, device=dev)
    stats = dt["stats"]
    X3 = {m: D.apply_norm(a3[m], a3["mask"][m], stats[m][0], stats[m][1])
          for m in C.MODS}
    M3 = a3["mask"]

    models, taus, seeds = load_v2_ensemble(dev)
    print("ensemble seeds:", seeds, "taus:", taus, flush=True)
    Ps, Rs, As, Vars = [], [], [], []
    for m in models:
        p, r, a, lv = T.predict(m, X3, M3, dev)
        Ps.append(p)
        Rs.append(r)
        As.append(a)
        if lv is not None:
            Vars.append(np.exp(lv))
    P = np.mean(Ps, 0)
    R = np.mean(Rs, 0)
    A = np.mean(As, 0)
    VAR = np.mean(Vars, 0) if Vars else np.zeros_like(R)
    SIG = np.sqrt(np.maximum(VAR, 0.0))
    tau = float(np.mean(taus)) if taus else 0.0
    cls_band = T.decision(P, R, tau, "band")
    cls_am = P.argmax(1)
    entropy = -(P * np.log(P + 1e-9)).sum(1)

    rows = []
    n_missing_all = 0
    for i, sid in enumerate(a3["id"]):
        per_mod = {}
        for m in C.MODS:
            st = D.missing_stats(M3[m][i:i + 1], a3["leff"][m][i:i + 1])[0]
            per_mod[m] = st
        n_miss = int(max(per_mod[m][0] for m in C.MODS))
        ratio = float(max(per_mod[m][2] for m in C.MODS))
        n_seg = int(max(per_mod[m][1] for m in C.MODS))
        miss_mods = [m for m in C.MODS if per_mod[m][0] > 0]
        if n_miss > 0:
            n_missing_all += 1
        rows.append({
            "id": sid,
            "predicted_class": int(cls_band[i]),
            "predicted_class_name": CLS_NAME[int(cls_band[i])],
            "predicted_score": round(float(R[i]), 4),
            "score_uncertainty_sigma": round(float(SIG[i]), 4),
            "classification_entropy": round(float(entropy[i]), 4),
            "prob_neg": round(float(P[i, 0]), 4),
            "prob_neu": round(float(P[i, 1]), 4),
            "prob_pos": round(float(P[i, 2]), 4),
            "predicted_class_argmax": int(cls_am[i]),
            "dominant_modality": C.MODS[int(np.argmax(A[i]))],
            "alpha_text": round(float(A[i, 0]), 4),
            "alpha_audio": round(float(A[i, 1]), 4),
            "alpha_vision": round(float(A[i, 2]), 4),
            "n_missing_bins": n_miss,
            "n_missing_segments": n_seg,
            "missing_ratio_in_valid": round(ratio, 4),
            "missing_modalities": ",".join(miss_mods) if miss_mods else "none",
        })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(C.OUT, "missing_modality_predictions.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("written", out_csv, df.shape, flush=True)

    summary = {
        "n_samples": int(len(df)),
        "ensemble_seeds": [int(s) for s in seeds],
        "tau": tau,
        "band_vs_argmax_agreement": float((cls_band == cls_am).mean()),
        "class_distribution": {CLS_NAME[k]: int(v)
                               for k, v in df["predicted_class"].value_counts().items()},
        "score_mean": float(df["predicted_score"].mean()),
        "score_std": float(df["predicted_score"].std()),
        "uncertainty_sigma_mean": float(df["score_uncertainty_sigma"].mean()),
        "entropy_mean": float(df["classification_entropy"].mean()),
        "n_samples_with_missing": n_missing_all,
        "mean_missing_bins": float(df["n_missing_bins"].mean()),
        "dominant_modality_distribution": df["dominant_modality"].value_counts().to_dict(),
    }
    with open(os.path.join(C.OUT, "infer_final_v2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
