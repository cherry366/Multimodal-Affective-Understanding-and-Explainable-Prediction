# -*- coding: utf-8 -*-
"""模块三 可解释测试集推理：预测 + 解释（Shapley 模态贡献 + 遮挡证据 + 关键帧抽取）。"""
import json
import os
import numpy as np
import pandas as pd

import config_q3 as C
import data_q3 as D3
import data_q2 as D
import train_q2 as T2
from explain_q3 import (load_models, ens_predict, shapley_values, occlusion_importance,
                        topk_positions, text_evidence, audio_evidence, vision_evidence)

CLS_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}


def main():
    models, dt = load_models()
    dev = T2.device_of()
    stats = T2._C["stats"]
    tau = float(np.mean([ck["tau"] for ck, _ in models]))
    print("models loaded, tau =", tau, flush=True)

    a4 = D3.load_all_a4()
    N = a4["text"].shape[0]
    Xn = {m: D.apply_norm(a4[m], a4["mask"][m], *stats[m]) for m in C.MODS}
    Mn = a4["mask"]

    rows, expl = [], {}
    for b in range(N):
        Xb = {m: Xn[m][b:b+1] for m in C.MODS}
        Mb = {m: Mn[m][b:b+1] for m in C.MODS}
        P, R, A, V = ens_predict(models, Xb, Mb, dev)
        cls = int(T2.decision(P, R, tau, "band")[0])
        cls_am = int(np.argmax(P[0]))
        name = CLS_NAME[cls]
        entropy = float(-(P[0] * np.log(P[0] + 1e-9)).sum())

        # Shapley（对预测类 c 归因）
        c = int(np.argmax(P[0]))
        phi, pi = shapley_values(models, Xb, Mb, c, dev)
        dom = max(C.MODS, key=lambda m: phi[m])

        # 遮挡 (LOPO) 局部证据 + IG 补充
        occl = occlusion_importance(models, Xb, Mb, c, dev)
        imp = {m: np.abs(occl[m]) for m in C.MODS}      # 用 |Δlogit| 排序
        top = {m: topk_positions(imp[m], C.TOP_K, Mn[m][b]) for m in C.MODS}
        ev_text = text_evidence(a4["raw_text"][b], top["text"])
        ev_audio = audio_evidence(b + 1, top["audio"])
        ev_vision = vision_evidence(b + 1, top["vision"])

        sigma = float(np.sqrt(max(float(V[0]), 0.0))) if V is not None else None
        row = {
            "id": a4["id"][b],
            "predicted_class": cls,
            "predicted_class_name": name,
            "predicted_score": round(float(R[0]), 4),
            "score_uncertainty_sigma": round(sigma, 4) if sigma is not None else "",
            "classification_entropy": round(entropy, 4),
            "prob_neg": round(float(P[0, 0]), 4),
            "prob_neu": round(float(P[0, 1]), 4),
            "prob_pos": round(float(P[0, 2]), 4),
            "predicted_class_argmax": cls_am,
            "dominant_modality": dom,
            "alpha_text": round(float(A[0, 0]), 4),
            "alpha_audio": round(float(A[0, 1]), 4),
            "alpha_vision": round(float(A[0, 2]), 4),
            "shapley_text": round(float(phi["text"]), 4),
            "shapley_audio": round(float(phi["audio"]), 4),
            "shapley_vision": round(float(phi["vision"]), 4),
            "pi_text": round(float(pi["text"]), 4),
            "pi_audio": round(float(pi["audio"]), 4),
            "pi_vision": round(float(pi["vision"]), 4),
            "evidence_text": " ".join(ev_text["words"]),
            "evidence_audio_windows": ";".join([f"{w[0]:.2f}-{w[1]:.2f}s" for w in ev_audio["time_windows"]]),
            "evidence_vision_frames": ",".join([f"t{p:02d}" for p in top["vision"]]),
        }
        rows.append(row)

        expl[a4["id"][b]] = {
            "id": a4["id"][b],
            "prediction": {"polarity_class": cls, "polarity_name": name,
                           "intensity": round(float(R[0]), 4),
                           "uncertainty_sigma": round(sigma, 4) if sigma is not None else None},
            "dominant_modality": dom,
            "modality_contribution": {
                "alpha": {"text": round(float(A[0,0]),4), "audio": round(float(A[0,1]),4), "vision": round(float(A[0,2]),4)},
                "shapley": {k: round(float(v),4) for k,v in phi.items()},
                "pi": {k: round(float(v),4) for k,v in pi.items()},
            },
            "evidence_text": ev_text,
            "evidence_audio": ev_audio,
            "evidence_vision": ev_vision,
            "local_importance_occlusion": {m: [round(float(x), 4) for x in occl[m]] for m in C.MODS},
            "raw_text": a4["raw_text"][b],
        }
        print(f"[a4 {b+1:02d}] {name:>8} int={float(R[0]):+.3f} dom={dom} "
              f"phi={ {k: round(v,3) for k,v in phi.items()} }", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.Q3_OUT, "explainable_predictions.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(C.Q3_OUT, "explainable_explanations.json"), "w", encoding="utf-8") as f:
        json.dump(expl, f, ensure_ascii=False, indent=2)

    summary = {
        "n_samples": int(len(df)),
        "ensemble_seeds": [int(s) for s in C.SEEDS],
        "tau": tau,
        "class_distribution": {CLS_NAME[k]: int(v) for k, v in df["predicted_class"].value_counts().items()},
        "score_mean": float(df["predicted_score"].mean()),
        "score_std": float(df["predicted_score"].std()),
        "uncertainty_sigma_mean": float(df["score_uncertainty_sigma"].mean()),
        "dominant_modality_distribution": df["dominant_modality"].value_counts().to_dict(),
        "mean_alpha": {m: float(df[f"alpha_{m}"].mean()) for m in C.MODS},
        "mean_pi": {m: float(df[f"pi_{m}"].mean()) for m in C.MODS},
    }
    with open(os.path.join(C.Q3_OUT, "explainable_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("saved explainable_predictions.csv + explainable_explanations.json + explainable_summary.json", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()


