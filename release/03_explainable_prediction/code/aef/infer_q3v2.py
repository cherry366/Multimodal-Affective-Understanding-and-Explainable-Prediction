# -*- coding: utf-8 -*-
"""模块三 新方案（v2）可解释测试集推理 + 原生证据解释。"""
import json
import os
import math
import numpy as np
import pandas as pd
import torch

import config_q3v2 as C
import train_q2 as T
import data_q2 as D
from model_q3v2 import Q3V2Net
import data_q3 as D3

CLS_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}


def build(dev):
    return Q3V2Net().to(dev)


def load_models():
    dev = T.device_of()
    models = []
    for s in C.SEEDS:
        ck = T.load_ckpt(os.path.join(C.Q3V2_CKPT, f"q3v2_s{s}.pt"))
        m = build(dev)
        m.load_state_dict(ck["state"])
        m.eval()
        models.append((ck, m))
    return models, dev


@torch.no_grad()
def forward_ens(models, Xn, Mn, dev):
    Ps, Rs, As, Vs, Evs = [], [], [], [], {}
    for _, m in models:
        out = m(T.to_t(Xn, dev), T.to_t(Mn, dev))
        Ps.append(torch.softmax(out["logits"], 1).cpu().numpy())
        Rs.append(out["reg"].cpu().numpy())
        As.append(out["alpha"].cpu().numpy())
        Vs.append(torch.exp(0.5 * out["log_var"]).cpu().numpy())
        for k in C.MODS:
            Evs.setdefault(k, []).append(out["evidence"][k].cpu().numpy())
    return (np.mean(Ps, 0), np.mean(Rs, 0), np.mean(As, 0), np.mean(Vs, 0),
            {k: np.mean(Evs[k], 0) for k in C.MODS})


def logit_c(models, Xn, Mn, c, dev):
    tot = 0.0
    for _, m in models:
        out = m(T.to_t(Xn, dev), T.to_t(Mn, dev))
        tot = tot + out["logits"][0, c]
    return float((tot / len(models)).item())


def text_evidence(raw, positions):
    picked = [D3.word_for_position(raw, t) for t in positions]
    return {"words": [w for w in picked if w], "positions": [int(t) for t in positions]}


def audio_evidence(idx, positions):
    dur, fps, nf = D3.video_meta(idx)
    wins = [[round(t * dur / 50.0, 2), round((t + 1) * dur / 50.0, 2)] for t in positions]
    return {"time_windows": wins, "positions": [int(t) for t in positions], "duration": round(dur, 2)}


def vision_evidence(idx, positions):
    frames = []
    for t in positions:
        op = os.path.join(C.Q3V2_FRAMES, f"a4_{idx:02d}_frame_{t:02d}.jpg")
        ok = D3.extract_frame(idx, t, op)
        frames.append(ok or None)
    return {"key_frames": frames, "positions": [int(t) for t in positions]}


def shapley(models, Xn, Mn, c, dev):
    mods = list(C.MODS); n = len(mods)
    zero = {m: np.zeros_like(Xn[m]) for m in mods}
    zmsk = {m: np.zeros_like(Mn[m]) for m in mods}
    def key(s): return "+".join(sorted(s))
    def v(s):
        Xt, Mt = {}, {}
        for m in mods:
            Xt[m], Mt[m] = (Xn[m], Mn[m]) if m in s else (zero[m], zmsk[m])
        return logit_c(models, Xt, Mt, c, dev)
    subsets = [tuple(mods[i] for i in range(n) if (r >> i) & 1) for r in range(1 << n)]
    vals = {key(s): v(s) for s in subsets}
    phi = {m: 0.0 for m in mods}
    for m in mods:
        for s in subsets:
            if m in s: continue
            k = len(s)
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            phi[m] += w * (vals[key(list(s) + [m])] - vals[key(s)])
    v = np.array([phi[m] for m in mods])
    v = (v - v.mean()) / (v.std() + 1e-8)
    e = np.exp(v)
    pi = {m: float(e[i] / e.sum()) for i, m in enumerate(mods)}
    return phi, pi


def main():
    models, dev = load_models()
    T.get_data()
    stats = T._C["stats"]
    tau = float(np.mean([ck["tau"] for ck, _ in models]))
    A = D3.load_all_a4()
    Xn = {m: D.apply_norm(A[m], A["mask"][m], *stats[m]) for m in C.MODS}
    Mn = A["mask"]
    P, R, Alpha, Sigma, Ev = forward_ens(models, Xn, Mn, dev)
    pred = T.decision(P, R, tau, "band")

    ids = A["id"]
    rows = []
    exp = {}
    for i in range(len(ids)):
        Xb = {m: Xn[m][i:i+1] for m in C.MODS}
        Mb = {m: Mn[m][i:i+1] for m in C.MODS}
        c = int(pred[i])
        phi, pi = shapley(models, Xb, Mb, c, dev)
        dom_phi = max(C.MODS, key=lambda m: phi[m])
        dom_alpha = max(C.MODS, key=lambda m: Alpha[i, C.MODS.index(m)])

        ev = {m: Ev[m][i] for m in C.MODS}
        topk = {}
        for m in C.MODS:
            order = np.argsort(-ev[m])
            valid = [t for t in order if Mb[m][0, t] > 0]
            topk[m] = valid[:C.TOP_K]

        evidence_text = text_evidence(A["raw_text"][i], topk["text"])
        evidence_audio = audio_evidence(i + 1, topk["audio"])
        evidence_vision = vision_evidence(i + 1, topk["vision"])

        rows.append({
            "id": ids[i], "predicted_class": c, "predicted_class_name": CLS_NAME[c],
            "predicted_score": round(float(R[i]), 4),
            "score_uncertainty_sigma": round(float(Sigma[i]), 4),
            "classification_entropy": round(float(-(P[i] * np.log(P[i] + 1e-9)).sum()), 4),
            "prob_neg": round(float(P[i, 0]), 4), "prob_neu": round(float(P[i, 1]), 4),
            "prob_pos": round(float(P[i, 2]), 4),
            "dominant_modality": dom_phi, "dominant_modality_alpha": dom_alpha,
            "alpha_text": round(float(Alpha[i, 0]), 4), "alpha_audio": round(float(Alpha[i, 1]), 4),
            "alpha_vision": round(float(Alpha[i, 2]), 4),
            "shapley_text": round(phi["text"], 4), "shapley_audio": round(phi["audio"], 4),
            "shapley_vision": round(phi["vision"], 4),
            "pi_text": round(pi["text"], 4), "pi_audio": round(pi["audio"], 4),
            "pi_vision": round(pi["vision"], 4),
            "evidence_text": json.dumps(evidence_text, ensure_ascii=False),
            "evidence_audio_windows": json.dumps(evidence_audio["time_windows"], ensure_ascii=False),
            "evidence_vision_frames": json.dumps([os.path.basename(f) if f else None for f in evidence_vision["key_frames"]], ensure_ascii=False),
        })
        exp[ids[i]] = {
            "prediction": {"class": CLS_NAME[c], "class_id": c, "score": round(float(R[i]), 4),
                           "probs": {CLS_NAME[k]: round(float(P[i, k]), 4) for k in (0, 1, 2)}},
            "dominant_modality": dom_phi, "dominant_modality_alpha": dom_alpha,
            "modality_contribution": {"alpha": {m: round(float(Alpha[i, C.MODS.index(m)]), 4) for m in C.MODS},
                                      "shapley": {m: round(phi[m], 4) for m in C.MODS},
                                      "pi": {m: round(pi[m], 4) for m in C.MODS}},
            "evidence_text": evidence_text,
            "evidence_audio": evidence_audio,
            "evidence_vision": evidence_vision,
            "local_importance_evidence": {m: [round(float(v), 4) for v in ev[m]] for m in C.MODS},
            "raw_text": A["raw_text"][i],
        }

    pd.DataFrame(rows).to_csv(os.path.join(C.Q3V2_OUT, "explainable_predictions.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(C.Q3V2_OUT, "explainable_explanations.json"), "w", encoding="utf-8") as f:
        json.dump(exp, f, ensure_ascii=False, indent=1)
    summ = {
        "n_samples": len(ids), "ensemble_seeds": [int(s) for s in C.SEEDS], "tau": tau,
        "class_distribution": {CLS_NAME[k]: int((pred == k).sum()) for k in (0, 1, 2)},
        "score_mean": float(R.mean()), "score_std": float(R.std()),
        "uncertainty_sigma_mean": float(Sigma.mean()),
        "dominant_modality_distribution": {m: int(sum(1 for r in rows if r["dominant_modality"] == m)) for m in C.MODS},
        "mean_alpha": {m: round(float(Alpha[:, C.MODS.index(m)].mean()), 4) for m in C.MODS},
    }
    with open(os.path.join(C.Q3V2_OUT, "explainable_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
