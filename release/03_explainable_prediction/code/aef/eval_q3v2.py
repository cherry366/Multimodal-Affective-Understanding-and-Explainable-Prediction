# -*- coding: utf-8 -*-
"""模块三 新方案（v2）验证集评估 + 可解释性忠实度。

可解释信号：
  - 原生局部证据：evidence e_m(t)（前向直接输出，参与决策）
  - 模态作用：门控 alpha + 后验 Shapley phi（互证）
忠实度检验：模态级 LOPO + 原生证据 top-k vs 随机（按模态分别）。
"""
import json
import os
import math
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import config_q3v2 as C
import data_q2 as D
import train_q2 as T
from model_q3v2 import Q3V2Net

CLS_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}


def build(dev):
    return Q3V2Net().to(dev)


def load_models():
    dev = T.device_of()
    dt = T.get_data()
    models = []
    for s in C.SEEDS:
        ck = T.load_ckpt(os.path.join(C.Q3V2_CKPT, f"q3v2_s{s}.pt"))
        m = build(dev)
        m.load_state_dict(ck["state"])
        m.eval()
        models.append((ck, m))
    return models, dt


@torch.no_grad()
def ens_forward(models, Xn, Mn, dev):
    Ps, Rs, As, Evs = [], [], [], {}
    for _, m in models:
        out = m(T.to_t(Xn, dev), T.to_t(Mn, dev))
        Ps.append(torch.softmax(out["logits"], 1).cpu().numpy())
        Rs.append(out["reg"].cpu().numpy())
        As.append(out["alpha"].cpu().numpy())
        for k in C.MODS:
            Evs.setdefault(k, []).append(out["evidence"][k].cpu().numpy())
    return (np.mean(Ps, 0), np.mean(Rs, 0), np.mean(As, 0),
            {k: np.mean(Evs[k], 0) for k in C.MODS})


def logit_c_ensemble(models, Xn, Mn, c, dev):
    tot = 0.0
    for _, m in models:
        out = m(T.to_t(Xn, dev), T.to_t(Mn, dev))
        tot = tot + out["logits"][0, c]
    return float((tot / len(models)).item())


def shapley(models, Xn, Mn, c, dev):
    mods = list(C.MODS)
    n = len(mods)
    zero = {m: np.zeros_like(Xn[m]) for m in mods}
    zmsk = {m: np.zeros_like(Mn[m]) for m in mods}
    def key(s):
        return "+".join(sorted(s))
    def v(s):
        Xt, Mt = {}, {}
        for m in mods:
            if m in s:
                Xt[m], Mt[m] = Xn[m], Mn[m]
            else:
                Xt[m], Mt[m] = zero[m], zmsk[m]
        return logit_c_ensemble(models, Xt, Mt, c, dev)
    subsets = []
    for r in range(1 << n):
        s = tuple(mods[i] for i in range(n) if (r >> i) & 1)
        subsets.append(s)
    vals = {key(s): v(s) for s in subsets}
    phi = {m: 0.0 for m in mods}
    for m in mods:
        for s in subsets:
            if m in s:
                continue
            k = len(s)
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            phi[m] += w * (vals[key(list(s) + [m])] - vals[key(s)])
    return phi


def main():
    models, dt = load_models()
    dev = T.device_of()
    tau = float(np.mean([ck["tau"] for ck, _ in models]))

    Xv = dt["X"]["valid"]
    Mv = dt["M"]["valid"]
    yc = dt["items"]["valid"]["y_cls"]
    yr = dt["items"]["valid"]["y_reg"]

    P, R, A, Ev = ens_forward(models, Xv, Mv, dev)
    pred = T.decision(P, R, tau, "band")
    metrics = {
        "accuracy": float(accuracy_score(yc, pred)),
        "f1_macro": float(f1_score(yc, pred, average="macro")),
        "f1_weighted": float(f1_score(yc, pred, average="weighted")),
        "mae": float(np.abs(R - yr).mean()),
        "pearson": float(np.corrcoef(R, yr)[0, 1]),
    }
    cm = confusion_matrix(yc, pred, labels=[0, 1, 2]).tolist()

    # 分层 200 条
    rng = np.random.default_rng(2026)
    idx = []
    for c in (0, 1, 2):
        cand = np.where(yc == c)[0]
        n = int(round(200 * len(cand) / len(yc)))
        idx.extend(rng.choice(cand, size=max(n, 1), replace=False).tolist())
    idx = np.array(idx[:200])

    rows = []
    ev_rows = []
    for i in idx:
        Xb = {m: Xv[m][i:i+1] for m in C.MODS}
        Mb = {m: Mv[m][i:i+1] for m in C.MODS}
        Pb, Rb, Ab, Eb = ens_forward(models, Xb, Mb, dev)
        c = int(np.argmax(Pb[0]))
        base = float(Pb[0, c])
        yc_i = int(yc[i])

        phi = shapley(models, Xb, Mb, c, dev)
        dom = max(C.MODS, key=lambda m: phi[m])

        # 模态级 LOPO
        drops = {}
        for m in C.MODS:
            Xo = {k: Xb[k].copy() for k in C.MODS}
            Mo = {k: Mb[k].copy() for k in C.MODS}
            Xo[m] = np.zeros_like(Xb[m])
            Mo[m] = np.zeros_like(Mb[m])
            Po, _, _, _ = ens_forward(models, Xo, Mo, dev)
            drops[m] = base - float(Po[0, c])
        comp_dom = drops[dom]
        comp_rand = float(np.mean([drops[m] for m in C.MODS if m != dom]))

        row = {
            "sample_idx": int(i), "y_true": yc_i, "y_pred": int(pred[i]), "cls": c,
            "dominant": dom, "base_conf": base,
            "alpha_text": float(Ab[0,0]), "alpha_audio": float(Ab[0,1]), "alpha_vision": float(Ab[0,2]),
            "phi_text": phi["text"], "phi_audio": phi["audio"], "phi_vision": phi["vision"],
            "drop_text": drops["text"], "drop_audio": drops["audio"], "drop_vision": drops["vision"],
            "comp_dom": comp_dom, "comp_rand": comp_rand,
            "dom_hit": 1.0 if comp_dom >= comp_rand else 0.0,
        }
        rows.append(row)

        # 原生证据 top-k vs 随机（按模态分别）
        for m in C.MODS:
            e = Eb[m][0]
            order = np.argsort(-e)
            valid = [t for t in order if Mb[m][0, t] > 0]
            eff = [t for t in range(50) if Mb[m][0, t] > 0]
            if not eff:
                continue
            topk = valid[:C.TOP_K]
            randk = list(rng.choice(eff, size=min(len(eff), C.TOP_K), replace=False))

            Xd = {k: Xb[k].copy() for k in C.MODS}
            Md = {k: Mb[k].copy() for k in C.MODS}
            Xd[m][0, topk, :] = 0.0; Md[m][0, topk] = 0.0
            Pd, _, _, _ = ens_forward(models, Xd, Md, dev)
            comp_topk = base - float(Pd[0, c])

            Xr = {k: Xb[k].copy() for k in C.MODS}
            Mr = {k: Mb[k].copy() for k in C.MODS}
            Xr[m][0, randk, :] = 0.0; Mr[m][0, randk] = 0.0
            Pr, _, _, _ = ens_forward(models, Xr, Mr, dev)
            comp_randk = base - float(Pr[0, c])
            ev_rows.append({"sample_idx": int(i), "cls": c, "modality": m,
                            "comp_topk": float(comp_topk), "comp_randk": float(comp_randk)})

    fdf = pd.DataFrame(rows)
    edf = pd.DataFrame(ev_rows)

    spe = {}
    for m in C.MODS:
        spe[f"rho_alpha_phi_{m}"] = float(spearmanr(fdf[f"alpha_{m}"], fdf[f"phi_{m}"]).correlation)
    a_dom = [C.MODS[k] for k in fdf[["alpha_text", "alpha_audio", "alpha_vision"]].values.argmax(1)]
    dom_agree = float(np.mean([a == s for a, s in zip(a_dom, fdf["dominant"].tolist())]))

    ev_summ = {}
    for m in C.MODS:
        g = edf[edf["modality"] == m]
        ev_summ[m] = {
            "n": int(len(g)),
            "mean_comp_topk": float(g["comp_topk"].mean()),
            "mean_comp_randk": float(g["comp_randk"].mean()),
            "comp_win_rate": float((g["comp_topk"] >= g["comp_randk"]).mean()),
        }

    per_class = {}
    for c in (0, 1, 2):
        mask = yc == c
        per_class[CLS_NAME[c]] = {
            "n": int(mask.sum()),
            "acc": float((pred[mask] == c).mean()),
            "f1": float(f1_score(yc == c, pred == c)),
            "mae": float(np.abs(R[mask] - yr[mask]).mean()),
        }
    err_pairs = {}
    for i in range(len(yc)):
        if pred[i] != yc[i]:
            k = f"{CLS_NAME[yc[i]]}->{CLS_NAME[pred[i]]}"
            err_pairs[k] = err_pairs.get(k, 0) + 1
    err_pairs = dict(sorted(err_pairs.items(), key=lambda kv: -kv[1]))

    out = {
        "valid_clean": metrics,
        "confusion_matrix": cm,
        "per_class": per_class,
        "error_pairs": err_pairs,
        "faithfulness": {
            "n_samples": int(len(fdf)),
            "dom_hit_rate": float(fdf["dom_hit"].mean()),
            "mean_comp_dom": float(fdf["comp_dom"].mean()),
            "mean_comp_rand": float(fdf["comp_rand"].mean()),
            "spearman": spe,
            "alpha_vs_shapley_dom_agreement": float(dom_agree),
        },
        "native_evidence_faithfulness": ev_summ,
        "tau": tau,
        "ensemble_seeds": [int(s) for s in C.SEEDS],
    }
    with open(os.path.join(C.Q3V2_OUT, "valid_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    fdf.to_csv(os.path.join(C.Q3V2_OUT, "faithfulness_samples.csv"), index=False, encoding="utf-8-sig")
    edf.to_csv(os.path.join(C.Q3V2_OUT, "native_evidence_faithfulness.csv"), index=False, encoding="utf-8-sig")
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
