# -*- coding: utf-8 -*-
"""模块三 验证集评估：基础性能 + 可解释性忠实度（Shapley/IG/LOMO）+ 错误归因。"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import config_q3 as C
import data_q2 as D
import train_q2 as T2
from explain_q3 import load_models, ens_predict, shapley_values, local_importance

CLS_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}


def drop_modality(Xn, Mn, m):
    """把模态 m 的特征与 mask 全部置 0（保持其它模态不变）。"""
    Xo = {k: Xn[k].copy() for k in C.MODS}
    Mo = {k: Mn[k].copy() for k in C.MODS}
    Xo[m] = np.zeros_like(Xn[m])
    Mo[m] = np.zeros_like(Mn[m])
    return Xo, Mo


def zero_positions(Xn, Mn, m, pos):
    """把模态 m 的指定位置置 0（特征与 mask）。"""
    Xo = {k: Xn[k].copy() for k in C.MODS}
    Mo = {k: Mn[k].copy() for k in C.MODS}
    if len(pos):
        Xo[m][0, pos, :] = 0.0
        Mo[m][0, pos] = 0.0
    return Xo, Mo


def keep_positions(Xn, Mn, m, pos):
    """只保留模态 m 的指定位置，其余置 0。"""
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
    stats = T2._C["stats"]
    tau = float(np.mean([ck["tau"] for ck, _ in models]))

    # ---------- 1) 验证集全量基础性能（clean，三模态完整） ----------
    Xv = dt["X"]["valid"]
    Mv = dt["M"]["valid"]
    yc = dt["items"]["valid"]["y_cls"]
    yr = dt["items"]["valid"]["y_reg"]
    P, R, A, V = ens_predict(models, Xv, Mv, dev)
    pred = T2.decision(P, R, tau, "band")
    metrics = {
        "accuracy": float(accuracy_score(yc, pred)),
        "f1_macro": float(f1_score(yc, pred, average="macro")),
        "f1_weighted": float(f1_score(yc, pred, average="weighted")),
        "mae": float(np.abs(R - yr).mean()),
        "pearson": float(np.corrcoef(R, yr)[0, 1]),
    }
    cm = confusion_matrix(yc, pred, labels=[0, 1, 2]).tolist()
    print("[valid clean]", json.dumps(metrics, ensure_ascii=False), flush=True)

    # ---------- 2) 可解释性忠实度（分层抽样 200 条） ----------
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
        Pb, Rb, Ab, _ = ens_predict(models, Xb, Mb, dev)
        c = int(np.argmax(Pb[0]))
        base = float(Pb[0, c])
        yc_i = int(yc[i])

        phi, pi = shapley_values(models, Xb, Mb, c, dev)
        dom = max(C.MODS, key=lambda m: phi[m])

        # LOMO：逐模态置 0 的置信度下降
        drops = {}
        for m in C.MODS:
            Xo, Mo = drop_modality(Xb, Mb, m)
            Po, _, _, _ = ens_predict(models, Xo, Mo, dev)
            drops[m] = base - float(Po[0, c])
        comp_dom = drops[dom]
        others = [drops[m] for m in C.MODS if m != dom]
        comp_rand = float(np.mean(others))
        hit = 1.0 if comp_dom >= comp_rand else 0.0

        row = {
            "sample_idx": int(i), "y_true": yc_i, "y_pred": int(pred[i]), "cls": c,
            "dominant": dom, "base_conf": base,
            "alpha_text": float(Ab[0,0]), "alpha_audio": float(Ab[0,1]), "alpha_vision": float(Ab[0,2]),
            "phi_text": phi["text"], "phi_audio": phi["audio"], "phi_vision": phi["vision"],
            "pi_text": pi["text"], "pi_audio": pi["audio"], "pi_vision": pi["vision"],
            "drop_text": drops["text"], "drop_audio": drops["audio"], "drop_vision": drops["vision"],
            "comp_dom": comp_dom, "comp_rand": comp_rand, "dom_hit": hit,
        }
        rows.append(row)
    fdf = pd.DataFrame(rows)

    # 门控-Shapley 一致性
    spe = {}
    for m in C.MODS:
        spe[f"rho_alpha_phi_{m}"] = float(spearmanr(fdf[f"alpha_{m}"], fdf[f"phi_{m}"]).correlation)
        spe[f"rho_alpha_pi_{m}"] = float(spearmanr(fdf[f"alpha_{m}"], fdf[f"pi_{m}"]).correlation)
    # 主导模态一致性（α 的 argmax vs Shapley 的 argmax）
    a_dom = fdf[[f"alpha_{m}" for m in C.MODS]].values.argmax(1)
    a_dom = [C.MODS[k] for k in a_dom]
    s_dom = fdf["dominant"].tolist()
    dom_agree = float(np.mean([a == s for a, s in zip(a_dom, s_dom)]))

    # ---------- 3) IG 局部证据忠实度（取前 100 条） ----------
    ig_idx = idx[:100]
    ig_rows = []
    for i in ig_idx:
        Xb = {m: Xv[m][i:i+1] for m in C.MODS}
        Mb = {m: Mv[m][i:i+1] for m in C.MODS}
        Pb, Rb, Ab, _ = ens_predict(models, Xb, Mb, dev)
        c = int(np.argmax(Pb[0]))
        base = float(Pb[0, c])
        imp = local_importance(models, Xb, Mb, c, dev)
        dom = max(C.MODS, key=lambda m: imp[m].sum() if imp[m].max() > 1e-12 else -1e9)
        w = imp[dom]
        order = np.argsort(-w)
        valid = [t for t in order if Mb[dom][0, t] > 0]
        topk = valid[:C.TOP_K]
        # 随机对照（在有效位中随机选 k 个）
        eff = [t for t in range(50) if Mb[dom][0, t] > 0]
        randk = list(rng.choice(eff, size=min(len(eff), C.TOP_K), replace=False)) if len(eff) else []
        # 删除 top-k（Comprehensiveness）
        Xd, Md = zero_positions(Xb, Mb, dom, topk)
        Pd, _, _, _ = ens_predict(models, Xd, Md, dev)
        comp_topk = base - float(Pd[0, c])
        # 删除随机 k
        Xr, Mr = zero_positions(Xb, Mb, dom, randk)
        Pr, _, _, _ = ens_predict(models, Xr, Mr, dev)
        comp_randk = base - float(Pr[0, c])
        # 只保留 top-k（Sufficiency）
        Xk, Mk = keep_positions(Xb, Mb, dom, topk)
        Pk, _, _, _ = ens_predict(models, Xk, Mk, dev)
        suff = base - float(Pk[0, c])
        ig_rows.append({"sample_idx": int(i), "cls": c, "modality": dom,
                        "comp_topk": float(comp_topk), "comp_randk": float(comp_randk),
                        "suff_topk": float(suff)})
    igdf = pd.DataFrame(ig_rows)

    faith = {
        "n_samples": int(len(fdf)),
        "dom_hit_rate": float(fdf["dom_hit"].mean()),
        "mean_comp_dom": float(fdf["comp_dom"].mean()),
        "mean_comp_rand": float(fdf["comp_rand"].mean()),
        "spearman": spe,
        "alpha_vs_shapley_dom_agreement": float(dom_agree),
        "ig": {
            "n_samples": int(len(igdf)),
            "mean_comp_topk": float(igdf["comp_topk"].mean()),
            "mean_comp_randk": float(igdf["comp_randk"].mean()),
            "mean_suff_topk": float(igdf["suff_topk"].mean()),
            "comp_win_rate": float((igdf["comp_topk"] >= igdf["comp_randk"]).mean()),
        },
    }
    print("[faithfulness]", json.dumps(faith, ensure_ascii=False, indent=1), flush=True)

    # ---------- 4) 错误归因 ----------
    per_class = {}
    for c in (0, 1, 2):
        mask = yc == c
        per_class[CLS_NAME[c]] = {
            "n": int(mask.sum()),
            "acc": float((pred[mask] == c).mean()),
            "f1": float(f1_score(yc == c, pred == c)),
            "mean_true_intensity": float(yr[mask].mean()),
            "mean_pred_intensity": float(R[mask].mean()),
            "mae": float(np.abs(R[mask] - yr[mask]).mean()),
        }
    # 混淆统计
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
        "faithfulness": faith,
        "tau": tau,
        "ensemble_seeds": [int(s) for s in C.SEEDS],
    }
    with open(os.path.join(C.Q3_OUT, "valid_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    fdf.to_csv(os.path.join(C.Q3_OUT, "faithfulness_samples.csv"), index=False, encoding="utf-8-sig")
    igdf.to_csv(os.path.join(C.Q3_OUT, "ig_faithfulness.csv"), index=False, encoding="utf-8-sig")
    print("saved valid_eval.json / faithfulness_samples.csv / ig_faithfulness.csv", flush=True)


if __name__ == "__main__":
    main()

