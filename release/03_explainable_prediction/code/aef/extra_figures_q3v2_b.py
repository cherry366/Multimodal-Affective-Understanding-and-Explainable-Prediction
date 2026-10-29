# -*- coding: utf-8 -*-
"""模块三 v2 补充图（第二批）：qx2_20 ~ qx2_26。

qx2_20 证据删除扫描（重要性删除 vs 随机删除，10/20/30/50%）
qx2_21 稳定性与敏感性（LR×WD 网格 / 中性带宽 tau / 多种子 / 收敛曲线）
qx2_22 显著性检验（v1 vs v2 配对 Wilcoxon + Bootstrap 95% CI）
qx2_23 局部片段重要性聚合规律（位置分布 / 高频证据词 / 判对判错 / 绝对槽位）
qx2_24 门控 alpha 与 Shapley phi 一致性散点
qx2_25 Shapley 精确分解示例（8 联盟 + 瀑布）
qx2_26 复杂度表（参数量 / 训练耗时 / 推理耗时）
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import accuracy_score, f1_score

import config_q3v2 as C
import train_q2 as T
import extra_figures_q3v2 as E2
from eval_q3v2 import shapley as exact_shapley

OUT, FIGS = C.Q3V2_OUT, C.Q3V2_FIGS
Q3_OUT = os.path.join(C.ROOT, "q3_output")
C_TEXT, C_AUDIO, C_VIS = E2.C_TEXT, E2.C_AUDIO, E2.C_VIS
C_TITLE, C_GRAY = E2.C_TITLE, E2.C_GRAY
CLS_COLOR, CLS_NAME, MOD_CN = E2.CLS_COLOR, E2.CLS_NAME, E2.MOD_CN
MOD_COLOR = {"text": C_TEXT, "audio": C_AUDIO, "vision": C_VIS}
save = E2.save
plt.rcParams["font.size"] = 10.5


# ------------------------------------------------------------------ 通用
def valid_pack():
    """返回 (models, dev, Xv, Mv, yc, yr, tau)。"""
    models, dev = E2.load_models()
    dt = T.get_data()
    Xv, Mv = dt["X"]["valid"], dt["M"]["valid"]
    yc = dt["items"]["valid"]["y_cls"]
    yr = dt["items"]["valid"]["y_reg"]
    tau = float(np.mean([ck["tau"] for ck, _ in models]))
    return models, dev, Xv, Mv, yc, yr, tau


@torch.no_grad()
def ens_batch(models, Xn, Mn, dev, bs=128, want_z=False):
    """分块多种子集成前向，返回 (P, R, Alpha, Evidence[, Z])。"""
    N = Xn["text"].shape[0]
    Ps, Rs, As, Zs = [], [], [], []
    Ev = {m: [] for m in C.MODS}
    for s in range(0, N, bs):
        idx = np.arange(s, min(s + bs, N))
        Xb = {m: Xn[m][idx] for m in C.MODS}
        Mb = {m: Mn[m][idx] for m in C.MODS}
        out = E2.ens_output(models, Xb, Mb, dev, want_z=want_z)
        Ps.append(out[0]); Rs.append(out[1]); As.append(out[2])
        for m in C.MODS:
            Ev[m].append(out[3][m])
        if want_z:
            Zs.append(out[4])
    res = [np.concatenate(Ps), np.concatenate(Rs), np.concatenate(As),
           {m: np.concatenate(Ev[m]) for m in C.MODS}]
    if want_z:
        res.append(np.concatenate(Zs))
    return tuple(res)


def _rank_from_scores(S):
    """(N,L) 分数 -> 名次（0 = 最大）。"""
    order = np.argsort(-S, axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(S.shape[1])[None, :].repeat(S.shape[0], 0), axis=1)
    return rank


def _k_of(rate, npos):
    k = np.maximum(1, np.ceil(rate * npos).astype(int))
    return np.minimum(k, np.maximum(npos.astype(int) - 1, 1))


def sel_topk(Ev, M, rate):
    """按原生证据取每个样本前 rate 比例的有效位置。"""
    S = np.where(M > 0, Ev, -np.inf)
    rank = _rank_from_scores(S)
    return (rank < _k_of(rate, M.sum(1))[:, None]) & (M > 0)


def sel_rand(M, rate, rng):
    """随机取同样数量的有效位置。"""
    S = np.where(M > 0, rng.random(M.shape), np.inf)
    rank = _rank_from_scores(-S)
    return (rank < _k_of(rate, M.sum(1))[:, None]) & (M > 0)


def delete_sel(Xm, Mm, sel):
    """按布尔选择置零特征与掩码（= 局部片段不可用）。"""
    Xn, Mn = Xm.copy(), Mm.copy()
    Xn[sel] = 0.0
    Mn[sel] = 0.0
    return Xn, Mn


def mets_of(yc, pred, R, yr):
    return {"acc": float(accuracy_score(yc, pred)),
            "f1": float(f1_score(yc, pred, average="macro")),
            "mae": float(np.abs(R - yr).mean()),
            "pearson": float(np.corrcoef(R, yr)[0, 1])}


def boot_delta(yc, pred1, R1, pred2, R2, yr, B=2000, seed=11):
    """配对 bootstrap：delta = AEF − 后验基线（mae 取正为好 -> 转成 neg_mae）。"""
    rng = np.random.default_rng(seed)
    N = len(yc)
    acc, f1, nmae, pear = [], [], [], []
    for _ in range(B):
        i = rng.integers(0, N, N)
        m1 = mets_of(yc[i], pred1[i], R1[i], yr[i])
        m2 = mets_of(yc[i], pred2[i], R2[i], yr[i])
        acc.append(m2["acc"] - m1["acc"])
        f1.append(m2["f1"] - m1["f1"])
        nmae.append(m1["mae"] - m2["mae"])
        pear.append(m2["pearson"] - m1["pearson"])
    out = {}
    for k, v in (("acc", acc), ("f1", f1), ("neg_mae", nmae), ("pearson", pear)):
        v = np.asarray(v)
        out[k] = {"mean": float(v.mean()),
                  "lo": float(np.percentile(v, 2.5)),
                  "hi": float(np.percentile(v, 97.5))}
    return out

# ================================================================= qx2_20 删除实验扫描
RATES = [0.10, 0.20, 0.30, 0.50]
N_RAND = 3


def qx2_20_deletion_scan():
    models, dev, Xv, Mv, yc, yr, tau = valid_pack()
    P0, R0, _, Ev = ens_batch(models, Xv, Mv, dev)
    pred0 = T.decision(P0, R0, tau, "band")
    base = mets_of(yc, pred0, R0, yr)

    rows = []
    for mi, m in enumerate(C.MODS):
        for r in RATES:
            sel = sel_topk(Ev[m], Mv[m], r)
            Xn, Mn = dict(Xv), dict(Mv)
            Xn[m], Mn[m] = delete_sel(Xv[m], Mv[m], sel)
            Pe, Re, _, _ = ens_batch(models, Xn, Mn, dev)
            pre = T.decision(Pe, Re, tau, "band")
            me = mets_of(yc, pre, Re, yr)
            flips_ev = float(np.mean(pre != pred0))

            rm = {"acc": [], "f1": [], "mae": [], "pearson": []}
            flips_rd = []
            for d in range(N_RAND):
                rng = np.random.default_rng(1000 * mi + 100 * int(r * 100) + d)
                rs = sel_rand(Mv[m], r, rng)
                Xr, Mr = dict(Xv), dict(Mv)
                Xr[m], Mr[m] = delete_sel(Xv[m], Mv[m], rs)
                Pr, Rr, _, _ = ens_batch(models, Xr, Mr, dev)
                prr = T.decision(Pr, Rr, tau, "band")
                mr = mets_of(yc, prr, Rr, yr)
                for k in rm:
                    rm[k].append(mr[k])
                flips_rd.append(float(np.mean(prr != pred0)))
            rmean = {k: float(np.mean(v)) for k, v in rm.items()}
            rows.append({
                "modality": m, "rate": r,
                "acc_ev": me["acc"], "f1_ev": me["f1"], "mae_ev": me["mae"], "pearson_ev": me["pearson"],
                "acc_rand": rmean["acc"], "f1_rand": rmean["f1"],
                "mae_rand": rmean["mae"], "pearson_rand": rmean["pearson"],
                "d_acc": me["acc"] - rmean["acc"], "d_f1": me["f1"] - rmean["f1"],
                "d_mae": me["mae"] - rmean["mae"],
                "flip_ev": flips_ev, "flip_rand": float(np.mean(flips_rd)),
                "acc_base": base["acc"], "f1_base": base["f1"], "mae_base": base["mae"],
                "n_samples": int(len(yc)), "n_rand_draws": N_RAND,
            })
            print("DEL", json.dumps(rows[-1], ensure_ascii=False), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "evidence_deletion_scan.csv"), index=False, encoding="utf-8-sig")

    x = np.array([0.0] + RATES)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4))
    panels = ((axes[0, 0], "acc", "Accuracy", base["acc"], "a"),
              (axes[0, 1], "f1", "F1-macro", base["f1"], "b"),
              (axes[1, 0], "mae", "MAE", base["mae"], "c"))
    for ax, key, name, basev, tag in panels:
        for m in C.MODS:
            g = df[df.modality == m].sort_values("rate")
            yv = np.concatenate([[basev], g[f"{key}_ev"].to_numpy()])
            yr_ = np.concatenate([[basev], g[f"{key}_rand"].to_numpy()])
            ax.plot(x, yv, "-o", color=MOD_COLOR[m], lw=2.0, ms=6, label=f"{MOD_CN[m]}·重要性删除")
            ax.plot(x, yr_, "--s", color=MOD_COLOR[m], lw=1.4, ms=5, alpha=0.6,
                    label=f"{MOD_CN[m]}·随机删除")
        ax.axhline(basev, color=C_GRAY, lw=1.0, ls=":")
        ax.set_xlabel("局部删除比例 r")
        ax.set_ylabel(name)
        ax.set_title(f"({tag}) 删除比例 - {name}", fontsize=10.5)
        ax.grid(alpha=0.25)

    ax = axes[1, 1]
    w = 0.22
    xi = np.arange(len(RATES))
    for j, m in enumerate(C.MODS):
        g = df[df.modality == m].sort_values("rate")
        d = g["d_acc"].to_numpy() * 100
        ax.bar(xi + (j - 1) * w, d, w, color=MOD_COLOR[m], label=MOD_CN[m], alpha=0.9)
        for xx, vv in zip(xi + (j - 1) * w, d):
            ax.text(xx, vv + (0.10 if vv >= 0 else -0.10), f"{vv:+.1f}", ha="center",
                    va="bottom" if vv >= 0 else "top", fontsize=7.4)
    ax.axhline(0, color="#444444", lw=1.0)
    ax.set_xticks(xi)
    ax.set_xticklabels([f"{int(r*100)}%" for r in RATES])
    ax.set_xlabel("局部删除比例 r")
    ax.set_ylabel("Accuracy 差（重要性 - 随机，百分点）")
    ax.set_ylim(-4.0, 1.0)
    ax.set_title("(d) 重要性删除相对随机删除的额外损失", fontsize=10.5)
    ax.legend(fontsize=8.5, framealpha=0.9, loc="lower left")
    ax.grid(axis="y", alpha=0.25)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("模块三 AEF 局部证据删除扫描（验证集 %d 条 × 3 种子集成；随机删除取 %d 次均值）"
                 % (len(yc), N_RAND), fontsize=12.5, color=C_TITLE)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    save(fig, "qx2_20_deletion_scan")




# ================================================================= qx2_21 稳定性与敏感性
def qx2_21_sensitivity():
    models, dev, Xv, Mv, yc, yr, tau = valid_pack()
    P0, R0, _, _ = ens_batch(models, Xv, Mv, dev)
    taus = [0.0] + list(C.TAU_GRID)
    tf1, tacc = [], []
    for t in taus:
        pr = T.decision(P0, R0, t, "band")
        tf1.append(float(f1_score(yc, pr, average="macro")))
        tacc.append(float(accuracy_score(yc, pr)))
    best_i = int(np.argmax(tf1))

    grid = json.load(open(os.path.join(OUT, "sensitivity_grid.json"), encoding="utf-8"))
    lrs, wds = grid["lrs"], grid["wds"]
    F = np.full((len(lrs), len(wds)), np.nan)
    for row in grid["rows"]:
        i, j = lrs.index(row["lr"]), wds.index(row["wd"])
        F[i, j] = row["valid_f1"]

    ts = pd.read_csv(os.path.join(OUT, "train_summary.csv"))
    seeds = ts["seed"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))

    ax = axes[0, 0]
    im = ax.imshow(F, cmap="YlGnBu", aspect="auto", vmin=float(np.nanmin(F)) - 0.002,
                   vmax=float(np.nanmax(F)) + 0.002)
    lo, hi = float(np.nanmin(F)), float(np.nanmax(F))
    for i in range(len(lrs)):
        for j in range(len(wds)):
            frac = (F[i, j] - lo) / max(hi - lo, 1e-9)
            ax.text(j, i, f"{F[i, j]:.4f}", ha="center", va="center", fontsize=9.5,
                    color="white" if frac > 0.55 else "#1B2A38",
                    fontweight="bold" if F[i, j] == hi else "normal")
    ax.set_xticks(range(len(wds))); ax.set_xticklabels([f"{w:g}" for w in wds])
    ax.set_yticks(range(len(lrs))); ax.set_yticklabels([f"{l:g}" for l in lrs])
    ax.set_xlabel("weight decay"); ax.set_ylabel("learning rate")
    ax.set_title("(a) 学习率 x 权重衰减：验证集 F1-macro", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    ax.plot(taus, tf1, "-o", color=C_TITLE, lw=2.0, ms=4.5, label="F1-macro")
    ax.plot(taus, tacc, "-^", color=C_AUDIO, lw=1.6, ms=4.5, label="Accuracy")
    ax.axvline(taus[best_i], color="#B3554D", ls="--", lw=1.2,
               label=f"最优 tau={taus[best_i]:.2f}")
    ax.set_xlabel("中性带宽阈值 tau"); ax.set_ylabel("验证集指标")
    ax.set_title("(b) 中性带宽阈值敏感性（3 种子集成）", fontsize=10.5)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    xi = np.arange(len(seeds)); w = 0.27
    accs = ts["valid_clean_acc"].to_numpy()
    f1s = ts["valid_clean_f1_macro"].to_numpy()
    maes = ts["valid_clean_mae"].to_numpy()
    ax.bar(xi - w, accs, w, color="#5B8FC9", label="Accuracy")
    ax.bar(xi, f1s, w, color="#E0A34B", label="F1-macro")
    ax.bar(xi + w, maes, w, color="#8E6BB4", label="MAE")
    for xx, vv in list(zip(xi - w, accs)) + list(zip(xi, f1s)) + list(zip(xi + w, maes)):
        ax.text(xx, vv + 0.012, f"{vv:.3f}", ha="center", fontsize=7.6)
    for arr, cc in ((accs, "#5B8FC9"), (f1s, "#E0A34B"), (maes, "#8E6BB4")):
        ax.axhline(float(arr.mean()), color=cc, ls=":", lw=1.0)
    ax.set_xticks(xi); ax.set_xticklabels([f"seed={s}" for s in seeds])
    ax.set_ylim(0, 1.0); ax.set_ylabel("验证集指标")
    ax.set_title("(c) 三种子独立训练（虚线为均值）", fontsize=10.5)
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    for ck, _ in models:
        h = ck["metrics"]["hist"]
        ep = [d["epoch"] for d in h]
        f1h = [d["val"]["f1_macro"] for d in h]
        ax.plot(ep, f1h, "-o", ms=3.4, lw=1.6,
                label=f"seed={ck['seed']} (best ep{ck['metrics']['best_epoch']})")
    ax.set_xlabel("训练轮次 epoch"); ax.set_ylabel("验证集 F1-macro")
    ax.set_title("(d) 收敛曲线（编码器冻结，仅训练融合头）", fontsize=10.5)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25)

    fig.suptitle("模块三 AEF 稳定性与敏感性分析（验证集）", fontsize=12.5, color=C_TITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "qx2_21_sensitivity")




# ================================================================= qx2_22 显著性检验
def qx2_22_significance():
    models, dev, Xv, Mv, yc, yr, tau2 = valid_pack()
    P2, R2, _, _ = ens_batch(models, Xv, Mv, dev)
    pred2 = T.decision(P2, R2, tau2, "band")

    v1 = pd.read_csv(os.path.join(Q3_OUT, "valid_predictions.csv")).sort_values("idx")
    yc1 = v1["y_cls"].to_numpy()
    yr1 = v1["y_reg"].to_numpy()
    pred1 = v1["pred_cls"].to_numpy()
    R1 = v1["pred_reg"].to_numpy()
    same = bool(np.allclose(yc1, yc) and np.allclose(yr1, yr, atol=1e-5))

    m1, m2 = mets_of(yc1, pred1, R1, yr1), mets_of(yc, pred2, R2, yr)
    boot = boot_delta(yc, pred1, R1, pred2, R2, yr, B=2000, seed=11)

    a = pd.read_csv(os.path.join(Q3_OUT, "occlusion_faithfulness_per_modality.csv"))
    b = pd.read_csv(os.path.join(OUT, "native_evidence_faithfulness.csv"))
    a["margin"] = a["comp_topk"] - a["comp_randk"]
    b["margin"] = b["comp_topk"] - b["comp_randk"]
    mg = a.merge(b, on=["sample_idx", "modality"], suffixes=("_v1", "_v2"))
    wil = {}
    for m in C.MODS:
        g = mg[mg.modality == m]
        if len(g) < 8:
            continue
        st, p = wilcoxon(g["margin_v2"].to_numpy(), g["margin_v1"].to_numpy(),
                         alternative="two-sided", zero_method="wilcox")
        wil[m] = {"n": int(len(g)), "wilcoxon_p_v2_vs_v1": float(p),
                  "median_margin_v1": float(g["margin_v1"].median()),
                  "median_margin_v2": float(g["margin_v2"].median()),
                  "win_rate_v1": float((g["comp_topk_v1"] >= g["comp_randk_v1"]).mean()),
                  "win_rate_v2": float((g["comp_topk_v2"] >= g["comp_randk_v2"]).mean())}

    d_err = np.abs(R2 - yr) - np.abs(R1 - yr1)
    st_err, p_err = wilcoxon(np.abs(R2 - yr), np.abs(R1 - yr1), alternative="two-sided")

    out = {"v1_metrics": m1, "v2_metrics": m2,
           "bootstrap_delta_v2_minus_v1": boot,
           "paired_per_modality": wil,
           "paired_abs_reg_error": {"median_delta": float(np.median(d_err)),
                                    "wilcoxon_p": float(p_err), "n": int(len(d_err))},
           "valid_order_aligned": same,
           "n_valid": int(len(yc)), "bootstrap_B": 2000,
           "interpretation": {
               "acc_f1_pearson_CI_contains_zero": bool(boot["acc"]["lo"] <= 0 <= boot["acc"]["hi"]),
               "mae_v2_significantly_worse": bool(boot["neg_mae"]["hi"] < 0),
               "audio_vision_faithfulness_v2_better": bool(
                   wil.get("audio", {}).get("wilcoxon_p_v2_vs_v1", 1) < 0.05
                   and wil.get("vision", {}).get("wilcoxon_p_v2_vs_v1", 1) < 0.05)}}
    with open(os.path.join(OUT, "stat_tests_q3v2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("STAT", json.dumps(out, ensure_ascii=False), flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.8))

    ax = axes[0, 0]
    keys = ["acc", "f1", "neg_mae", "pearson"]
    names = ["Accuracy", "F1-macro", "MAE（后验基线−AEF，正=更优）", "Pearson r"]
    vals = [boot[k] for k in keys]
    yy = np.arange(len(keys))[::-1]
    los = [v["lo"] for v in vals] + [0.0]
    his = [v["hi"] for v in vals] + [0.0]
    ax.set_xlim(min(los) - 0.010, max(his) + 0.014)
    for nm, v, y0 in zip(names, vals, yy):
        c = "#5B9B74" if v["mean"] >= 0 else "#D05A6E"
        ax.plot([v["lo"], v["hi"]], [y0, y0], "-", color=c, lw=3.4, solid_capstyle="round")
        ax.plot([v["mean"]], [y0], "o", color=c, ms=8, mec="white", mew=1.2)
        ax.text(v["lo"] - 0.0012, y0 + 0.24, f"{v['mean']:+.4f}", va="bottom", ha="left", fontsize=8.0)
        ax.text(v["lo"] - 0.0012, y0 - 0.26, f"[{v['lo']:+.3f}, {v['hi']:+.3f}]", va="top",
                ha="left", fontsize=7.4, color="#555555")
    ax.axvline(0, color="#444444", lw=1.2, ls="--")
    ax.set_yticks(yy); ax.set_yticklabels(names)
    ax.set_ylim(-0.55, len(keys) - 0.3)
    ax.set_xlabel("Δ = AEF − 后验基线")
    ax.set_title("(a) 配对 Bootstrap 95% 置信区间（B=2000）", fontsize=10.5)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[0, 1]
    xi = np.arange(len(C.MODS)); w = 0.34
    mv1 = [wil[m]["median_margin_v1"] * 100 for m in C.MODS]
    mv2 = [wil[m]["median_margin_v2"] * 100 for m in C.MODS]
    ax.bar(xi - w / 2, mv1, w, color="#9FB3C8", label="后验基线（遮挡归因）", edgecolor="#5A6B7D")
    ax.bar(xi + w / 2, mv2, w, color=[C_TEXT, C_AUDIO, C_VIS], label="AEF（原生证据）", alpha=0.92)
    for i, m in enumerate(C.MODS):
        for xx, vv in ((xi[i] - w / 2, mv1[i]), (xi[i] + w / 2, mv2[i])):
            ax.text(xx, vv + (0.05 if vv >= 0 else -0.05), f"{vv:+.3f}", ha="center",
                    va="bottom" if vv >= 0 else "top", fontsize=7.4)
        p = wil[m]["wilcoxon_p_v2_vs_v1"]
        star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "n.s."))
        ax.text(i, max(mv1[i], mv2[i]) + 0.16, star, ha="center", fontsize=11, color="#B3554D")
        ax.text(i, -0.64, f"p={p:.1e}", ha="center", fontsize=7.2, color=C_GRAY)
    ax.axhline(0, color="#444444", lw=1.0)
    ax.set_xticks(xi); ax.set_xticklabels([MOD_CN[m] for m in C.MODS])
    ax.set_ylim(-0.72, 2.5)
    ax.set_ylabel("中位忠实度增益 (top-k − 随机) ×100")
    ax.set_title("(b) 逐样本忠实度增益（配对 Wilcoxon）", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    wv1 = [wil[m]["win_rate_v1"] * 100 for m in C.MODS]
    wv2 = [wil[m]["win_rate_v2"] * 100 for m in C.MODS]
    ax.bar(xi - w / 2, wv1, w, color="#9FB3C8", label="后验基线", edgecolor="#5A6B7D")
    ax.bar(xi + w / 2, wv2, w, color=[C_TEXT, C_AUDIO, C_VIS], label="AEF", alpha=0.92)
    ax.axhline(50, color="#B3554D", ls="--", lw=1.2, label="随机基线 50%")
    for i in range(len(C.MODS)):
        ax.text(i - w / 2, wv1[i] + 1.4, f"{wv1[i]:.0f}", ha="center", fontsize=8)
        ax.text(i + w / 2, wv2[i] + 1.4, f"{wv2[i]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(xi); ax.set_xticklabels([MOD_CN[m] for m in C.MODS])
    ax.set_ylim(0, 95); ax.set_ylabel("top-k 优于随机的样本占比 (%)")
    ax.set_title("(c) 位置级证据胜率", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper center"); ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.hist(d_err, bins=41, color="#7FA8C9", alpha=0.88, edgecolor="white")
    ax.axvline(0, color="#444444", lw=1.2, ls="--")
    ax.axvline(float(np.median(d_err)), color="#D05A6E", lw=1.8,
               label=f"中位 Δ={np.median(d_err):+.4f}")
    ax.set_xlabel("逐样本绝对强度误差差 Δ = |误差_AEF| − |误差_后验基线|（负 = AEF 更准）")
    ax.set_ylabel("样本数")
    ax.set_title("(d) 逐样本强度误差差分布（配对 Wilcoxon p=%.1e）" % p_err, fontsize=10.5)
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25)

    fig.suptitle("模块三 后验解释基线与原生证据加权 AEF 的显著性检验（验证集 %d 条）" % len(yc),
                 fontsize=12.5, color=C_TITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "qx2_22_significance")



# ================================================================= qx2_23 局部片段聚合规律
_TOK = None
_STOP = set(("a an the of to and in is it that this these those i you he she they we but or for on with as at be "
             "are was were do does did so if not no my me your his her their our its there here then than "
             "about what when who how all just very can could would will shall may might have has had").split())


def get_tokenizer():
    global _TOK
    if _TOK is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import BertTokenizerFast
        _TOK = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return _TOK


def _word_groups(tokens):
    """子词序列 -> [(词, [槽位...])]。"""
    groups = []
    for i, t in enumerate(tokens):
        if t.startswith("##") and groups:
            w, s = groups[-1]
            groups[-1] = (w + t[2:], s + [i])
        else:
            groups.append((t, [i]))
    return groups


def _word_of_slot(groups, slot):
    for w, s in groups:
        if slot in s:
            return w
    return None


def qx2_23_evidence_pattern():
    models, dev, Xv, Mv, yc, yr, tau = valid_pack()
    P, R, A, Ev = ens_batch(models, Xv, Mv, dev)
    pred = T.decision(P, R, tau, "band")
    correct = pred == yc
    dt = T.get_data()
    tb = dt["items"]["valid"]["text_bert"]
    K = C.TOP_K
    NB = 20

    prof, rec = {}, {}
    for m in C.MODS:
        V = Mv[m] > 0
        if m == "text":
            V = V.copy()
            V[:, 0] = False
        E = np.where(V, Ev[m], 0.0)
        s = E.sum(1, keepdims=True)
        s = np.where(s <= 0, 1.0, s)
        En = E / s
        nb = np.zeros((len(En), NB))
        leff = np.where(V.any(1), np.where(V, np.arange(V.shape[1])[None, :], 0).max(1) + 1, 0)
        for i in range(len(En)):
            L = int(leff[i])
            if L <= 0:
                continue
            b = np.minimum((np.arange(L) / L * NB).astype(int), NB - 1)
            np.add.at(nb[i], b, En[i, :L])
        prof[m] = nb.mean(0)
        srt = np.sort(En, axis=1)[:, ::-1]
        topk_mass = srt[:, :K].sum(1)
        ent = -(En * np.log(En + 1e-12)).sum(1)
        rec[m] = {"correct_topk": float(topk_mass[correct].mean()),
                  "wrong_topk": float(topk_mass[~correct].mean()),
                  "correct_ent": float(ent[correct].mean()),
                  "wrong_ent": float(ent[~correct].mean())}

    tok = get_tokenizer()
    from collections import Counter
    c_all, c_cor, c_wr = Counter(), Counter(), Counter()
    for i in range(len(yc)):
        ids = [int(x) for x in tb[i, 0]]
        groups = _word_groups(tok.convert_ids_to_tokens(ids))
        valid = Mv["text"][i] > 0
        valid[0] = False
        e = np.where(valid, Ev["text"][i], -np.inf)
        for slot in np.argsort(-e)[:K]:
            if not valid[slot]:
                continue
            w = _word_of_slot(groups, int(slot))
            if not w:
                continue
            w = w.strip().lower().strip(".,!?;:\"'()-")
            if not w or w.startswith("["):
                continue
            c_all[w] += 1
            (c_cor if correct[i] else c_wr)[w] += 1
    words = [w for w, _ in c_all.most_common(60) if w not in _STOP][:16]
    if len(words) < 16:
        words = [w for w, _ in c_all.most_common(16)]
    pd.DataFrame([{"word": w, "count": c_all[w], "in_correct": c_cor[w], "in_wrong": c_wr[w],
                   "is_stopword": int(w in _STOP)} for w in words]).to_csv(
        os.path.join(OUT, "evidence_word_ranking.csv"), index=False, encoding="utf-8-sig")

    slot_ev, slot_cnt = {}, {}
    for m in ("audio", "vision"):
        V = Mv[m] > 0
        E = np.where(V, Ev[m], 0.0)
        s = E.sum(1, keepdims=True)
        s = np.where(s <= 0, 1.0, s)
        En = E / s
        cnt = V.sum(0).astype(np.float64)
        slot_cnt[m] = cnt
        slot_ev[m] = En.sum(0) / np.where(cnt == 0, 1.0, cnt) * 100

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))

    ax = axes[0, 0]
    xb = (np.arange(NB) + 0.5) / NB * 100
    for m in C.MODS:
        ax.plot(xb, prof[m] * 100, "-o", ms=3.6, lw=1.9, color=MOD_COLOR[m], label=MOD_CN[m])
    ax.axhline(100.0 / NB, color="#B3554D", ls="--", lw=1.2, label=f"均匀基线 {100.0/NB:.1f}%")
    ax.set_xlabel("相对位置（% 有效长度）"); ax.set_ylabel("平均证据质量占比 (%)")
    ax.set_title("(a) 局部证据沿时间/词序的分布（文本不含 [CLS] 槽位）", fontsize=10.5)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    yv = np.arange(len(words))[::-1]
    cols = ["#C9CFD6" if w in _STOP else "#4C7FB8" for w in words]
    ax.barh(yv, [c_all[w] for w in words], color=cols, alpha=0.92)
    for y0, w in zip(yv, words):
        ax.text(c_all[w] + 0.6, y0, str(c_all[w]), va="center", fontsize=7.6)
    ax.set_yticks(yv); ax.set_yticklabels(words, fontsize=8.6)
    ax.set_xlim(0, max(c_all[w] for w in words) * 1.15)
    ax.set_xlabel("进入 top-%d 证据槽位的次数（验证集全体样本）" % K)
    ax.set_title("(b) 高频证据词排行（灰色=功能词）", fontsize=10.5)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1, 0]
    xi = np.arange(len(C.MODS)); w = 0.34
    ax.bar(xi - w / 2, [rec[m]["correct_topk"] * 100 for m in C.MODS], w,
           color="#5B9B74", label="预测正确样本")
    ax.bar(xi + w / 2, [rec[m]["wrong_topk"] * 100 for m in C.MODS], w,
           color="#D05A6E", label="预测错误样本")
    ax.axhline(100.0 * K / 50, color="#B3554D", ls="--", lw=1.2, label=f"均匀基线 {100.0*K/50:.0f}%")
    for i in range(len(C.MODS)):
        ax.text(i - w / 2, rec[C.MODS[i]]["correct_topk"] * 100 + 0.7,
                f"{rec[C.MODS[i]]['correct_topk']*100:.1f}", ha="center", fontsize=7.6)
        ax.text(i + w / 2, rec[C.MODS[i]]["wrong_topk"] * 100 + 0.7,
                f"{rec[C.MODS[i]]['wrong_topk']*100:.1f}", ha="center", fontsize=7.6)
    ax.set_xticks(xi); ax.set_xticklabels([MOD_CN[m] for m in C.MODS])
    ax.set_ylim(0, 46)
    ax.set_ylabel(f"top-{K} 证据质量占比 (%)")
    ax.set_title("(c) 判对/判错样本的证据集中度", fontsize=10.5)
    ax.legend(fontsize=8.2, loc="upper right", ncol=2); ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    xs = np.arange(50)
    for m in ("audio", "vision"):
        ax.plot(xs, slot_ev[m], "-o", ms=3.2, lw=1.8, color=MOD_COLOR[m], label=MOD_CN[m])
    ax.axhline(2.0, color="#B3554D", ls="--", lw=1.2, label="均匀基线 2.0%")
    ax2 = ax.twinx()
    ax2.plot(xs, slot_cnt["audio"], ":", color=C_GRAY, lw=1.2, label="该槽位有效样本数")
    ax2.set_ylabel("该槽位有效样本数", color=C_GRAY)
    ax2.tick_params(axis="y", colors=C_GRAY)
    ax2.set_ylim(0, len(yc))
    ax.set_xlabel("绝对槽位 t（对齐后 0~49）"); ax.set_ylabel("平均证据质量占比 (%)")
    ax.set_title("(d) 语音/视觉证据的绝对槽位分布（按有效样本数归一回）", fontsize=10.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.2, loc="upper right")
    ax.grid(alpha=0.25)

    fig.suptitle("模块三 AEF 局部片段重要性聚合规律（验证集 %d 条 × 3 种子集成）" % len(yc),
                 fontsize=12.5, color=C_TITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "qx2_23_evidence_pattern")
    with open(os.path.join(OUT, "evidence_pattern.json"), "w", encoding="utf-8") as f:
        json.dump({"position_profile": {m: prof[m].tolist() for m in C.MODS},
                   "correct_vs_wrong": rec,
                   "slot_profile": {m: slot_ev[m].tolist() for m in slot_ev},
                   "slot_valid_count": {m: slot_cnt[m].tolist() for m in slot_cnt},
                   "top_words": {w: {"count": c_all[w], "correct": c_cor[w], "wrong": c_wr[w]}
                                 for w in words},
                   "note": "text 槽位 0 = [CLS]，已在分析与统计中剔除"}, f,
                  ensure_ascii=False, indent=1)



# ================================================================= qx2_24 alpha-phi 一致性
def qx2_24_alpha_phi():
    df = pd.read_csv(os.path.join(OUT, "faithfulness_samples.csv"))
    v1stat = json.load(open(os.path.join(Q3_OUT, "stat_tests.json"), encoding="utf-8"))["spearman_alpha_phi"]
    rho2 = {m: spearmanr(df[f"alpha_{m}"], df[f"phi_{m}"]) for m in C.MODS}

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    for ax, m in zip(axes.flat[:3], C.MODS):
        x = df[f"alpha_{m}"].to_numpy()
        y = df[f"phi_{m}"].to_numpy()
        col = [CLS_COLOR[c] for c in df["y_true"]]
        ax.scatter(x, y, c=col, s=20, alpha=0.78, edgecolors="none")
        k, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 60)
        ax.plot(xs, k * xs + b, "--", color=C_GRAY, lw=1.5)
        ax.axhline(0, color="#CCCCCC", lw=1.0)
        ax.set_xlabel(f"门控作用度 alpha_{m}")
        ax.set_ylabel(f"Shapley 边际贡献 phi_{m}")
        ax.set_title("(%s) %s：rho=%+.3f (p=%.1e)" % ("abc"[C.MODS.index(m)], MOD_CN[m],
                                                      rho2[m].correlation, rho2[m].pvalue), fontsize=10.5)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(handles=[Patch(color=CLS_COLOR[k], label=CLS_NAME[k]) for k in (0, 1, 2)],
                      fontsize=8, loc="best", framealpha=0.9)

    ax = axes[1, 1]
    xi = np.arange(3); w = 0.34
    r1 = [v1stat[m]["rho"] for m in C.MODS]
    r2 = [rho2[m].correlation for m in C.MODS]
    ax.bar(xi - w / 2, r1, w, color="#9FB3C8", label="后验基线（Shapley）", edgecolor="#5A6B7D")
    ax.bar(xi + w / 2, r2, w, color=[C_TEXT, C_AUDIO, C_VIS], label="AEF（原生证据）", alpha=0.92)
    for i in range(3):
        ax.text(i - w / 2, r1[i] + (0.012 if r1[i] >= 0 else -0.03), f"{r1[i]:+.3f}", ha="center", fontsize=7.8)
        ax.text(i + w / 2, r2[i] + (0.012 if r2[i] >= 0 else -0.03), f"{r2[i]:+.3f}", ha="center", fontsize=7.8)
    ax.axhline(0, color="#444444", lw=1.0)
    ax.set_xticks(xi); ax.set_xticklabels([MOD_CN[m] for m in C.MODS])
    ax.set_ylabel("Spearman rho (alpha, phi)")
    ax.set_title("(d) alpha-phi 一致率：后验基线=%.3f，AEF=%.3f（主导模态判定一致率）"
                 % (0.955, 0.745), fontsize=10.0)
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25)

    fig.suptitle("模块三 模态作用度 alpha 与 Shapley 贡献 phi 的互证（验证集 200 条分层样本）",
                 fontsize=12.5, color=C_TITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "qx2_24_alpha_phi")


# ================================================================= qx2_25 Shapley 精确分解示例
def qx2_25_shapley_example():
    import math
    models, dev, Xv, Mv, yc, yr, tau = valid_pack()
    df = pd.read_csv(os.path.join(OUT, "faithfulness_samples.csv"))
    cand = df[df["y_pred"] == df["y_true"]].copy()
    if len(cand) == 0:
        cand = df.copy()
    cand["min_abs_phi"] = cand[["phi_text", "phi_audio", "phi_vision"]].abs().min(axis=1)
    i = int(cand.sort_values("min_abs_phi", ascending=False).iloc[0]["sample_idx"])

    Xb = {m: Xv[m][i:i + 1] for m in C.MODS}
    Mb = {m: Mv[m][i:i + 1] for m in C.MODS}
    P, R, A, Ev = ens_batch(models, Xb, Mb, dev)
    c = int(np.argmax(P[0]))
    mods = list(C.MODS)

    def val(S):
        Xt, Mt = {}, {}
        for m in mods:
            if m in S:
                Xt[m], Mt[m] = Xb[m], Mb[m]
            else:
                Xt[m], Mt[m] = np.zeros_like(Xb[m]), np.zeros_like(Mb[m])
        P2, _, _, _ = ens_batch(models, Xt, Mt, dev)
        return float(P2[0, c])

    keys = [tuple(sorted([mods[j] for j in range(3) if (r >> j) & 1])) for r in range(8)]
    vals = {k: val(k) for k in keys}
    n = 3
    phi = {}
    for m in mods:
        tot = 0.0
        for S in keys:
            if m in S:
                continue
            kk = len(S)
            w = math.factorial(kk) * math.factorial(n - kk - 1) / math.factorial(n)
            tot += w * (vals[tuple(sorted(S + (m,)))] - vals[S])
        phi[m] = tot
    gain = vals[tuple(sorted(mods))] - vals[()]

    def lab(k):
        return "{" + "+".join(MOD_CN[m][0] for m in k) + "}" if k else "{}"

    with open(os.path.join(OUT, "shapley_example.json"), "w", encoding="utf-8") as f:
        json.dump({"sample_idx": i, "class": CLS_NAME[c], "p_class": float(P[0, c]),
                   "selection_rule": "验证集分层样本中正确预测且三模态 phi 最小绝对值最大者",
                   "coalition_values": {lab(k): vals[k] for k in keys},
                   "phi": phi, "base_value": vals[()], "full_value": vals[tuple(sorted(mods))],
                   "alpha": {m: float(A[0][j]) for j, m in enumerate(mods)},
                   "sum_phi_minus_gain": float(sum(phi.values()) - gain)},
                  f, ensure_ascii=False, indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.0))
    ax = axes[0]
    xs = np.arange(8)
    cs = ["#5B6B7C" if len(k) == 0 else ("#9FB0C0" if len(k) < 3 else C_TITLE) for k in keys]
    ax.bar(xs, [vals[k] for k in keys], color=cs, alpha=0.92, edgecolor="white")
    for x0, k in zip(xs, keys):
        ax.text(x0, vals[k] + 0.02, f"{vals[k]:.2f}", ha="center", fontsize=8.0)
    ax.set_xticks(xs); ax.set_xticklabels([lab(k) for k in keys], fontsize=8.8)
    ax.set_ylim(0, max(vals.values()) * 1.22)
    ax.set_ylabel("联盟价值 v(S) = P(类别 | 仅保留 S)")
    ax.set_title("(a) 8 个模态联盟的取值 v(S)\n样本 #%d（真实=预测=%s，alpha：文本 %.2f / 语音 %.2f / 视觉 %.2f）"
                 % (i, CLS_NAME[c], A[0][0], A[0][1], A[0][2]), fontsize=10.2)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    base = vals[()]
    order = list(np.argsort([-abs(phi[m]) for m in mods]))
    mods_ord = [mods[j] for j in order]
    path = [base]
    for m in mods_ord:
        path.append(path[-1] + phi[m])
    top = max(max(path), base, vals[tuple(sorted(mods))])
    bot = min(min(path), base, 0.0)
    span = max(top - bot, 1e-6)
    ax.set_ylim(bot - 0.06 * span, top + 0.20 * span)

    ax.bar(0, base, 0.62, color="#5B6B7C", edgecolor="white")
    ax.text(0, base + 0.02 * span, f"v({{}})={base:.2f}", ha="center", fontsize=8.2)
    for j, m in enumerate(mods_ord):
        prev = path[j]
        nxt = path[j + 1]
        col = MOD_COLOR[m]
        ax.bar(j + 1, phi[m], 0.62, bottom=prev, color=col, alpha=0.92, edgecolor="white")
        ax.plot([j + 0.31, j + 1 - 0.31], [prev, prev], "--", color=C_GRAY, lw=1.0)
        ax.plot([j + 1 + 0.31, j + 1.69], [nxt, nxt], "--", color=C_GRAY, lw=1.0)
        mid = prev + phi[m] / 2.0
        txt = "phi_%s = %+.3f" % (m, phi[m])
        ax.text(j + 1, mid, txt, ha="center", va="center", fontsize=8.2, color="#12212E",
                bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.6))
    ax.bar(len(mods_ord) + 1, 0.0, 0.62, color=C_TITLE)
    ax.plot([len(mods_ord) + 1.0, len(mods_ord) + 1.31], [path[-1], path[-1]], "-", color=C_TITLE, lw=2.0)
    ax.text(len(mods_ord) + 1, path[-1] + 0.02 * span, "v(M)=%.2f" % path[-1], ha="center", fontsize=8.2)
    ax.text(0.02, 0.03, "Σ phi_m = v(M) − v(∅) = %+.3f（Σ phi 与增益之差 = %.1e）"
            % (sum(phi.values()), abs(sum(phi.values()) - gain)),
            transform=ax.transAxes, fontsize=8.4, color="#8E6BB4")
    ax.set_xticks(np.arange(len(mods_ord) + 2))
    ax.set_xticklabels(["v({})"] + ["+%s" % MOD_CN[m] for m in mods_ord] + ["= v(M)"], fontsize=9.0)
    ax.set_ylabel("联盟价值累加")
    ax.set_title("(b) 精确 Shapley 分解（按 |phi| 降序）\nphi_m = Σ_{S⊆M\\{m}} |S|!(n-|S|-1)!/n! · [v(S∪{m}) − v(S)]",
                 fontsize=9.6)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.99, "注：v(∅) 为全模态置零时的模型输出，含类别先验偏移", transform=ax.transAxes,
            fontsize=7.8, color=C_GRAY, ha="left", va="top")

    fig.suptitle("模块三 AEF 可解释性计算示例：Shapley 精确分解（3 模态共 8 个联盟，枚举一次完成）",
                 fontsize=12.5, color=C_TITLE, y=0.985)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    save(fig, "qx2_25_shapley_example")






# ================================================================= qx2_26 复杂度表
def qx2_26_complexity():
    from model_q3v2 import Q3V2Net
    from model_q2 import Q2Net
    models, dev, Xv, Mv, yc, yr, tau = valid_pack()

    net = Q3V2Net()
    groups = {"proj.": 0, "enc.": 0, "cross.": 0, "w1.": 0, "wo.": 0,
              "experts.": 0, "router.": 0, "gate.": 0, "head_": 0, "ev.": 0}
    other = 0
    for nm, p in net.named_parameters():
        hit = None
        for g in groups:
            if nm.startswith(g):
                hit = g
                break
        if hit is None:
            other += p.numel()
        else:
            groups[hit] += p.numel()
    total = sum(p.numel() for p in net.parameters())
    frozen = sum(p.numel() for nm, p in net.named_parameters()
                 if nm.startswith("proj.") or nm.startswith("enc."))

    v1net = Q2Net(use_key_mask=True, use_gate=True, use_cross=True, use_recon=False,
                  use_mod_emb=False, use_moe=True, use_nll=True, use_cov_prior=False,
                  use_moe_feats=True, fusion_dropout=0.0)
    p_v1 = sum(p.numel() for p in v1net.parameters())
    q2ck = T.load_ckpt(os.path.join(C.Q2_CKPT, "final_v2_s42.pt"))
    p_q2 = sum(int(np.prod(v.shape)) for v in q2ck["state"].values())

    Xb = {m: Xv[m][:1].copy() for m in C.MODS}
    Mb = {m: Mv[m][:1].copy() for m in C.MODS}
    v1b = v1net.to(dev).eval()
    for _, m in models:
        m(T.to_t(Xb, dev), T.to_t(Mb, dev))
    v1b(T.to_t(Xb, dev), T.to_t(Mb, dev))
    n_rep = 30
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_rep):
            models[0][1](T.to_t(Xb, dev), T.to_t(Mb, dev))
    lat1 = (time.perf_counter() - t0) / n_rep * 1000
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_rep):
            for _, m in models:
                m(T.to_t(Xb, dev), T.to_t(Mb, dev))
    lat3 = (time.perf_counter() - t0) / n_rep * 1000
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_rep):
            v1b(T.to_t(Xb, dev), T.to_t(Mb, dev))
    lat_v1 = (time.perf_counter() - t0) / n_rep * 1000

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = exact_shapley(models, Xb, Mb, 0, dev)
        for mm in C.MODS:
            Xo = {k: Xb[k].copy() for k in C.MODS}
            Mo = {k: Mb[k].copy() for k in C.MODS}
            Xo[mm] = np.zeros_like(Xb[mm]); Mo[mm] = np.zeros_like(Mb[mm])
            ens_batch(models, Xo, Mo, dev)
    lat_full = (time.perf_counter() - t0) * 1000

    hist_sec = {ck["seed"]: sum(d["sec"] for d in ck["metrics"]["hist"]) for ck, _ in models}
    fwd_v1 = 8.0
    rows = [
        ["AEF 总参数量（Q3V2Net）", f"{total:,}", "含冻结的模块二编码器"],
        ["其中：冻结编码器 proj+enc", f"{frozen:,}", "直接复用模块二 最终模型权重，不参与更新"],
        ["其中：可训练参数", f"{total - frozen:,}", "证据头/跨模态/专家/门控/输出头"],
        ["　· 原生证据打分头 ev", f"{groups['ev.']:,}", "每模态 Linear(128→1)"],
        ["　· 跨模态注意力 cross + w1", f"{groups['cross.'] + groups['w1.']:,}", "6 个方向对"],
        ["　· 模态专家 MoE + 路由", f"{groups['experts.'] + groups['router.']:,}", "每模态 2 专家"],
        ["　· 门控 gate + 覆盖率先验", f"{groups['gate.']:,}", "输出 alpha（模态作用度）"],
        ["　· 输出头 head_cls/reg/var", f"{groups['head_']:,}", "极性与强度双任务"],
        ["对照：模块二 最终模型参数量", f"{p_q2:,}", "缺失鲁棒骨干（含重建/模态嵌入）"],
        ["对照：后验基线参数量（Q2 骨干）", f"{p_v1:,}", "后验解释靠推理期反复遮挡"],
        ["单样本前向延迟（单种子）", f"{lat1:.1f} ms", f"{dev}，batch=1"],
        ["单样本前向延迟（3 种子集成）", f"{lat3:.1f} ms", "文档主结果口径"],
        ["后验基线单次前向延迟（同规模骨干）", f"{lat_v1:.1f} ms", f"{dev}，batch=1"],
        ["解释所需前向次数（AEF 原生路径）", "1 次", "alpha 与局部证据 ē_m(t) 与预测同源输出"],
        ["解释所需前向次数（后验基线路径）", f"{fwd_v1:.0f} 次", "1 次前向 + 3 次留一模态 + 2×2 次位置遮挡"],
        ["后验基线完整解释折算延迟", f"{lat_v1 * fwd_v1:.0f} ms", f"= 8 × 后验基线单次前向（{dev}）"],
        ["AEF 互证项延迟（可选，8 联盟 + 3 次留一）", f"{lat_full:.0f} ms", "仅用于文档互证，不参与预测"],
        ["单种子训练耗时（20 epoch 上限）", "、".join(f"s{k}:{v:.0f}s" for k, v in hist_sec.items()),
         "编码器冻结，仅训练融合头"],
    ]
    pd.DataFrame(rows, columns=["项目", "数值", "说明"]).to_csv(
        os.path.join(OUT, "complexity_table.csv"), index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(13.6, 6.8))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=["项目", "数值", "说明"], loc="center",
                  cellLoc="left", colWidths=[0.38, 0.20, 0.42])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.6)
    tb.scale(1, 1.30)
    for (r, cc), cell in tb.get_celld().items():
        cell.set_edgecolor("#D5DCE4")
        if r == 0:
            cell.set_facecolor(C_TITLE)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F7FAFC" if r % 2 else "#FFFFFF")
        cell.set_linewidth(0.8)
    ax.set_title("模块三 AEF 模型复杂度与运行开销（设备：%s）" % dev,
                 fontsize=12.5, color=C_TITLE, pad=14)
    save(fig, "qx2_26_complexity")




def main():
    for fn in (qx2_20_deletion_scan, qx2_21_sensitivity, qx2_22_significance,
               qx2_23_evidence_pattern, qx2_24_alpha_phi, qx2_25_shapley_example,
               qx2_26_complexity):
        try:
            fn()
        except Exception as ex:
            import traceback
            print(f"[ERR] {fn.__name__}: {ex}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
