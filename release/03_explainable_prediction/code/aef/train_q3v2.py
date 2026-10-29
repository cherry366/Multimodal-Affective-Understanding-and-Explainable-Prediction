# -*- coding: utf-8 -*-
"""模块三 新方案（v2）训练：冻结 Q2 编码器 + 训练证据加权融合头。"""
import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config_q3v2 as C
import data_q2 as D
from model_q3v2 import Q3V2Net
from train_q2 import (get_data, device_of, decision, metrics_of, best_tau, predict,
                      save_ckpt, load_ckpt, lr_schedule)


def build_v2(dev):
    m = Q3V2Net().to(dev)
    # 用 final_v2 权重做热启动（strict=False 忽略 rec_* 与新增 ev.*）
    ck = load_ckpt(os.path.join(C.Q2_CKPT, "final_v2_s42.pt"))
    m.load_state_dict(ck["state"], strict=False)
    if C.FREEZE_ENCODER:
        for name, p in m.named_parameters():
            if name.startswith("proj.") or name.startswith("enc."):
                p.requires_grad = False
    return m


@torch.no_grad()
def eval_clean(model, dt, split, device):
    X, M = dt["X"][split], dt["M"][split]
    idx = np.arange(X["text"].shape[0])
    Xb = {m: X[m][idx] for m in C.MODS}
    Mb = {m: M[m][idx] for m in C.MODS}
    y_cls = dt["items"][split]["y_cls"][idx]
    y_reg = dt["items"][split]["y_reg"][idx]
    p, r, _, _ = predict(model, Xb, Mb, device)
    return p, r, y_cls, y_reg


def train_one(seed, epochs=None, verbose=True):
    dt = get_data()
    device = device_of()
    epochs = epochs or C.EPOCHS
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_v2(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=C.LR, weight_decay=C.WD)

    y_tr = dt["items"]["train"]["y_cls"]
    if C.CLS_WEIGHTED:
        cnt = np.bincount(y_tr, minlength=3).astype(np.float64)
        w = cnt.sum() / (3.0 * np.maximum(cnt, 1))
        cls_w = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        cls_w = None

    rng = np.random.default_rng(seed)
    N = dt["X"]["train"]["text"].shape[0]
    Xtr, Mtr = dt["X"]["train"], dt["M"]["train"]
    y_reg_tr = dt["items"]["train"]["y_reg"]
    hist, best = [], {"score": -9, "state": None, "epoch": -1, "tau": 0.0, "mode": "band"}

    steps_per_epoch = int(math.ceil(N / C.BATCH))
    total_steps = epochs * steps_per_epoch
    warm_steps = C.WARMUP_EPOCHS * steps_per_epoch
    global_step = 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(N)
        tot, nb = 0.0, 0
        t0 = time.time()
        for i in range(0, N, C.BATCH):
            for g in opt.param_groups:
                g["lr"] = lr_schedule(global_step, warm_steps, total_steps)
            idx = perm[i:i + C.BATCH]
            Xb = {m: torch.tensor(Xtr[m][idx], device=device) for m in C.MODS}
            Mb = {m: torch.tensor(Mtr[m][idx], device=device) for m in C.MODS}
            yc = torch.tensor(y_tr[idx], device=device)
            yr = torch.tensor(y_reg_tr[idx], device=device)
            out = model(Xb, Mb)
            loss = F.cross_entropy(out["logits"], yc, weight=cls_w)
            loss = loss + C.LAMBDA_REG * F.smooth_l1_loss(out["reg"], yr, beta=1.0)
            if "log_var" in out:
                nll = 0.5 * (((yr - out["reg"].detach()) ** 2) * torch.exp(-out["log_var"]) + out["log_var"]).mean()
                loss = loss + C.LAMBDA_NLL * nll
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            global_step += 1
            tot += float(loss.item()); nb += 1

        P, R, yc_v, yr_v = eval_clean(model, dt, "valid", device)
        f1a, tau, tmode = best_tau(P, R, yc_v)
        m = metrics_of(P, R, yc_v, yr_v, tau, tmode)
        score = m["f1_macro"] + 0.2 * max(0.0, m["pearson"]) - C.SCORE_MAE_W * m["mae"]
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), "val": m, "tau": tau,
                     "tau_mode": tmode, "score": score, "sec": time.time() - t0})
        if verbose:
            print(f"[q3v2 s{seed}] ep{ep:02d} loss={tot/max(nb,1):.4f} "
                  f"F1={m['f1_macro']:.4f} Acc={m['acc']:.4f} MAE={m['mae']:.4f} r={m['pearson']:.4f} "
                  f"tau={tau:.2f} {time.time()-t0:.1f}s", flush=True)
        if ep >= C.MIN_EPOCHS and score > best["score"]:
            best = {"score": score, "epoch": ep, "tau": tau, "mode": tmode,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif ep >= C.MIN_EPOCHS and ep - best["epoch"] >= C.PATIENCE:
            if verbose:
                print(f"[q3v2 s{seed}] early stop @ ep{ep} (best ep{best['epoch']})", flush=True)
            break

    if best["state"] is None:
        best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best["state"])
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    res = {"variant": "q3v2", "seed": seed, "best_epoch": best["epoch"], "tau": best["tau"],
           "tau_mode": best["mode"], "hist": hist}
    for split in ("valid", "test"):
        P, R, yc_, yr_ = eval_clean(model, dt, split, device)
        res[f"{split}_clean"] = metrics_of(P, R, yc_, yr_, best["tau"], best["mode"])
    res["ckpt"] = os.path.join(C.Q3V2_CKPT_OUT, f"q3v2_s{seed}.pt")
    save_ckpt(res["ckpt"], {"state": final_state, "seed": seed, "tau": best["tau"],
                            "tau_mode": best["mode"], "variant": "q3v2", "metrics": res})
    if verbose:
        print(f"[q3v2 s{seed}] DONE valid={json.dumps(res['valid_clean'])} test={json.dumps(res['test_clean'])}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    seeds = [args.seed] if args.seed is not None else C.SEEDS
    rows = []
    for s in seeds:
        r = train_one(s, epochs=args.epochs)
        rows.append({k: r[k] for k in ("variant", "seed", "best_epoch", "tau", "tau_mode")} |
                    {f"valid_clean_{k}": v for k, v in r["valid_clean"].items()} |
                    {f"test_clean_{k}": v for k, v in r["test_clean"].items()})
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(C.Q3V2_OUT, "train_summary.csv"), index=False, encoding="utf-8-sig")
    json.dump({"rows": rows}, open(os.path.join(C.Q3V2_OUT, "train_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("q3v2 training done:", len(rows))
