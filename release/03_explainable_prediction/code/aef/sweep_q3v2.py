# -*- coding: utf-8 -*-
"""模块三 v2 敏感性分析：学习率 x 权重衰减 网格（单种子，验证集）。不覆盖已有权重。"""
import os, json, time
import config_q3v2 as C
import train_q3v2 as TR

LRS = [5e-5, 1e-4, 2e-4]
WDS = [1e-3, 1e-2, 5e-2]
SEED = 123          # 专用种子，避免覆盖主模型的 42/2024/7
EPOCHS = 12

_orig_lr_schedule = TR.lr_schedule
_orig_save = TR.save_ckpt


def main():
    rows = []
    TR.save_ckpt = lambda path, obj: None          # 不落盘，保护主权重
    for lr in LRS:
        for wd in WDS:
            TR.C.LR = lr
            TR.C.WD = wd
            TR.lr_schedule = (lambda base: (lambda step, warm, total: _orig_lr_schedule(step, warm, total, base=base)))(lr)
            t0 = time.time()
            r = TR.train_one(SEED, epochs=EPOCHS, verbose=False)
            row = {"lr": lr, "wd": wd, "seed": SEED,
                   "valid_acc": r["valid_clean"]["acc"],
                   "valid_f1": r["valid_clean"]["f1_macro"],
                   "valid_mae": r["valid_clean"]["mae"],
                   "valid_pearson": r["valid_clean"]["pearson"],
                   "best_epoch": r["best_epoch"],
                   "train_sec": round(time.time() - t0, 1)}
            rows.append(row)
            print("ROW", json.dumps(row), flush=True)
    with open(os.path.join(C.Q3V2_OUT, "sensitivity_grid.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "lrs": LRS, "wds": WDS, "seed": SEED, "epochs": EPOCHS}, f, ensure_ascii=False, indent=1)
    print("SWEEP DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
