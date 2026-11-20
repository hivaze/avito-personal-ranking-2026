"""why/fit F1 context (read-only): the same paired logged-training-loss comparison as f1_logs.py for the earlier single-knob variants of
the same recipe (same seeds = same batch order; kd_w 0.5 teacher_mean11 unless noted), to put the arm loss deltas on a scale.
Output: why/fit/f1_logs_ladder.txt"""
import json
import numpy as np

R = "<repo>"
FAMS = [("kd05", "d128"), ("kd05w224", "d224"), ("kw256d2", "d256 + dropout 0.2"), ("kwd2", "dropout 0.2"), ("kw_drop0", "dropout 0"),
        ("kwh6", "6 heads"), ("kw_ff4", "FFN x4"), ("kw_hl2", "2 history layers"),
        ("<repo>/lab/nn_x/w12/moe/out/logs/w5_e10_kd05w192moeffn", "arm A"), ("<repo>/lab/nn_x/w12/moe/out/logs/w5_e10_kd05w192moevo", "arm B"),
        ("<repo>/lab/nn_x/w12/moe/out/logs/w5_e10_kd05w192b3ff1", "arm C")]


def hist(p):
    return {e["step"]: e for e in json.load(open(p))["history"]}


out = []
for fam, label in FAMS:
    rows = []
    for s in range(951, 955):
        p = f"{fam}_s{s}.json" if fam.startswith("/") else f"{R}/nn_x/w5/logs/w5_e10_{fam}_s{s}.json"
        try:
            a = hist(p)
        except FileNotFoundError:
            continue
        r = hist(f"{R}/nn_x/w5/logs/w5_e10_kd05w192_s{s}.json")
        pl = [k for k in a if 40000 <= k <= 77500 and k in r]
        la = [k for k in a if 90000 <= k <= 98350 and k in r]
        rows.append([np.mean([a[k]["loss"] - r[k]["loss"] for k in pl]), np.mean([a[k]["loss"] - r[k]["loss"] for k in la]),
                     np.mean([a[k]["val_ndcg"] - r[k]["val_ndcg"] for k in la]), np.mean([a[k]["holdout_ndcg"] - r[k]["holdout_ndcg"] for k in la])])
    rows = np.array(rows)
    se = rows.std(0, ddof=1) / np.sqrt(len(rows)) if len(rows) > 1 else np.full(4, np.nan)
    m = rows.mean(0)
    out.append(f"{label:20s} n={len(rows)}  dloss plateau 40k-77.5k {m[0]:+.4f}+-{se[0]:.4f} | last 5 evals {m[1]:+.4f}+-{se[1]:.4f} | "
               f"in-training val {m[2]:+.5f}+-{se[2]:.5f} holdout {m[3]:+.5f}+-{se[3]:.5f} (last 5 evals, seed SE)")
open(f"{R}/lab/nn_x/w12/moe/why/fit/f1_logs_ladder.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
