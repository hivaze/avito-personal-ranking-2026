"""F1/F4 (log part): paired training loss, val and holdout curves of the w12/moe arms vs the stored references (same seed = same
batch order). Logged 'loss' = mean over the 50 steps with step % 50 == 0 inside each 2,500-step eval window (main KD-mixture listwise
loss + K9 aux, train mode with dropout). Read-only. Output: why/fit/f1_logs.txt"""
import json
import numpy as np

R = "<repo>"
ARMS = {"A": "kd05w192moeffn", "B": "kd05w192moevo", "C": "kd05w192b3ff1"}
SEEDS = [951, 952, 953, 954]


def hist(path):
    h = json.load(open(path))["history"]
    return {e["step"]: e for e in h}


ref = {s: hist(f"{R}/nn_x/w5/logs/w5_e10_kd05w192_s{s}.json") for s in SEEDS}
out = []
P = out.append
windows = {"plateau 40k-77.5k": (40000, 77500), "cooldown 80k-98.35k": (80000, 98350), "last 5 evals 90k-98.35k": (90000, 98350),
           "final eval 98350": (98350, 98350)}
for a, v in ARMS.items():
    arm = {s: hist(f"{R}/lab/nn_x/w12/moe/out/logs/w5_e10_{v}_s{s}.json") for s in SEEDS}
    P(f"== arm {a} {v} vs kd05w192 (paired per seed, same batches)")
    for name, (lo, hi) in windows.items():
        rows = []
        for s in SEEDS:
            st = [k for k in sorted(arm[s]) if lo <= k <= hi and k in ref[s]]
            d = np.array([[arm[s][k][q] - ref[s][k][q] for q in ("loss", "val_ndcg", "holdout_ndcg")] for k in st])
            rl = np.mean([ref[s][k]["loss"] for k in st])
            rows.append(np.r_[d.mean(0), rl, len(st)])
        rows = np.array(rows)
        m, se = rows[:, :3].mean(0), rows[:, :3].std(0, ddof=1) / 2
        P(f"  {name:26s} n_evals/seed {int(rows[0, 4]):2d} | dloss {m[0]:+.4f}+-{se[0]:.4f} (rel {m[0] / rows[:, 3].mean() * 100:+.2f}% of ref "
          f"{rows[:, 3].mean():.4f}) | dval {m[1]:+.5f}+-{se[1]:.5f} | dhold {m[2]:+.5f}+-{se[2]:.5f} | per-seed dloss "
          + " ".join(f"{x:+.4f}" for x in rows[:, 0]))
    # F4: slope over the last 20k steps (evals at >= 78,680 cooldown) and the gain from 77.5k to the end
    for q in ("val_ndcg", "holdout_ndcg"):
        sl_a, sl_r, g_a, g_r, late_a, late_r = [], [], [], [], [], []
        for s in SEEDS:
            for H, L1, L2, L3 in ((arm[s], sl_a, g_a, late_a), (ref[s], sl_r, g_r, late_r)):
                st = np.array([k for k in sorted(H) if k >= 80000])
                yv = np.array([H[k][q] for k in st])
                L1.append(np.polyfit(st / 1e4, yv, 1)[0])
                L2.append(H[98350][q] - H[77500][q])
                st2 = np.array([k for k in sorted(H) if k >= 90000])
                L3.append(np.polyfit(st2 / 1e4, np.array([H[k][q] for k in st2]), 1)[0])
        d = np.array(sl_a) - np.array(sl_r)
        dg = np.array(g_a) - np.array(g_r)
        dl = np.array(late_a) - np.array(late_r)
        P(f"  F4 {q:12s}: slope 80k-98.35k per 10k steps arm {np.mean(sl_a):+.5f} ref {np.mean(sl_r):+.5f} diff {d.mean():+.5f}+-{d.std(ddof=1) / 2:.5f}"
          f" | slope 90k-98.35k arm {np.mean(late_a):+.5f} ref {np.mean(late_r):+.5f} diff {dl.mean():+.5f}+-{dl.std(ddof=1) / 2:.5f}"
          f" | gain 77.5k->end arm {np.mean(g_a):+.5f} ref {np.mean(g_r):+.5f} diff {dg.mean():+.5f}+-{dg.std(ddof=1) / 2:.5f}")
    # full curve of mean paired deltas (every 5th eval) for a readable trajectory
    traj = []
    for k in sorted(ref[951]):
        if k % 10000 == 0 or k in (77500, 98350, 2500, 5000):
            d = np.array([[arm[s][k][q] - ref[s][k][q] for q in ("loss", "val_ndcg", "holdout_ndcg")] for s in SEEDS])
            traj.append(f"{k}:{d[:, 0].mean():+.4f}/{d[:, 1].mean():+.5f}/{d[:, 2].mean():+.5f}")
    P("  trajectory step:dloss/dval/dhold  " + "  ".join(traj))
    P(f"  best step (holdout-selected) arm {[json.load(open(f'{R}/lab/nn_x/w12/moe/out/logs/w5_e10_{v}_s{s}.json'))['best_step'] for s in SEEDS]}"
      f" ref {[json.load(open(f'{R}/nn_x/w5/logs/w5_e10_kd05w192_s{s}.json'))['best_step'] for s in SEEDS]}")
open(f"{R}/lab/nn_x/w12/moe/why/fit/f1_logs.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
