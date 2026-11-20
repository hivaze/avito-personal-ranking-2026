"""winner_model recheck, check 3: read-out of the sub-feed TTA replay (c02) on val. Paired per feed, SE = sd / sqrt(n).
Usage: CUDA_VISIBLE_DEVICES="" python c03_eval_tta.py <variant> [<variant> ...]
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "<repo>/src")
import numpy as np
from common import FLAT, feed_meta
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

OUT = "<repo>/nn_x/w12/recheck/winner_model/out"
W = "<repo>/nn_x/w5/preds"
vs = sys.argv[1:]
fv, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy")
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), y, off, M)
z = lambda x: feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]
se = lambda d: d.std() / np.sqrt(len(d))
f = lambda d: f"{d.mean():+.5f} +- {se(d):.5f} (z {d.mean() / se(d):+.1f})"
sl = lambda d: f"M<50 {d[M < 50].mean():+.5f} | M>=50 {d[M >= 50].mean():+.5f} | M>=150 {d[M >= 150].mean():+.5f}"
SCH = ["2x4", "5x1"]

full, gpu, tta = {}, {}, {s: {} for s in SCH}
for v in vs:
    full[v] = np.load(f"{OUT}/{v}_full.npy")
    gpu[v] = np.load(f"{W}/nn_{v}_trainonly_val.npy")
    e0, eg = ev(full[v]), ev(gpu[v])
    print(f"== {v}: CPU fp32 full-feed val {e0.mean():.5f} | stored GPU bf16 {eg.mean():.5f} | fp32 - bf16 {f(e0 - eg)}; "
          f"Pearson {np.corrcoef(full[v], gpu[v])[0, 1]:.6f}")
    for s in SCH:
        p = f"{OUT}/{v}_sub{s}_z.npy"
        if not os.path.exists(p):
            continue
        tz, tr = np.load(p), np.load(f"{OUT}/{v}_sub{s}_raw.npy")
        tta[s][v] = tz
        rho = np.load(f"{OUT}/{v}_sub{s}_rho.npy")
        print(f"   scheme {s}: Spearman(full, sub) mean {np.nanmean(rho):.4f} (M<50 {np.nanmean(rho[M < 50]):.4f}, "
              f"M>=50 {np.nanmean(rho[M >= 50]):.4f}, M>=150 {np.nanmean(rho[M >= 150]):.4f})")
        d = ev(tz) - e0
        print(f"     TTA (mean sub-feed z) - full: {f(d)} | {sl(d)}")
        d = ev(tr) - e0
        print(f"     TTA (mean raw logit)  - full: {f(d)}")
        d = ev(0.5 * z(full[v]) + 0.5 * z(tz)) - e0
        print(f"     0.5 z(full) + 0.5 z(TTA) - full: {f(d)} | {sl(d)}")
        for r in range(1, 9):
            pr = f"{OUT}/{v}_sub{s}_z_r{r}.npy"
            if os.path.exists(pr):
                d = ev(np.load(pr)) - e0
                print(f"       K-curve reps {r}: TTA - full {d.mean():+.5f}")

if len(vs) > 1:
    P0 = np.mean([z(full[v]) for v in vs], 0)
    PG = np.mean([z(gpu[v]) for v in vs], 0)
    e0 = ev(P0)
    print(f"== POOL of {len(vs)} (member-equal z-mean): CPU full-feed {e0.mean():.5f} | stored GPU preds {ev(PG).mean():.5f} "
          f"| fp32 - bf16 {f(e0 - ev(PG))}")
    for s in SCH:
        if len(tta[s]) != len(vs):
            continue
        PT = np.mean([z(tta[s][v]) for v in vs], 0)
        d = ev(PT) - e0
        print(f"   scheme {s}: pool of TTA members - pool of full members: {f(d)} | {sl(d)}")
        d = ev(0.5 * z(P0) + 0.5 * z(PT)) - e0
        print(f"   scheme {s}: 0.5 z(pool full) + 0.5 z(pool TTA) - pool full: {f(d)} | {sl(d)}")
        d = ev(np.mean([0.5 * z(full[v]) + 0.5 * z(tta[s][v]) for v in vs], 0)) - e0
        print(f"   scheme {s}: member-level mix then pool - pool full: {f(d)}")
        print(f"   scheme {s}: corr(pool full, pool TTA) {np.corrcoef(P0, PT)[0, 1]:.5f}; member-pair corr full "
              f"{np.mean([np.corrcoef(z(full[a]), z(full[b]))[0, 1] for i, a in enumerate(vs) for b in vs[i + 1:]]):.4f} "
              f"vs TTA {np.mean([np.corrcoef(z(tta[s][a]), z(tta[s][b]))[0, 1] for i, a in enumerate(vs) for b in vs[i + 1:]]):.4f}")
