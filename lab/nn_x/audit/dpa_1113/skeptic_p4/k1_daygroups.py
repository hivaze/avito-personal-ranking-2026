"""Skeptic P4 (FPCD-lite) check 1: by holdout day group (E = Sep 6-8, L1 = Sep 9-14, L2 = Sep 17-19 MSK):
(a) popularity advantage base - xdrop_pop (4 seed-paired e10 members), (b) past cooldown-data arms vs m02 (roots 101-104):
rec33 / rho70 / hcap, and m02 - c0 (the pool-mix lever), (c) train-feed and pool-eligible composition by day and the
repetition factor implied by dropping Sep 6-8 from the cooldown. CPU only; reads saved preds only."""
import os, sys
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
ROOT = "<repo>"
M, off = np.load(f"{ROOT}/preds/holdout/holdout_meta.npy")
y = np.load(f"{ROOT}/preds/holdout/holdout_y.npy").astype(np.float64)
tr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp", "M"])
pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
trM = tr["M"].to_numpy(); prM = pr.copy(); prM[trM < 50] = 0.0
cand = np.where((prM > 0.9) & (trM >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
assert (trM[hold] == M).all()
ts_all = tr["timestamp"].to_numpy().astype(np.int64)
day_all = (ts_all + 10800) // 86400; d0 = day_all.min(); day_all = day_all - d0
day = day_all[hold]
G = {"E Sep6-8": day <= 2, "L1 Sep9-14": (day >= 3) & (day <= 8), "L2 Sep17-19": day >= 11}
print("holdout day groups:", {k: int(v.sum()) for k, v in G.items()}, "days present", np.unique(day))


def nd(path):
    return ndcg_per_feed(np.load(path), y, off, M)


def rep(name, d):
    s = f"{name:34s} all {d.mean():+.5f} ({d.std()/np.sqrt(len(d)):.5f})"
    for k, g in G.items():
        s += f" | {k} {d[g].mean():+.5f} ({d[g].std()/np.sqrt(g.sum()):.5f})"
    e, l = d[G["E Sep6-8"]], d[~G["E Sep6-8"]]
    s += f" | E-L {e.mean()-l.mean():+.5f} ({np.sqrt(e.var()/len(e)+l.var()/len(l)):.5f})"
    l1, l2 = d[G["L1 Sep9-14"]], d[G["L2 Sep17-19"]]
    s += f" | E-L1 {e.mean()-l1.mean():+.5f} ({np.sqrt(e.var()/len(e)+l1.var()/len(l1)):.5f})"
    print(s)


print("\n== (a) popularity advantage, base minus alt (mean over seeds 951-954, per feed) ==")
for alt in ["abl_e10_xdrop_pop", "abl_e10_pop1"]:
    d = np.mean([nd(f"{ROOT}/preds/holdout/nn_w5_e10_base_s{s}_holdout.npy") - nd(f"{ROOT}/preds/holdout/nn_{alt}_s{s}_holdout.npy")
                 for s in (951, 952, 953, 954)], 0)
    rep(f"base - {alt}", d)

print("\n== (b) cooldown-data arms vs m02 (roots 101-104, per-feed mean over roots) ==")
H6 = f"{ROOT}/nn_x/r6/preds/holdout"
ref = {r: nd(f"{H6}/nn_r6_cd_m02_r{r}_holdout.npy") for r in (101, 102, 103, 104)}
for arm in ["rec33", "rho70", "hcap", "m02_cs813"]:
    if arm == "m02_cs813":
        d = np.mean([nd(f"{H6}/nn_r6_cd_m02_r{r}_cs813_holdout.npy") - ref[r] for r in (101, 102, 103, 104)], 0)
    else:
        d = np.mean([nd(f"{H6}/nn_r6_cd_{arm}_r{r}_holdout.npy") - ref[r] for r in (101, 102, 103, 104)], 0)
    rep(f"{arm} - m02", d)
d = np.mean([ref[r] - nd(f"{ROOT}/preds/holdout/nn_r6_cd_c0_r{r}_holdout.npy") for r in (101, 102, 103, 104)], 0)
rep("m02 - c0 (pool-mix lever)", d)

print("\n== (c) train composition by Moscow day (from Sep 6 = day 0) ==")
n = len(ts_all); pool = (pr > 0.5)
ishold = np.zeros(n, bool); ishold[hold] = True
for dd in np.unique(day_all):
    m = day_all == dd
    print(f"  d{dd:2d}: feeds {m.mean():.4f}  pool-eligible(p>0.5, non-hold) share of all pool {(m & pool & ~ishold).sum()/(pool & ~ishold).sum():.4f}"
          f"  holdout {np.mean(day == dd):.4f}")
e = day_all <= 2
print(f"Sep 6-8 share: all train feeds {e.mean():.4f}; non-holdout train feeds {(e & ~ishold).sum()/(~ishold).sum():.4f}; "
      f"pool-eligible non-hold {(e & pool & ~ishold).sum()/(pool & ~ishold).sum():.4f}; rows {np.sum(trM[e])/trM.sum():.4f}")
keep = 1 - (e & ~ishold).sum() / (~ishold).sum()
print(f"if cooldown steps are kept and Sep 6-8 batches are redrawn: passes per remaining feed x{1/keep:.3f} (2.0 -> {2/keep:.2f} epochs-equivalent)")
