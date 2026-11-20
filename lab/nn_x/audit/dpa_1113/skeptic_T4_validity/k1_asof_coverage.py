"""Skeptic check of T4: what an as-of (cutoff = feed day - 7) rebuild of the holdout popularity inputs would look like.
CPU only, read-only on data/; writes nothing except stdout. Exact for it_hist_all as-of (snapshots of feeds with day < d-7
cannot be same-user LATER snapshots, so LUO removes nothing) and for 'item was a train candidate before d-7' (tte/xp cold).
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import numpy as np
import polars as pl
R = "<repo>"
FL = f"{R}/data/flat/train"
t0 = time.time()
f = pl.read_parquet(f"{FL}/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
day = (f["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
pr = np.load(f"{R}/eda/labels_position/out/train_prandom.npz")
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[M < 50] = 0.0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
Mh, offh = np.load(f"{R}/preds/holdout/holdout_meta.npy")
assert (M[hold] == Mh).all()
hd = day[hold]
print("train feeds per day:", dict(zip(*np.unique(day, return_counts=True))))
print("holdout feeds per day:", dict(zip(*np.unique(hd, return_counts=True))))
ntr_before = np.array([(day < d - 7).sum() for d in hd])
print(f"holdout feeds with ZERO train feeds of day < d-7: {(ntr_before == 0).sum()} / 30000 = {(ntr_before == 0).mean():.4f}")
print("hist arrays present:", sorted(x for x in os.listdir(FL) if x.startswith("hist_")), "(no event timestamp column)")
# holdout candidate rows
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off[hold], M[hold])])
rday = np.repeat(hd, M[hold])
ci = np.load(f"{FL}/cand_item_id.npy", mmap_mode="r")
hitem = np.asarray(ci[rows])
cur = np.load(f"{R}/data/feats/train/it_hist_all_luo.npy", mmap_mode="r")
cur_h = np.asarray(cur[rows])
print(f"current holdout luo0 (it_hist_all_luo==0, row share) {np.mean(cur_h == 0):.4f}; feed-mean {np.mean(np.add.reduceat((cur_h == 0).astype(float), np.r_[0, np.cumsum(M[hold])[:-1]]) / M[hold]):.4f}")
NI = int(max(hitem.max(), 111_400_000)) + 1
# cumulative seen-bitmaps by cutoff day (history snapshots and train candidates)
hf = np.load(f"{FL}/hist_fidx.npy", mmap_mode="r"); hi = np.load(f"{FL}/hist_item_id.npy", mmap_mode="r")
cf = np.load(f"{FL}/cand_fidx.npy", mmap_mode="r")
cuts = sorted(set((hd - 7).tolist()))
print("cutoffs needed (day < c):", cuts)
seen_h = np.zeros(NI, bool); seen_c = np.zeros(NI, bool)
res = {}
CH = 50_000_000
hday_cache = None
for c in cuts:
    pass
# one pass per array: compute per-row feed day, then for each cutoff mark items with day < c (cumulative)
def first_day(item_arr, fidx_arr):
    fd = np.full(NI, 99999, np.int32)
    for s in range(0, len(item_arr), CH):
        it = np.asarray(item_arr[s:s + CH]); dd = day[np.asarray(fidx_arr[s:s + CH])].astype(np.int32)
        np.minimum.at(fd, it, dd) if False else None
        # faster: sort-based min
        o = np.lexsort((dd, it)); it = it[o]; dd = dd[o]
        u, i0 = np.unique(it, return_index=True)
        fd[u] = np.minimum(fd[u], dd[i0])
    return fd
fdh = first_day(hi, hf); print(f"hist first-day done {time.time()-t0:.0f}s", flush=True)
fdc = first_day(ci, cf); print(f"cand first-day done {time.time()-t0:.0f}s", flush=True)
coldH = fdh[hitem] >= (rday - 7)   # item in no snapshot of a feed with day < d-7
coldC = fdc[hitem] >= (rday - 7)   # item was no train candidate on a day < d-7 (tte / xp as-of = 0)
offs = np.r_[0, np.cumsum(M[hold])[:-1]]
def fm(x):
    return np.add.reduceat(x.astype(float), offs) / M[hold]
fH, fC = fm(coldH), fm(coldC)
print(f"\nAS-OF (d-7) holdout: luo0 feed-mean {fH.mean():.4f} | cold_tte feed-mean {fC.mean():.4f}   (public: luo0 0.4292, cold_tte 0.4385; val 0.3808 / 0.3480)")
print("per day: n | as-of luo0 | as-of cold_tte | current luo0")
curz = fm(cur_h == 0)
for d in np.unique(hd):
    m = hd == d
    print(f"  {d} n {m.sum():5d} | {fH[m].mean():.4f} | {fC[m].mean():.4f} | {curz[m].mean():.4f}")
late = hd >= 20710
print(f"\nfeeds with any as-of history (day >= 20710): {late.sum()}; their as-of luo0 {fH[late].mean():.4f}, cold_tte {fC[late].mean():.4f}")
print(f"feeds whose as-of luo0 is within public's 10-90% band [{0.4292-0.15:.2f},{0.4292+0.15:.2f}] (approx): {np.mean((fH > 0.28) & (fH < 0.58)):.4f}")
np.save("/tmp/scratchpad/dpa/skeptic_T4_validity/asof_feed.npy", np.stack([hd, fH, fC, curz]))
print(f"done {time.time()-t0:.0f}s")
