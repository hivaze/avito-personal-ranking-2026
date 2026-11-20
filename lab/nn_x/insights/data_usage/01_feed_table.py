"""Per-train-feed table for the data-usage / position-bias analysis (CPU only, read-only on all inputs).

Writes out/train_feeds.parquet: one row per train feed (3,151,208) with
  fidx, M, H, user_id, ts, day (MSK date index), hour, npos, prand_raw (EDA posterior, NaN where unscored),
  prand_nn (train.py rule: NaN->0, M<50 -> 0), cls (0 production, 1 pool-eligible p>0.5, 2 holdout),
  gaps (missing positions below max pos), gap_share, minpos/maxpos of positives (shown pos), first-positive within-feed
  rank (r0 = 0-based order of shown positions), last-positive rank-from-end (e0), pos_last (any positive is the last
  shown item), pos_first (any positive is the first shown item), mean normalised rank of positives (nr = r/(M-1)),
  cat of the first positive, ncat (distinct categories in feed), cat_share (share of feed candidates in the positive's
  category), coldpos (share of positives with it_hist_all_luo == 0), coldshare (share of feed candidates cold),
  user_n_train (train feeds of this user), user_in_valtest.
"""
import os
import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path("<repo>")
OUT = ROOT / "nn_x/insights/data_usage/out"
OUT.mkdir(parents=True, exist_ok=True)
FL = ROOT / "data/flat/train"

f = pl.read_parquet(FL / "feeds.parquet").sort("fidx")
n = f.height
M = f["M"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64)
assert (off[1:] == off[:-1] + M[:-1]).all()
N = int(off[-1] + M[-1])
fid = np.repeat(np.arange(n, dtype=np.int64), M)
pos = np.load(FL / "cand_pos.npy").astype(np.int64)
y = np.load(FL / "cand_target.npy").astype(np.int64)
cat = np.load(FL / "cand_category_id.npy").astype(np.int64)
print("loaded", N, flush=True)

# within-feed rank of the shown position
order = np.lexsort((pos, fid))
rank = np.empty(N, np.int64)
rank[order] = np.arange(N) - off[fid[order]]
rkend = M[fid] - 1 - rank
del order

npos = np.bincount(fid, weights=y, minlength=n)
maxpos = np.full(n, -1, np.int64)
np.maximum.at(maxpos, fid, pos)
gaps = maxpos + 1 - M  # missing positions below max pos (no duplicates, EDA 2.3)
yi = y > 0
pf = fid[yi]
minpos_p = np.full(n, 10 ** 6, np.int64); np.minimum.at(minpos_p, pf, pos[yi])
maxpos_p = np.full(n, -1, np.int64); np.maximum.at(maxpos_p, pf, pos[yi])
r0 = np.full(n, 10 ** 6, np.int64); np.minimum.at(r0, pf, rank[yi])
e0 = np.full(n, 10 ** 6, np.int64); np.minimum.at(e0, pf, rkend[yi])
nr_sum = np.bincount(pf, weights=rank[yi] / np.maximum(M[pf] - 1, 1), minlength=n)
pos_last = np.bincount(pf, weights=(rkend[yi] == 0), minlength=n) > 0
pos_first = np.bincount(pf, weights=(rank[yi] == 0), minlength=n) > 0
n_top3 = np.bincount(pf, weights=(pos[yi] < 3), minlength=n)
n_last = np.bincount(pf, weights=(rkend[yi] == 0), minlength=n)
# category of the first (lowest-rank) positive
first_idx = np.full(n, -1, np.int64)
cand_r = np.where(yi, rank, 10 ** 7)
o2 = np.lexsort((cand_r, fid))
firsts = o2[off]  # per feed, candidate with the smallest rank among positives (if any positive)
has = npos > 0
pcat = np.where(has, cat[firsts], -1)
# distinct categories per feed and share of feed candidates in the positive's category
o3 = np.lexsort((cat, fid))
cs, fs = cat[o3], fid[o3]
newgrp = np.ones(N, bool); newgrp[1:] = (cs[1:] != cs[:-1]) | (fs[1:] != fs[:-1])
ncat = np.bincount(fs[newgrp], minlength=n)
same = (cat == pcat[fid])
cat_share = np.bincount(fid, weights=same, minlength=n) / M
del o2, o3, cs, fs, newgrp, same, cand_r
# coldness (leave-user-out item history count == 0)
luo = np.load(ROOT / "data/feats/train/it_hist_all_luo.npy", mmap_mode="r")
cold = np.asarray(luo) == 0
coldshare = np.bincount(fid, weights=cold, minlength=n) / M
coldpos = np.bincount(pf, weights=cold[yi], minlength=n) / np.maximum(npos, 1)
del cold, luo
# p_random
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(n)).all()
prand_raw = pr["p_random"].astype(np.float32)
prand_nn = np.nan_to_num(prand_raw, nan=0.0); prand_nn[M < 50] = 0.0
cand = np.where((prand_nn > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
cls = np.where(prand_nn > 0.5, 1, 0); cls[hold] = 2
# user stats
uid = f["user_id"].to_numpy()
uc = np.unique(uid, return_counts=True)
user_n = uc[1][np.searchsorted(uc[0], uid)]
vt = np.unique(np.concatenate([pl.read_parquet(ROOT / f"data/flat/{s}/feeds.parquet")["user_id"].to_numpy() for s in ("val", "test")]))
in_vt = np.isin(uid, vt) & (uid != 0)
ts = f["timestamp"].to_numpy().astype(np.int64) + 3 * 3600
day = (ts // 86400) - (ts.min() // 86400)
hour = (ts % 86400) // 3600
tab = pl.DataFrame(dict(
    fidx=np.arange(n, dtype=np.int32), M=M.astype(np.int32), H=f["H"].to_numpy(), user_id=uid, ts=f["timestamp"].to_numpy(),
    day=day.astype(np.int16), hour=hour.astype(np.int8), npos=npos.astype(np.int16), prand_raw=prand_raw, prand_nn=prand_nn,
    cls=cls.astype(np.int8), gaps=gaps.astype(np.int32), gap_share=(gaps / (gaps + M)).astype(np.float32),
    minpos_p=minpos_p.astype(np.int32), maxpos_p=maxpos_p.astype(np.int32), r0=r0.astype(np.int32), e0=e0.astype(np.int32),
    nr_mean=(nr_sum / np.maximum(npos, 1)).astype(np.float32), pos_last=pos_last, pos_first=pos_first,
    n_top3=n_top3.astype(np.int16), n_last=n_last.astype(np.int16), pcat=pcat.astype(np.int32), ncat=ncat.astype(np.int16),
    cat_share=cat_share.astype(np.float32), coldpos=coldpos.astype(np.float32), coldshare=coldshare.astype(np.float32),
    user_n_train=user_n.astype(np.int32), user_in_valtest=in_vt))
tab.write_parquet(OUT / "train_feeds.parquet")
print(tab.describe())
# candidate-level position table for positives (for rate curves): rank, rkend, M, cls of every positive
np.savez(OUT / "train_pos_cands.npz", fidx=pf.astype(np.int32), rank=rank[yi].astype(np.int16), rkend=rkend[yi].astype(np.int16),
         pos=pos[yi].astype(np.int16), cat=cat[yi].astype(np.int32))
# counts of all candidates by (cls, rank bucket) for rate curves
rb = np.minimum(rank, 299); eb = np.minimum(rkend, 299)
c_all = np.zeros((3, 300), np.int64); c_pos = np.zeros((3, 300), np.int64)
e_all = np.zeros((3, 300), np.int64); e_pos = np.zeros((3, 300), np.int64)
cf = cls[fid]
for k in range(3):
    mk = cf == k
    c_all[k] = np.bincount(rb[mk], minlength=300); c_pos[k] = np.bincount(rb[mk & yi], minlength=300)
    e_all[k] = np.bincount(eb[mk], minlength=300); e_pos[k] = np.bincount(eb[mk & yi], minlength=300)
np.savez(OUT / "rate_curves.npz", c_all=c_all, c_pos=c_pos, e_all=e_all, e_pos=e_pos)
print("done", flush=True)
