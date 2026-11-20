"""Staleness / coldness of the proposed T5 eval feeds (train exploration M>=50 feeds on MSK days 20713-15) under
train cutoffs C (train days <= C), vs val / public / private (val/test: all train days). Same definitions as
dpa/validation/s1_dist.py (cold = item never a train candidate on days <= cutoff; stale = days since last train-candidate day,
warm items, clipped 20). CPU only."""
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","POLARS_MAX_THREADS"): os.environ[k]="12"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys; sys.path.insert(0,"/tmp/scratchpad/dpa/validation")
import numpy as np
from lib import meta, day, holdout_index, _HOLD, public_mask, FLAT
D0 = 20702
ftr, Mtr, offtr = meta("train")
dtr = day(ftr["timestamp"].to_numpy())
hold = holdout_index(); pool = _HOLD["pool134"]
ev = np.sort(np.concatenate([hold[dtr[hold] >= 20713], pool[(dtr[pool] >= 20713) & (dtr[pool] <= 20715)]]))
ev_h = np.sort(hold[dtr[hold] >= 20713])
fv, Mv, offv = meta("val"); ft, Mt, offt = meta("test")
pub = public_mask()
def rows(off, M, fi):
    m = M[fi]; o = off[fi]
    return np.repeat(o, m) + (np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)), m
S = {"T5": ("train", ev), "T5_holdpart": ("train", ev_h), "val": ("val", np.arange(len(Mv))),
     "public": ("test", np.where(pub)[0]), "private": ("test", np.where(~pub)[0])}
MO = {"train": (offtr, Mtr), "val": (offv, Mv), "test": (offt, Mt)}
R = {k: rows(*MO[sp], fi) for k, (sp, fi) in S.items()}
IT = {k: np.load(FLAT / f"{S[k][0]}/cand_item_id.npy", mmap_mode="r")[R[k][0]].astype(np.int64) for k in S}
Q = np.unique(np.concatenate(list(IT.values())))
tr_item = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")
bits = np.zeros(len(Q), np.int32)
for dd in np.unique(dtr):
    fi = np.where(dtr == dd)[0]
    lo, hi = offtr[fi[0]], offtr[fi[-1]] + Mtr[fi[-1]]
    it = np.unique(np.asarray(tr_item[lo:hi], np.int64))
    j = np.minimum(np.searchsorted(Q, it), len(Q) - 1); hit = Q[j] == it
    bits[j[hit]] |= (1 << int(dd - D0))
def stats(k, cutoff):
    sp, fi = S[k]; r, m = R[k]
    f = {"train": ftr, "val": fv, "test": ft}[sp]
    dfe = day(f["timestamp"].to_numpy())[fi]
    n = len(fi); fid = np.repeat(np.arange(n), m); dc = dfe[fid]
    b = bits[np.searchsorted(Q, IT[k])]
    nb_ = cutoff - D0 + 1
    mask = b & np.int32((1 << nb_) - 1)
    has = mask > 0
    last = np.full(len(b), -1, np.int64)
    last[has] = np.floor(np.log2(mask[has].astype(np.float64))).astype(np.int64) + D0
    st = np.where(has, np.minimum(dc - last, 20), 0)
    cold = np.bincount(fid, (~has).astype(float), n) / m
    stm = np.bincount(fid, st * has, n) / np.maximum(np.bincount(fid, has.astype(float), n), 1)
    gap = dfe - cutoff
    return n, cold.mean(), stm.mean(), np.percentile(stm, 50), gap.mean(), (m < 50).mean(), m.mean()
print(f"{'set':12s} {'cutoff':>7s} {'n':>6s} {'cold':>6s} {'stale_mean':>10s} {'stale_p50':>9s} {'gap_d':>6s} {'M<50':>5s} {'M':>6s}")
for k in ["val", "public", "private"]:
    print(f"{k:12s} {'all':>7s} " + " ".join(f"{v:{w}}" for v, w in zip(stats(k, 20716), ["6d", "6.3f", "10.2f", "9.2f", "6.2f", "5.2f", "6.1f"])))
for k in ["T5", "T5_holdpart"]:
    for C in [20710, 20709, 20708, 20707, 20706, 20705, 20704]:
        print(f"{k:12s} {C:7d} " + " ".join(f"{v:{w}}" for v, w in zip(stats(k, C), ["6d", "6.3f", "10.2f", "9.2f", "6.2f", "5.2f", "6.1f"])))
# training data kept under each cutoff
tot = len(dtr)
for C in [20710, 20708, 20706]:
    print(f"cutoff {C}: train feeds kept {(dtr <= C).sum():,} of {tot:,} ({(dtr <= C).mean():.3f}), train days kept {len(np.unique(dtr[dtr <= C]))}")
