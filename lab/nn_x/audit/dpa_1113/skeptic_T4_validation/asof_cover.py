"""Skeptic check for T4 (stale-holdout counterfactual): what does an as-of popularity input with cutoff
'history snapshots of feeds with day < d - L' look like on the 30K holdout feeds, vs public / private / stale-val?
CPU only; read-only on data/; writes nothing but stdout."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time
import numpy as np, polars as pl
sys.path.insert(0, "/tmp/scratchpad/dpa/validation")
import lib
t0 = time.time()
def pr(*a): print(*a, flush=True)
FLAT = lib.FLAT
NI = 111_400_000
ftr, Mtr, offtr = lib.meta("train")
trday = lib.day(ftr["timestamp"].to_numpy())
pr("train feed days", int(trday.min()), int(trday.max()), np.unique(trday, return_counts=True))
hf = np.load(FLAT / "train/hist_fidx.npy", mmap_mode="r")
hi = np.load(FLAT / "train/hist_item_id.npy", mmap_mode="r")
first = np.full(NI, 99999, np.int32)
CH = 50_000_000
for s in range(0, len(hf), CH):
    d = trday[np.asarray(hf[s:s + CH])].astype(np.int32)
    it = np.asarray(hi[s:s + CH])
    np.minimum.at(first, it, d) if False else None
    # faster than minimum.at: sort by item then take min via lexsort-free approach per chunk
    o = np.lexsort((d, it)); it2 = it[o]; d2 = d[o]
    u, idx = np.unique(it2, return_index=True)
    first[u] = np.minimum(first[u], d2[idx])
pr(f"first snapshot day per item done ({time.time()-t0:.0f}s); items with any train snapshot {(first < 99999).sum()}")
# also first train-CANDIDATE day (tte-style coldness)
cf = np.load(FLAT / "train/cand_fidx.npy", mmap_mode="r"); ci = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")
firstc = np.full(NI, 99999, np.int32)
for s in range(0, len(cf), CH):
    d = trday[np.asarray(cf[s:s + CH])].astype(np.int32); it = np.asarray(ci[s:s + CH])
    o = np.lexsort((d, it)); it2 = it[o]; d2 = d[o]
    u, idx = np.unique(it2, return_index=True)
    firstc[u] = np.minimum(firstc[u], d2[idx])
pr(f"first candidate day per item done ({time.time()-t0:.0f}s)")

hold = lib.holdout_index(); Mh = lib._HOLD["M"]
rows = np.concatenate([np.arange(offtr[i], offtr[i] + Mtr[i]) for i in hold])
assert len(rows) == Mh.sum()
hitem = np.asarray(ci[rows]) if False else np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")[rows]
hday = np.repeat(trday[hold], Mh).astype(np.int32)
fid = np.repeat(np.arange(len(hold)), Mh)
def fmean(x, fid, n, M):
    return np.bincount(fid, x.astype(np.float64), n) / M
live_luo = np.load("<repo>/nn/feat_cache/train/it_hist_all_luo.npy", mmap_mode="r")[rows] if os.path.exists("<repo>/nn/feat_cache/train/it_hist_all_luo.npy") else None
if live_luo is not None:
    pr("holdout live luo0 (feed mean)", round(fmean(live_luo == 0, fid, len(hold), Mh).mean(), 4))
hd = trday[hold]
pr("\n== holdout as-of zero share (item in no train snapshot of a feed with day < d - L), feed-mean ==")
pr("L | luo0 all | feeds with 0 history days (d-L <= first train day) | luo0 on feeds with >=1 history day (n) | history days mean")
for L in (1, 3, 5, 7):
    cut = hday - L
    z = first[hitem] >= cut
    fz = fmean(z, fid, len(hold), Mh)
    nod = (hd - L) <= trday.min()
    ndays = np.clip(hd - L - trday.min(), 0, None)
    pr(f"L={L}: luo0 {fz.mean():.4f} | no-history feeds {nod.sum()} ({nod.mean():.3f}) | luo0 rest {fz[~nod].mean() if (~nod).any() else float('nan'):.4f} (n {(~nod).sum()}) | hist days mean {ndays.mean():.2f}")
pr("\nper holdout day at L=7: n, as-of luo0, live luo0, history days available")
z7 = first[hitem] >= (hday - 7); fz7 = fmean(z7, fid, len(hold), Mh)
fl = fmean(live_luo == 0, fid, len(hold), Mh) if live_luo is not None else np.full(len(hold), np.nan)
for dd in np.unique(hd):
    m = hd == dd
    pr(f"  day {dd}: n {m.sum():5d} as-of luo0 {fz7[m].mean():.3f} live {fl[m].mean():.3f} hist days {max(dd - 7 - trday.min(), 0)}")
# reference: public / private / val with train-snapshot-only coverage (all train days for test; < 20713 for stale-val)
for sp, cutfn in (("test", None), ("val", 20713)):
    f, M, off = lib.meta(sp)
    it = np.load(FLAT / f"{sp}/cand_item_id.npy")
    fidx = np.repeat(np.arange(len(M)), M)
    c = 99998 if cutfn is None else cutfn
    z = first[it] >= c
    fz = fmean(z, fidx, len(M), M)
    if sp == "test":
        pub = f["timestamp"].to_numpy() < lib.BOUND
        pr(f"public train-snapshot-only zero share {fz[pub].mean():.4f}; private {fz[~pub].mean():.4f}; train history days 14")
    else:
        pr(f"val, train snapshots day < 20713 only (stale-val 'both' cut) zero share {fz.mean():.4f}; history days 11")
pr(f"done {time.time()-t0:.0f}s")
