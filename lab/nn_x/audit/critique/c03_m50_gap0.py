"""Critique check 3: the weakspots claim that 409K M<50 train feeds (gap == 0) are 'random-order' and that the K9 aux
distils a random order into them. Label-free re-check: is `pos` a production rank at all in those feeds?
A production rank must correlate with the production scores f1..f5 within the feed (the ranker ordered by them);
a random display order does not change that (pos is the production rank, not the display slot). So:
  - if pos is a real production rank: within-feed corr(rank of pos, rank of f1) strongly negative in every group;
  - if pos is an arbitrary index (e.g. array order): corr ~0.
Also: shares, whether pos == array index in gap-0 feeds, and the P(y | pos) curves by M bin (lift relative to the group mean).
CPU only, train split read through memmaps; sampled feeds for the correlation statistic."""
import os, sys, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ.setdefault(k, "12")
sys.path.insert(0, "<repo>/src")
import numpy as np
from common import feed_meta, FLAT
t0 = time.time()
f, M, off = feed_meta("train"); n = len(M)
pos = np.load(FLAT / "train/cand_pos.npy", mmap_mode="r")
y = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
print(f"train feeds {n}, rows {len(pos)}, pos dtype {pos.dtype}; M<50 share {np.mean(M < 50):.3f}; M<50 rows share {M[M<50].sum()/M.sum():.3f}")
pos_a = np.asarray(pos)
maxpos = np.maximum.reduceat(pos_a, off)
gap = 1.0 - M / (maxpos + 1.0)
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"]
small = M < 50
print(f"M<50 feeds {small.sum()}: gap==0 {np.sum(small & (gap == 0))} ({np.mean(gap[small] == 0):.3f}); M>=50: gap==0 share {np.mean(gap[~small] == 0):.3f}; M>=50 random-like (p>0.9) gap==0 share {np.mean(gap[(~small) & (pr > 0.9)] == 0):.3f}; production (p<0.1) {np.mean(gap[(~small) & (pr < 0.1)] == 0):.3f}")
for lo, hi in ((5, 10), (10, 20), (20, 35), (35, 50)):
    mm = (M >= lo) & (M < hi)
    print(f"  M[{lo},{hi}): feeds {mm.sum()}, gap==0 share {np.mean(gap[mm] == 0):.3f}, median maxpos+1 {np.median(maxpos[mm] + 1):.0f}, median M {np.median(M[mm]):.0f}")
# pos == array index?
idx_in_feed = np.arange(len(pos_a)) - np.repeat(off, M)
eq = (pos_a == idx_in_feed)
eq_feed = np.add.reduceat(eq.astype(np.int64), off) == M
print(f"feeds where pos == array index exactly: all {eq_feed.mean():.4f}; M<50 gap==0 {eq_feed[small & (gap == 0)].mean():.4f}; M<50 gap>0 {eq_feed[small & (gap > 0)].mean():.4f}; M>=50 gap==0 {eq_feed[(~small) & (gap == 0)].mean():.4f}")
# within-feed correlation of pos with f1 (production score) -- sampled feeds per group
f1 = np.load(FLAT / "train/cand_f1.npy", mmap_mode="r")
rng = np.random.default_rng(0)
groups = {"M<50 gap==0": small & (gap == 0), "M<50 gap>0": small & (gap > 0),
          "M>=50 random-like p>0.9": (~small) & (pr > 0.9), "M>=50 production p<0.1": (~small) & (pr < 0.1),
          "M>=50 gap==0": (~small) & (gap == 0)}
def pct_rank_within(x, o, m):
    fidl = np.repeat(np.arange(len(m)), m); order = np.lexsort((x, fidl)); r = np.empty(len(x)); r[order] = np.arange(len(x)) - np.repeat(o, m)
    return r / np.maximum(np.repeat(m, m) - 1, 1)
for name, g in groups.items():
    ids = np.nonzero(g)[0]
    if len(ids) == 0:
        print(name, "empty"); continue
    s = np.sort(rng.choice(ids, min(60000, len(ids)), replace=False))
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off[s], M[s])])
    o2 = np.concatenate([[0], np.cumsum(M[s])[:-1]])
    pp = pct_rank_within(pos_a[rows].astype(np.float64), o2, M[s]); ff = pct_rank_within(np.asarray(f1[rows], np.float64), o2, M[s])
    ii = pct_rank_within(idx_in_feed[rows].astype(np.float64), o2, M[s])
    yy = np.asarray(y[rows], np.float64)
    c_pf = np.corrcoef(pp, ff)[0, 1]; c_if = np.corrcoef(ii, ff)[0, 1]; c_pi = np.corrcoef(pp, ii)[0, 1]
    # P(y | pos rank r) lift for r = 0..5 relative to the group's mean rate; and by array index
    pr_ = pos_a[rows]; rate = yy.mean()
    lifts = [yy[pr_ == r].mean() / rate for r in range(6)]
    arr_l = [yy[idx_in_feed[rows] == r].mean() / rate for r in range(6)]
    print(f"{name:26s} feeds {len(s):6d}: corr(pos pct, f1 pct) {c_pf:+.3f} | corr(array idx pct, f1 pct) {c_if:+.3f} | corr(pos, array idx) {c_pi:+.3f} | lift P(y|pos=r)/mean r=0..5 " + " ".join(f"{l:.2f}" for l in lifts) + " | by array idx " + " ".join(f"{l:.2f}" for l in arr_l) + f" | mean rate {rate:.4f}")
print(f"done c03 in {time.time()-t0:.0f}s")
