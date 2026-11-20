"""c03b: the same label-free test (within-feed corr of pos with the production scores f1/f3/f5) and the P(y|pos) lift
curve, split by M bin and gap==0 / gap>0, so the weakspots 'M[5,20) gap==0 rank-0 lift 1.00' subgroup is read on its own."""
import os, sys, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ.setdefault(k, "12")
sys.path.insert(0, "<repo>/src")
import numpy as np
from common import feed_meta, FLAT
t0 = time.time()
f, M, off = feed_meta("train"); n = len(M)
pos_a = np.asarray(np.load(FLAT / "train/cand_pos.npy", mmap_mode="r"))
y = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
F = {k: np.load(FLAT / f"train/cand_{k}.npy", mmap_mode="r") for k in ("f1", "f3", "f5")}
maxpos = np.maximum.reduceat(pos_a, off); gap0 = maxpos == M - 1
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"]
rng = np.random.default_rng(1)
def pct_rank_within(x, o, m):
    fidl = np.repeat(np.arange(len(m)), m); order = np.lexsort((x, fidl)); r = np.empty(len(x)); r[order] = np.arange(len(x)) - np.repeat(o, m)
    return r / np.maximum(np.repeat(m, m) - 1, 1)
def stats(name, g, nmax=60000):
    ids = np.nonzero(g)[0]
    if len(ids) < 1000:
        print(name, "n", len(ids)); return
    s = np.sort(rng.choice(ids, min(nmax, len(ids)), replace=False))
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off[s], M[s])]); o2 = np.concatenate([[0], np.cumsum(M[s])[:-1]])
    pp = pct_rank_within(pos_a[rows].astype(np.float64), o2, M[s]); yy = np.asarray(y[rows], np.float64)
    cs = {k: np.corrcoef(pp, pct_rank_within(np.asarray(F[k][rows], np.float64), o2, M[s]))[0, 1] for k in F}
    pr_ = pos_a[rows]; rate = yy.mean(); lifts = [yy[pr_ == r].mean() / rate for r in range(8)]
    # positives per feed and the share of feeds whose (only) positive is the last shown item (rkend 0)
    last = (pr_ == np.repeat(maxpos[s], M[s])); p_last = yy[last].mean() / rate
    print(f"{name:34s} feeds {len(ids):7d} (sampled {len(s):6d}) corr(pos,f1/f3/f5) {cs['f1']:+.3f}/{cs['f3']:+.3f}/{cs['f5']:+.3f} | lift r=0..7 " + " ".join(f"{l:.2f}" for l in lifts) + f" | lift last-shown {p_last:.2f} | rate {rate:.4f} | mean p_random {np.nan_to_num(pr[s]).mean():.2f}")
for lo, hi in ((5, 10), (10, 20), (20, 35), (35, 50)):
    mm = (M >= lo) & (M < hi)
    stats(f"M[{lo},{hi}) gap==0", mm & gap0); stats(f"M[{lo},{hi}) gap>0", mm & ~gap0)
stats("M>=50 random-like p>0.9", (M >= 50) & (pr > 0.9)); stats("M>=50 production p<0.1", (M >= 50) & (pr < 0.1))
stats("M>=50 gap==0 (p any)", (M >= 50) & gap0); stats("M[50,80) gap==0", (M >= 50) & (M < 80) & gap0); stats("M[50,80) production p<0.1", (M >= 50) & (M < 80) & (pr < 0.1))
print(f"done c03b in {time.time()-t0:.0f}s")
