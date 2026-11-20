"""C06 Bradley-Terry / first-choice item strength (PREDECLARED_ccb.txt C06). Build only -> cache/c06_{val,hold}.npz.
theta maximises sum_feeds sum_pos [theta_i - log sum_{j in feed} exp theta_j] - lam ||theta||^2 (lam 3 primary; 1 and 10 info, val
only), L-BFGS over items seen >= 1 time in the source; unseen theta = 0 (bt_seen = 0).
Source feeds: train feeds with p_random > 0.5 (NaN -> 0) and >= 1 positive, the 30K holdout feeds excluded.
val: all train days. holdout feed of train day d: source days < d (one fit per holdout day, warm-started; strictly convex).
No val labels anywhere in the source."""
import os
import sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
from pathlib import Path
import numba as nb
import numpy as np
import polars as pl
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
FLAT = Path("<repo>/data/flat")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


@nb.njit(parallel=True, cache=True)
def gather_rows(sel, off, M, out_off, src, out):
    for q in nb.prange(len(sel)):
        f = sel[q]
        o = out_off[q]
        for j in range(M[f]):
            out[o + j] = src[off[f] + j]


@nb.njit(parallel=True, cache=True)
def feed_lse(s, off, M, out_lse, out_e):
    for f in nb.prange(len(M)):
        o = off[f]
        mx = -1e300
        for j in range(M[f]):
            if s[o + j] > mx:
                mx = s[o + j]
        Z = 0.0
        for j in range(M[f]):
            e = np.exp(s[o + j] - mx)
            out_e[o + j] = e
            Z += e
        out_lse[f] = mx + np.log(Z)
        for j in range(M[f]):
            out_e[o + j] /= Z


def fit_bt(it, y, off, M, npos, n_items, lam, th0=None, maxiter=500):
    fidr = np.repeat(np.arange(len(M)), M)
    lse = np.empty(len(M))
    p = np.empty(len(it))
    cpos = np.bincount(it, weights=y, minlength=n_items)

    def fg(th):
        s = th[it]
        feed_lse(s, off, M, lse, p)
        L = -(cpos * th).sum() + (npos * lse).sum() + lam * (th * th).sum()
        g = np.bincount(it, weights=npos[fidr] * p, minlength=n_items) - cpos + 2 * lam * th
        return L, g

    x0 = np.zeros(n_items) if th0 is None else th0
    r = minimize(fg, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, gtol=1e-6, ftol=1e-12))
    return r.x, r


def main():
    nb.set_num_threads(12)
    ft = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M", "timestamp", "cand_off", "user_id"])
    Mt = ft["M"].to_numpy().astype(np.int64)
    ot = ft["cand_off"].to_numpy().astype(np.int64)
    tday = (ft["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
    praw = np.nan_to_num(np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    ph = praw.copy()
    ph[Mt < 50] = 0.0
    HOLD = np.sort(np.random.default_rng(12345).choice(np.where((ph > 0.9) & (Mt >= 50))[0], 30000, replace=False))
    is_hold = np.zeros(len(Mt), bool)
    is_hold[HOLD] = True
    cand = np.nonzero((praw > 0.5) & ~is_hold)[0]
    yt = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
    it_t = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")
    Mc = Mt[cand]
    oc = np.concatenate([[0], np.cumsum(Mc)[:-1]]).astype(np.int64)
    y_c = np.empty(int(Mc.sum()), np.int8)
    i_c = np.empty(int(Mc.sum()), np.int32)
    gather_rows(cand, ot, Mt, oc, np.asarray(yt), y_c)
    gather_rows(cand, ot, Mt, oc, np.asarray(it_t), i_c)
    npos_c = np.add.reduceat(y_c.astype(np.int64), oc)
    keep = npos_c >= 1
    src = cand[keep]
    log(f"source candidates p>0.5 non-holdout: {len(cand)} feeds; with >=1 positive: {len(src)} feeds")
    Ms = Mt[src]
    os_ = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64)
    rowsel = np.repeat(keep, Mc)
    y_s = y_c[rowsel].astype(np.float64)
    i_s = i_c[rowsel]
    sday = tday[src]
    del y_c, i_c

    def fit_subset(fmask, lam, warm=None):
        rmask = np.repeat(fmask, Ms)
        items = i_s[rmask]
        uit, inv = np.unique(items, return_inverse=True)
        Msub = Ms[fmask]
        osub = np.concatenate([[0], np.cumsum(Msub)[:-1]]).astype(np.int64)
        yy = y_s[rmask]
        npos = np.add.reduceat(yy, osub)
        th0 = None
        if warm is not None:
            wu, wt = warm
            k = np.searchsorted(wu, uit)
            k = np.minimum(k, len(wu) - 1)
            th0 = np.where(wu[k] == uit, wt[k], 0.0)
        th, r = fit_bt(inv.astype(np.int64), yy, osub, Msub, npos, len(uit), lam, th0)
        log(f"  fit lam {lam}: {fmask.sum()} feeds, {rmask.sum()} rows, {len(uit)} items, nit {r.nit}, conv {r.success} "
            f"({r.message}), theta sd {th.std():.4f} min {th.min():.3f} max {th.max():.3f}")
        return uit, th

    def lookup(uit, th, items):
        k = np.searchsorted(uit, items)
        k = np.minimum(k, len(uit) - 1)
        seen = uit[k] == items
        return np.where(seen, th[k], 0.0), seen.astype(np.float64)

    iv = np.load(FLAT / "val/cand_item_id.npy")
    res_v = {}
    allf = np.ones(len(src), bool)
    for lam in (3.0, 1.0, 10.0):
        uit, th = fit_subset(allf, lam)
        t, s = lookup(uit, th, iv)
        res_v[f"theta_l{int(lam)}"] = t
        res_v[f"seen_l{int(lam)}"] = s
        if lam == 3.0:
            warm_all = (uit, th)
            log(f"val: bt_seen share {s.mean():.4f}")
    np.savez(HERE / "cache/c06_val.npz", **res_v)

    # holdout: per train day d, source days < d
    hd = tday[HOLD]
    Mh = Mt[HOLD]
    hoff = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64)
    ih = np.empty(int(Mh.sum()), np.int32)
    gather_rows(HOLD, ot, Mt, hoff, np.asarray(it_t), ih)
    th_h = np.zeros(len(ih))
    seen_h = np.zeros(len(ih))
    warm = None
    for d in np.unique(hd):
        fm = sday < d
        q = np.nonzero(hd == d)[0]
        rows = np.concatenate([np.arange(hoff[j], hoff[j] + Mh[j]) for j in q])
        if fm.sum() == 0:
            log(f"hold day {d}: empty source ({len(q)} feeds) -> theta 0, seen 0")
            continue
        uit, th = fit_subset(fm, 3.0, warm)
        warm = (uit, th)
        t, s = lookup(uit, th, ih[rows])
        th_h[rows] = t
        seen_h[rows] = s
        log(f"hold day {d}: {len(q)} feeds, seen share {s.mean():.4f}")
    np.savez(HERE / "cache/c06_hold.npz", theta_l3=th_h, seen_l3=seen_h)
    log("DONE")


if __name__ == "__main__":
    main()
