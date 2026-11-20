"""why/fit shared CPU helpers (read-only): per-feed loss components of the KD-mixture objective and agreement with the teacher.
For a feed with scores s (student logits), teacher logits t and clicks y (k = sum y > 0):
  p = softmax(s), q = softmax(t), target = 0.5 y/k + 0.5 q (train.py, kd_w 0.5, T 1)
  cc   = -(1/k) sum_y log p            click cross-entropy
  ct   = -sum q log p                  teacher cross-entropy;  kl = ct - H(q) = KL(q || p)
  main = 0.5 cc + 0.5 ct               the main (listwise) training loss of the feed (aux K9 term not included)
  exc  = main - H(target) = KL(target || p)   the reducible part (0 only if p = target, i.e. the clicks memorised)
  r    = Pearson corr(s, t) within the feed
  cct  = -(1/k) sum_y log q            the teacher's own click cross-entropy
Feeds without clicks get NaN for cc / main / exc / cct (kl and r are defined for every feed)."""
import os
import sys

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(k, "12")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numba as nb
import numpy as np

sys.path.insert(0, "<repo>/nn_x/w12/recheck/pool_composition")
from pc_common import ndcg_fast, Meta, se, feed_rank_z  # noqa: E402,F401


@nb.njit(parallel=True, cache=True)
def feed_stats(s, t, y, off, M):
    n = len(M)
    out = np.full((n, 8), np.nan)  # cc, ct, kl, main, exc, r, cct, k
    for f in nb.prange(n):
        o, m = off[f], M[f]
        sv = s[o:o + m].astype(np.float64)
        tv = t[o:o + m].astype(np.float64)
        yv = y[o:o + m].astype(np.float64)
        ms, mt = sv.max(), tv.max()
        lse_s = ms + np.log(np.exp(sv - ms).sum())
        lse_t = mt + np.log(np.exp(tv - mt).sum())
        lps = sv - lse_s
        lqt = tv - lse_t
        q = np.exp(lqt)
        ct = -(q * lps).sum()
        hq = -(q * lqt).sum()
        kl = ct - hq
        sm, tm = sv.mean(), tv.mean()
        num = ((sv - sm) * (tv - tm)).sum()
        den = np.sqrt(((sv - sm) ** 2).sum() * ((tv - tm) ** 2).sum())
        r = num / den if den > 0 else np.nan
        k = yv.sum()
        out[f, 1] = ct
        out[f, 2] = kl
        out[f, 5] = r
        out[f, 7] = k
        if k > 0:
            cc = -(yv * lps).sum() / k
            cct = -(yv * lqt).sum() / k
            tg = 0.5 * yv / k + 0.5 * q
            ht = 0.0
            for i in range(m):
                if tg[i] > 0:
                    ht -= tg[i] * np.log(tg[i])
            main = 0.5 * cc + 0.5 * ct
            out[f, 0] = cc
            out[f, 3] = main
            out[f, 4] = main - ht
            out[f, 6] = cct
    return out


COLS = ("cc", "ct", "kl", "main", "exc", "r", "cct", "k")


def stats_dict(s, t, y, off, M):
    a = feed_stats(np.ascontiguousarray(s, np.float32), np.ascontiguousarray(t, np.float32), np.ascontiguousarray(y, np.float32),
                   np.asarray(off, np.int64), np.asarray(M, np.int64))
    d = {c: a[:, i] for i, c in enumerate(COLS)}
    d["nd"] = ndcg_fast(s, np.ascontiguousarray(y, np.float32), np.asarray(off, np.int64), np.asarray(M, np.int64))
    return d


def pse(x):
    """mean and SE of a per-feed vector (NaN-dropped)."""
    x = x[np.isfinite(x)]
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)), len(x)
