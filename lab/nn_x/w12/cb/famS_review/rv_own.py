"""famS_review: reviewer's OWN evaluation code (no import of nats.py, cbx.py, pc_common.py, src/metrics.py or any famS script).
Witnesses are rebuilt from raw files:
  val     : data/flat/val/feeds.parquet (M, cand_off) + cand_target.npy
  holdout : rebuilt from data/flat/train (feeds.parquet, cand_target.npy) + eda/labels_position/out/train_prandom.npz with the documented draw
            (rng 12345, 30,000 of p > 0.9 & M >= 50), and compared with nn_x/w5/preds/holdout/holdout_{meta,y}.npy
  W16     : member-equal mean of the per-feed z of the 16 train-only members, then z again (rebuilt from the member files)
Own implementations: per-feed z (reduceat), NDCG with pessimistic ties by COUNTING (no sort), conditional logit by scipy BFGS + analytic gradient."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "4"
import numpy as np
import numba as nb
import polars as pl
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path("<repo>")
W5 = ROOT / "nn_x/w5/preds"
FAM = ROOT / "nn_x/w12/cb/famS"
INFRA = ROOT / "nn_x/w12/cb/infra"
W16 = [f"w5_e10_kd05w192_s{s}" for s in range(951, 959)] + [f"w5_e10_kd05t25w192_s{s}" for s in range(951, 959)]


def train_hold_index():
    """feed indices (into data/flat/train/feeds.parquet) of the 30K holdout and of xm, by the documented recipe."""
    f = pl.read_parquet(ROOT / "data/flat/train/feeds.parquet", columns=["M", "cand_off", "user_id"])
    M = f["M"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    xm = np.setdiff1d(np.where((p > 0.5) & (M >= 50))[0], hold)
    return f, M, hold, xm


def witnesses():
    out = {}
    f = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet", columns=["M", "cand_off"])
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    assert (off == np.r_[0, np.cumsum(M)[:-1]]).all()
    y = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float64)
    assert len(y) == M.sum()
    out["val"] = dict(M=M, off=off, y=y)
    ft, Mt, hold, xm = train_hold_index()
    offt = ft["cand_off"].to_numpy().astype(np.int64)
    Mh = Mt[hold]; offh = np.r_[0, np.cumsum(Mh)[:-1]].astype(np.int64)
    yt = np.load(ROOT / "data/flat/train/cand_target.npy", mmap_mode="r")
    yh = np.concatenate([np.asarray(yt[o:o + m]) for o, m in zip(offt[hold], Mh)]).astype(np.float64)
    meta = np.load(W5 / "holdout/holdout_meta.npy"); yref = np.load(W5 / "holdout/holdout_y.npy")
    chk = dict(meta_M_equal=bool((meta[0] == Mh).all()), meta_off_equal=bool((meta[1] == offh).all()), y_equal=bool((yref == yh).all()),
               n_hold=int(len(hold)), n_xm=int(len(xm)), hold_in_xm=int(np.isin(hold, xm).sum()))
    out["hold"] = dict(M=Mh, off=offh, y=yh, check=chk)
    return out


def fid_of(M):
    return np.repeat(np.arange(len(M)), M)


def zfeed(x, off, M):
    """per-feed z-score, population sd (+1e-9 as the project convention), float64."""
    x = np.asarray(x, np.float64); fid = fid_of(M)
    mu = np.add.reduceat(x, off) / M
    d = x - mu[fid]
    sd = np.sqrt(np.add.reduceat(d * d, off) / M) + 1e-9
    return d / sd[fid]


@nb.njit(parallel=True, cache=True)
def _ndcg_count(s, y, off, M):
    n = len(M); out = np.ones(n, np.float64)
    for f in nb.prange(n):
        o = off[f]; m = M[f]
        npos = 0
        for k in range(m):
            if y[o + k] > 0:
                npos += 1
        if npos == 0:
            continue
        dcg = 0.0
        for i in range(m):
            if y[o + i] > 0:
                si = s[o + i]; g = 0; tn = 0; tp = 0
                for j in range(m):
                    sj = s[o + j]
                    if sj > si:
                        g += 1
                    elif sj == si:
                        if y[o + j] > 0:
                            if j < i:
                                tp += 1
                        else:
                            tn += 1
                dcg += 1.0 / np.log2(g + tn + tp + 2.0)      # pessimistic: tied negatives first
        idcg = 0.0
        for k in range(npos):
            idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


def ndcg(s, w):
    """per-feed NDCG, binary gain, full list, ties pessimistic; feeds without a click = 1 (the project's convention)."""
    return _ndcg_count(np.ascontiguousarray(s, np.float64), np.ascontiguousarray(w["y"], np.float64), w["off"], w["M"])


class CL:
    """conditional logit (listwise softmax, target y / n_pos) on the feeds with at least one click of a witness (optionally a feed subset)."""

    def __init__(self, w, feeds=None):
        M, off, y = w["M"], w["off"], w["y"]
        npos = np.add.reduceat(y, off)
        keep = npos > 0
        if feeds is not None:
            keep &= feeds
        self.keep = keep
        self.rows = np.repeat(keep, M)
        self.M = M[keep]; self.off = np.r_[0, np.cumsum(self.M)[:-1]].astype(np.int64)
        self.fid = fid_of(self.M)
        self.t = y[self.rows] / npos[keep][self.fid]
        self.nf = int(keep.sum())

    def _lossgrad(self, b, X):
        s = X @ b
        mx = np.maximum.reduceat(s, self.off)
        e = np.exp(s - mx[self.fid]); den = np.add.reduceat(e, self.off)
        lse = mx + np.log(den)
        lf = lse - np.add.reduceat(self.t * s, self.off)
        p = e / den[self.fid]
        g = X.T @ (p - self.t)
        return lf, g

    def fit(self, cols):
        """cols: list of per-row arrays (full witness rows). Returns (beta, per-feed loss over kept feeds)."""
        X = np.stack([np.asarray(c, np.float64)[self.rows] for c in cols], 1)
        fun = lambda b: (lambda lf, g: (lf.sum() / self.nf, g / self.nf))(*self._lossgrad(b, X))
        r = minimize(fun, np.zeros(X.shape[1]), jac=True, method="BFGS", options=dict(gtol=1e-9, maxiter=500))
        lf, g = self._lossgrad(r.x, X)
        assert np.abs(g / self.nf).max() < 1e-6, (r.message, g / self.nf)
        return r.x, lf


def pse(d):
    d = np.asarray(d, np.float64)
    return float(d.mean()), float(d.std() / np.sqrt(len(d)))


def w16_pool(sp, w):
    acc = np.zeros(int(w["M"].sum()))
    for m in W16:
        f = W5 / (f"holdout/nn_{m}_holdout.npy" if sp == "hold" else f"nn_{m}_trainonly_val.npy")
        x = np.load(f); assert x.shape == acc.shape and np.isfinite(x).all(), f
        acc += zfeed(x, w["off"], w["M"])
    return zfeed(acc / len(W16), w["off"], w["M"])
