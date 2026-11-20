"""Shared loader / evaluator for the post-hoc stream (CPU only).
Base = current best file FD16P20@m0.4_0.2: final = 0.5 z(ft) + 0.5 z(to), z = per-feed z-score.
Cross-fitting: users split into halves A/B by a hash of user_id; user_id 0 feeds are singletons (assigned by fidx parity)."""
import os, sys
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(k, "12")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "<repo>/src")
import numpy as np, polars as pl
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
import pmix

P = "<repo>/preds"
TAG = "FD16P20_m04_02kd"


def zf(x, off, M):
    fid = np.repeat(np.arange(len(M)), M)
    x = np.asarray(x, np.float64)
    mu = np.bincount(fid, x, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, x * x, len(M)) / M - mu ** 2, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]


class Data:
    def __init__(self, split="val", tag=TAG):
        self.split = split
        self.f, self.M, self.off = feed_meta(split)
        self.n = len(self.M)
        self.fid = np.repeat(np.arange(self.n), self.M)
        self.ft = np.load(f"{P}/ens_{tag}_ft_{split}.npy").astype(np.float64)
        self.to = np.load(f"{P}/ens_{tag}_to_{split}.npy").astype(np.float64)
        self.zft = zf(self.ft, self.off, self.M); self.zto = zf(self.to, self.off, self.M)
        self.s = 0.5 * self.zft + 0.5 * self.zto
        if split == "val":
            self.y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
            self.base = ndcg_per_feed(self.s, self.y, self.off, self.M)
            self.K = np.bincount(self.fid, self.y, self.n)
            W = pmix.weights(); self.wpub, self.wpriv = W["public"], W["private"]
        u = self.f["user_id"].to_numpy().astype(np.int64)
        h = (u * 2654435761 + 12345) % 1000003
        half = (h % 2).astype(np.int64)
        half[u == 0] = np.arange(self.n)[u == 0] % 2
        self.half = half                      # per feed
        self.halfr = half[self.fid]           # per row

    def ev(self, s):
        return ndcg_per_feed(np.asarray(s, np.float64), self.y, self.off, self.M)

    def report(self, s, label, ref=None, mask=None):
        """paired delta of per-feed NDCG vs ref (default the base file); s may be per-feed NDCG array or row scores."""
        e = s if len(s) == self.n else self.ev(s)
        b = self.base if ref is None else ref
        d = e - b
        big = self.M >= 50
        def f(x, w=None):
            m, se = pmix.wmean_se(x, w)
            return m, se
        out = dict(label=label, val=float(e.mean()))
        for nm, x, w in (("all", d, None), ("pub", d, self.wpub), ("priv", d, self.wpriv), ("M<50", d[~big], None), ("M>=50", d[big], None)):
            m, se = f(x, w); out[nm] = (float(m), float(se))
        line = f"{label:44s} val {e.mean():.5f} | d {out['all'][0]:+.5f}+-{out['all'][1]:.5f} (z {out['all'][0]/max(out['all'][1],1e-12):+.2f}) | pub {out['pub'][0]:+.5f}+-{out['pub'][1]:.5f} | priv {out['priv'][0]:+.5f}+-{out['priv'][1]:.5f} | M<50 {out['M<50'][0]:+.5f}+-{out['M<50'][1]:.5f} | M>=50 {out['M>=50'][0]:+.5f}+-{out['M>=50'][1]:.5f}"
        print(line, flush=True)
        out["line"] = line
        return out
