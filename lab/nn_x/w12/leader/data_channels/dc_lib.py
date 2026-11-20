"""Shared helpers of the data-channel inventory probes (role: data_channels).

CPU only. Reads the flat competition arrays and the stored predictions of the best file
FD16P20@m0.4_0.2 (artifacts/predictions/ens_FD16P20_m04_02kd_{ft,to}_{val,test}.npy); writes only under this directory.
Base score on val = 0.5 z(ft) + 0.5 z(to), z within feed (float32 numba kernel of avito_ranker.ensemble.zscore),
which must reproduce the recorded validation NDCG 0.40293.
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>")
from avito_ranker.metrics import ndcg_per_feed  # noqa: E402
from avito_ranker.ensemble.zscore import feed_rank_z  # noqa: E402

ROOT = "<repo>"
FLAT = f"{ROOT}/artifacts/data/flat"
FEATS = f"{ROOT}/artifacts/data/feats"
PREDS = f"{ROOT}/artifacts/predictions"
OUT = f"{ROOT}/lab/nn_x/w12/leader/data_channels"
BOUND = 1790497920          # public = test feeds with timestamp < BOUND (docs/01)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class Split:
    def __init__(self, split, hist=True):
        self.split = split
        f = pl.read_parquet(f"{FLAT}/{split}/feeds.parquet")
        self.f = f
        self.n = len(f)
        self.M = f["M"].to_numpy().astype(np.int64)
        self.off = f["cand_off"].to_numpy().astype(np.int64)
        self.H = f["H"].to_numpy().astype(np.int64)
        self.hoff = f["hist_off"].to_numpy().astype(np.int64)
        self.user = f["user_id"].to_numpy().astype(np.int64)
        self.ts = f["timestamp"].to_numpy().astype(np.int64)
        self.feed_id = f["feed_id"].to_numpy()
        self.N = int(self.M.sum())
        self.fid = np.repeat(np.arange(self.n), self.M)
        self.item = np.load(f"{FLAT}/{split}/cand_item_id.npy")
        self.cat = np.load(f"{FLAT}/{split}/cand_category_id.npy")
        self.y = np.load(f"{FLAT}/{split}/cand_target.npy").astype(np.float64) if split != "test" else None
        if hist:
            self.hfid = np.repeat(np.arange(self.n), self.H)
            self.hitem = np.load(f"{FLAT}/{split}/hist_item_id.npy")
            self.htype = np.load(f"{FLAT}/{split}/hist_event_type.npy")
            self.hsrc = np.load(f"{FLAT}/{split}/hist_event_source.npy")
            self.hpos = np.load(f"{FLAT}/{split}/hist_hpos.npy")
        self.day = (self.ts + 10800) // 86400  # Moscow day
        if split == "test":
            self.public = self.ts < BOUND

    def zf(self, x):
        return feed_rank_z(self.off, self.M, np.asarray(x, np.float32))[:, 1].astype(np.float64)

    def best(self, name="FD16P20_m04_02kd"):
        ft = np.load(f"{PREDS}/ens_{name}_ft_{self.split}.npy")
        to = np.load(f"{PREDS}/ens_{name}_to_{self.split}.npy")
        return self.zf(0.5 * self.zf(ft) + 0.5 * self.zf(to))

    def ndcg(self, s):
        return ndcg_per_feed(np.asarray(s, np.float64), self.y, self.off, self.M)


def pmix_weights():
    """Public / private reweighting of the 14,423 val feeds (lab/src/pmix.py, M bin x cold-share bin)."""
    sys.path.insert(0, f"{ROOT}/lab/src")
    import pmix  # noqa
    w = pmix.weights()
    return w["public"], w["private"]


def wmean_se(x, w=None):
    if w is None:
        return float(x.mean()), float(x.std() / np.sqrt(len(x)))
    m = np.average(x, weights=w)
    neff = w.sum() ** 2 / (w ** 2).sum()
    return float(m), float(np.sqrt(np.average((x - m) ** 2, weights=w) / neff))


def user_halves(user, seed):
    """Boolean half per feed; non-zero users kept together, every user-0 feed its own group."""
    rng = np.random.default_rng(seed)
    g = user.copy()
    z = np.flatnonzero(user == 0)
    g[z] = -(np.arange(len(z)) + 1)          # user-0 feeds as singleton groups
    uu, inv = np.unique(g, return_inverse=True)
    h = rng.random(len(uu)) < 0.5
    return h[inv]
