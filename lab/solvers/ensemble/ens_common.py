"""Shared helpers for the ensemble / stacking design study (CPU only, read-only on preds/ and data/).

All arrays are flat in feed order (aligned with data/flat/<split>/cand_*.npy).
Folds = nn/finetune.py: sklearn GroupKFold(5) over np.arange(n_val_feeds), groups = val user_id (user 0 = one group).
"""
import os
import sys

for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ.setdefault(k, "16")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

ROOT = "<repo>"
sys.path.insert(0, f"{ROOT}/src")
from metrics import ndcg_per_feed  # noqa: E402

PREDS = f"{ROOT}/preds"
HOLD = f"{ROOT}/preds/holdout"
BOUND = 1790497920
EDGES = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250])


class Split:
    def __init__(self, split):
        f = pl.read_parquet(f"{ROOT}/data/flat/{split}/feeds.parquet")
        self.split = split
        self.M = f["M"].to_numpy().astype(np.int64)
        self.off = f["cand_off"].to_numpy().astype(np.int64)
        self.H = f["H"].to_numpy().astype(np.int64)
        self.user = f["user_id"].to_numpy().astype(np.int64)
        self.ts = f["timestamp"].to_numpy().astype(np.int64)
        self.n = len(self.M)
        self.fid = np.repeat(np.arange(self.n), self.M)
        self.N = int(self.M.sum())
        if split == "val":
            self.y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
            day = (self.ts + 3 * 3600) // 86400
            self.day = np.where(day <= day.min() + 1, 0, 1)  # 0 = Tue, 1 = Thu (as nn/pcmp.py)
            gkf = GroupKFold(n_splits=5)
            self.fold = np.full(self.n, -1)
            for k, (_, ho) in enumerate(gkf.split(np.arange(self.n), groups=self.user)):
                self.fold[ho] = k
        else:
            self.y = None
        luo = np.load(f"{ROOT}/data/feats/{split}/it_hist_all_luo.npy", mmap_mode="r")
        self.cold = (np.asarray(luo) == 0).astype(np.float64)  # candidate has no leave-user-out history popularity
        self.cold_share = np.bincount(self.fid, self.cold, self.n) / self.M

    # ---------- per-feed transforms ----------
    def z(self, s):
        s = np.asarray(s, dtype=np.float64)
        mu = np.bincount(self.fid, s, self.n) / self.M
        sd = np.sqrt(np.maximum(np.bincount(self.fid, s * s, self.n) / self.M - mu ** 2, 0)) + 1e-6
        return (s - mu[self.fid]) / sd[self.fid]

    def rank01(self, s):
        """Within-feed rank scaled to [0,1] (1 = best), average ties not needed (continuous scores)."""
        s = np.asarray(s, dtype=np.float64)
        order = np.lexsort((s, self.fid))  # ascending score within feed
        r = np.empty(self.N)
        r[order] = np.arange(self.N) - np.repeat(self.off, self.M)
        return r / np.maximum(self.M[self.fid] - 1, 1)

    def softmax(self, s, T=1.0):
        s = np.asarray(s, dtype=np.float64) / T
        mx = np.full(self.n, -np.inf)
        np.maximum.at(mx, self.fid, s)
        e = np.exp(s - mx[self.fid])
        return e / np.bincount(self.fid, e, self.n)[self.fid]

    def per_feed(self, s):
        return ndcg_per_feed(s, self.y, self.off, self.M)

    def private_w(self, kind="private"):
        """calib.py weights: reweight val feeds to the test public/private M-mix."""
        te = pl.read_parquet(f"{ROOT}/data/flat/test/feeds.parquet")
        Mt = te["M"].to_numpy()
        tt = te["timestamp"].to_numpy()
        mask = tt < BOUND if kind == "public" else tt >= BOUND
        bv = np.searchsorted(EDGES, self.M, side="right")
        bt = np.searchsorted(EDGES, Mt[mask], side="right")
        pv = np.bincount(bv, minlength=len(EDGES) + 1) / len(bv)
        pt = np.bincount(bt, minlength=len(EDGES) + 1) / len(bt)
        w = (pt / np.maximum(pv, 1e-12))[bv]
        return w / w.mean()


def load(name, split):
    """name = preds file stem without split suffix handling: we pass the full stem, e.g. nn_final5_val_oof."""
    return np.load(f"{PREDS}/{name}.npy").astype(np.float64)


class Holdout:
    def __init__(self):
        M, off = np.load(f"{HOLD}/holdout_meta.npy")
        self.M = M.astype(np.int64)
        self.off = off.astype(np.int64)
        self.n = len(self.M)
        self.N = int(self.M.sum())
        self.fid = np.repeat(np.arange(self.n), self.M)
        self.y = np.load(f"{HOLD}/holdout_y.npy").astype(np.float64)
        tr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp", "M"])
        pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
        trM = tr["M"].to_numpy()
        pr[trM < 50] = 0.0
        cand = np.where((pr > 0.9) & (trM >= 50))[0]
        hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
        assert (trM[hold] == self.M).all()
        self.ts = tr["timestamp"].to_numpy()[hold]
        self.late = self.ts >= np.quantile(self.ts, 0.8)

    z = Split.z
    rank01 = Split.rank01
    softmax = Split.softmax

    def per_feed(self, s):
        return ndcg_per_feed(s, self.y, self.off, self.M)

    def load(self, member):
        return np.load(f"{HOLD}/nn_{member}_holdout.npy").astype(np.float64)


def se(d, w=None):
    if w is None:
        return d.mean(), d.std() / np.sqrt(len(d))
    w = w / w.mean()
    x = w * d
    return x.mean(), x.std() / np.sqrt(len(d))


def slices(S, d):
    """Standard slices for a val per-feed delta d."""
    out = {}
    masks = {"all": np.ones(S.n, bool), "M<50": S.M < 50, "M>=50": S.M >= 50, "Tue": S.day == 0, "Thu": S.day == 1}
    for k, m in masks.items():
        out[k] = (d[m].mean(), d[m].std() / np.sqrt(m.sum()))
    out["priv_w"] = se(d, S.private_w("private"))
    return out


def fmt_slices(sl):
    return "  ".join(f"{k} {m:+.5f}+-{s:.5f}" for k, (m, s) in sl.items())
