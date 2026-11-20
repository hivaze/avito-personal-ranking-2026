"""Shared loaders for the overfit / selection-optimism audit (CPU only, read-only on preds/)."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "12")
import sys
sys.path.insert(0, "<repo>/src")
from pathlib import Path
import numpy as np
import polars as pl
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path("<repo>")
P = ROOT / "preds"
H = P / "holdout"
OUT = ROOT / "nn_x/pubdiag/overfit"
PUB_TS = 1790497920

FAM = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84",
             "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1"] + [f"v14_s{s}" for s in range(92, 97)],
    "F7": [f"v17_pre2s_s{s}" for s in range(101, 108)],
    "F9": [f"v22_pre2s_lr1_s{s}" for s in range(401, 407)],
    "F10": [f"v23_gru_pre2s_s{s}" for s in range(501, 507)],
    "F12": [f"v26_d192_pre2s_s{s}" for s in range(701, 708)],
    "F7x": [f"r6_cd_m02_r{s}" for s in range(101, 108)],
    "F7x70": [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)],
    "F7b": [f"r6_cd_c0_r{s}" for s in range(101, 108)],
}
POOLS = {
    "Cd": ["F5cd", "F6", "F7"],
    "Pall": ["F5cd", "F6", "F7", "F9", "F10", "F12"],
    "PF7x": ["F5cd", "F6", "F7x", "F9", "F10", "F12"],
    "G3": ["F10", "F7x", "F7x70"],
    "M2": ["F7x", "F7x70"],
}
PUBLIC = {"Cd": 0.39208, "PF7x": 0.392723, "G3": 0.392819, "M2": 0.392434}  # context only, never fitted
MEMBERS = [m for f in FAM.values() for m in f]


def pool_members(pool):
    return [m for f in POOLS[pool] for m in FAM[f]]


def se(d):
    d = np.asarray(d, np.float64)
    return d.std() / np.sqrt(len(d))


class Split:
    """Per-feed meta + z-scoring for val / test / holdout."""

    def __init__(self, name):
        self.name = name
        if name == "holdout":
            M, off = np.load(H / "holdout_meta.npy")
            self.M, self.off = M.astype(np.int64), off.astype(np.int64)
            self.y = np.load(H / "holdout_y.npy").astype(np.float64)
            self.users, self.ts = holdout_users_ts(self.M)
        else:
            f, M, off = feed_meta(name)
            self.M, self.off = M.astype(np.int64), off.astype(np.int64)
            self.users = f["user_id"].to_numpy()
            self.ts = f["timestamp"].to_numpy()
            self.y = np.load(FLAT / "val/cand_target.npy").astype(np.float64) if name == "val" else None
        self.n = len(self.M)
        self.fid = np.repeat(np.arange(self.n), self.M)

    def z(self, x):
        return feed_rank_z(self.off, self.M, np.asarray(x, np.float32))[:, 1]

    def nd(self, s):
        return ndcg_per_feed(s, self.y, self.off, self.M)

    def load(self, m, half):
        """half 'to' (train-only) or 'ft' (n2cpv3 fine-tune); returns z-scored member scores."""
        if self.name == "holdout":
            assert half == "to"
            return self.z(np.load(H / f"nn_{m}_holdout.npy"))
        if half == "to":
            return self.z(np.load(P / f"nn_{m}_trainonly_{self.name}.npy"))
        return self.z(np.load(P / f"nn_{m}_n2cpv3_{'val_oof' if self.name == 'val' else 'test'}.npy"))


def holdout_users_ts(Mh):
    """Recreate the train.py --holdout selection (rng 12345 over p_random > 0.9 & M >= 50 train feeds)."""
    tf = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["user_id", "timestamp", "M"])
    M = tf["M"].to_numpy()
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    prn = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    prn[M < 50] = 0.0
    cand = np.where((prn > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    assert (M[hold] == Mh).all(), "holdout mapping mismatch"
    return tf["user_id"].to_numpy()[hold], tf["timestamp"].to_numpy()[hold]


def user_groups(users):
    """Group ids for user-grouped resampling; user 0 = pooled bucket -> singleton groups."""
    g = users.astype(np.int64).copy()
    z = np.where(users == 0)[0]
    g[z] = -(np.arange(len(z)) + 1)
    _, gi = np.unique(g, return_inverse=True)
    return gi
