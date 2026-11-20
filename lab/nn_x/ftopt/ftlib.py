"""Shared evaluation code for the FT-stage Optuna study (nn_x/ftopt). CPU only, no model code."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("<repo>")
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed  # noqa: E402

FT_DIR = ROOT / "nn_x" / "ftopt"
PREDS = ROOT / "preds"
FLAT = ROOT / "data" / "flat"

SET_A = ["r6_cd_m02_r101", "r6_cd_m02_r103", "v27_gru_m02_s511", "v28_lr1_m02_s411"]
SET_B = ["v23_gru_pre2s_s501", "v23_gru_pre2s_s502", "v29_d192_m02_s711", "r6_cd_m02_p70k_r102"]


def code_of(m):
    """finetune.py code dir of a member: r6 runs live in nn_x/r6, the rest in nn."""
    return "r6" if m.startswith("r6_") else "nn"


def splitmix64(x):
    x = (np.asarray(x, np.uint64) + np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


class Val:
    def __init__(self):
        f = pl.read_parquet(FLAT / "val" / "feeds.parquet")
        self.M = f["M"].to_numpy().astype(np.int64)
        self.off = f["cand_off"].to_numpy().astype(np.int64)
        self.uid = f["user_id"].to_numpy().astype(np.int64)
        self.ts = f["timestamp"].to_numpy().astype(np.int64)
        self.fidx = np.arange(len(self.M), dtype=np.int64)
        self.n = len(self.M)
        self.y = np.load(FLAT / "val" / "cand_target.npy").astype(np.float64)
        self.fid = np.repeat(np.arange(self.n), self.M)
        sp = FT_DIR / "split.npy"
        if sp.exists():
            self.confirm = np.load(sp).astype(bool)
        else:
            h = np.where(self.uid != 0, splitmix64(self.uid), splitmix64(self.fidx + (1 << 40)))
            self.confirm = (h & np.uint64(1)).astype(bool)
            np.save(sp, self.confirm)
        self.tune = ~self.confirm
        tue = (self.ts + 10800) // 86400 < 20720
        self.slices = {"all": np.ones(self.n, bool), "M<50": self.M < 50, "M>=50": self.M >= 50, "Tue": tue,
                       "Thu": ~tue}
        self.cluster = np.where(self.uid != 0, self.uid, -1 - self.fidx)

    def z(self, s):
        s = np.asarray(s, np.float64)
        mu = np.bincount(self.fid, s, self.n) / self.M
        sd = np.sqrt(np.maximum(np.bincount(self.fid, s * s, self.n) / self.M - mu ** 2, 0)) + 1e-6
        return (s - mu[self.fid]) / sd[self.fid]

    def zmean(self, arrs):
        return self.z(np.mean([self.z(a) for a in arrs], 0))

    def pool(self, ft_list, to_z, w_ft=0.5):
        """w z(mean z(ft)) + (1 - w) z(mean z(to)); to_z = precomputed z(mean z(to))."""
        return w_ft * self.zmean(ft_list) + (1 - w_ft) * to_z

    def nd(self, s):
        return ndcg_per_feed(s, self.y, self.off, self.M)

    def stat(self, d, mask):
        """paired mean delta, per-feed SE, user-clustered SE over feeds in mask."""
        d = d[mask]
        n = len(d)
        mu = d.mean()
        se = d.std(ddof=1) / np.sqrt(n)
        cl = self.cluster[mask]
        _, inv = np.unique(cl, return_inverse=True)
        r = np.bincount(inv, d - mu)
        se_c = np.sqrt((r ** 2).sum()) / n
        return float(mu), float(se), float(se_c), int(n)


def load_to(m):
    return np.load(PREDS / f"nn_{m}_trainonly_val.npy").astype(np.float64)


def load_base_ft(m):
    return np.load(PREDS / f"nn_{m}_n2cpv3_val_oof.npy").astype(np.float64)
