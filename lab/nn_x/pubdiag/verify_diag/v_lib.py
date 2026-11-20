"""Independent re-implementation of pool building / per-feed NDCG for the adversarial review.
Family sums are cached in cache/ (this dir only)."""
import os, sys
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ.setdefault(k, "12")
sys.path.insert(0, "<repo>/src")
import numpy as np
from pathlib import Path
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

P = Path("<repo>/preds")
H = P / "holdout"
D = Path("<repo>/nn_x/pubdiag/verify_diag")
C = D / "cache"; C.mkdir(exist_ok=True)

FAM = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1"] + [f"v14_s{i}" for i in range(92, 97)],
    "F7": [f"v17_pre2s_s{i}" for i in range(101, 108)],
    "F9": [f"v22_pre2s_lr1_s{i}" for i in range(401, 407)],
    "F10": [f"v23_gru_pre2s_s{i}" for i in range(501, 507)],
    "F12": [f"v26_d192_pre2s_s{i}" for i in range(701, 708)],
    "F7x": [f"r6_cd_m02_r{i}" for i in range(101, 108)],
    "F7x70": [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)],
    "F7b": [f"r6_cd_c0_r{i}" for i in range(101, 108)],
}
POOLS = {
    "Cd": ["F5cd", "F6", "F7"],
    "Pall": ["F5cd", "F6", "F7", "F9", "F10", "F12"],
    "PF7x": ["F5cd", "F6", "F7x", "F9", "F10", "F12"],
    "G3": ["F10", "F7x", "F7x70"],
    "M2": ["F7x", "F7x70"],
}

_meta = {}


def meta(split):
    if split not in _meta:
        if split == "holdout":
            M, off = np.load(H / "holdout_meta.npy")
            _meta[split] = (None, M.astype(np.int64), off.astype(np.int64))
        else:
            _meta[split] = feed_meta(split)
    return _meta[split]


def z(split, x):
    f, M, o = meta(split)
    return feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1]


def member_file(m, split, half):
    if split == "holdout":
        return H / f"nn_{m}_holdout.npy"
    if half == "to":
        return P / f"nn_{m}_trainonly_{split}.npy"
    return P / f"nn_{m}_n2cpv3_{'val_oof' if split == 'val' else 'test'}.npy"


def fam_sum(fam, split, half):
    fn = C / f"fs_{fam}_{split}_{half}.npy"
    if fn.exists():
        return np.load(fn)
    s = None
    for m in FAM[fam]:
        v = z(split, np.load(member_file(m, split, half)))
        s = v.astype(np.float64) if s is None else s + v
    np.save(fn, s.astype(np.float32))
    return s.astype(np.float32)


def member_z(m, split, half):
    return z(split, np.load(member_file(m, split, half)))


def pool_half(fams, split, half):
    n = sum(len(FAM[f]) for f in fams)
    return sum(fam_sum(f, split, half).astype(np.float64) for f in fams) / n


def pool_final(fams, split):
    if split == "holdout":
        return pool_half(fams, split, "to")
    return 0.5 * z(split, pool_half(fams, split, "ft")) + 0.5 * z(split, pool_half(fams, split, "to"))


def y_of(split):
    if split == "val":
        return np.load(FLAT / "val/cand_target.npy")
    if split == "holdout":
        return np.load(H / "holdout_y.npy")
    raise ValueError


def pf(split, s):
    f, M, o = meta(split)
    return ndcg_per_feed(s, y_of(split), o, M)


def groups_val():
    f, M, o = meta("val")
    u = f["user_id"].to_numpy().astype(np.int64)
    g = u.copy()
    z0 = u == 0
    g[z0] = -1 - np.arange(z0.sum())  # user 0 = singleton groups
    return g


def se_iid(d):
    return d.std(ddof=1) / np.sqrt(len(d))


def se_clu(d, g):
    _, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=d - d.mean())
    n = len(d)
    G = len(s)
    return np.sqrt((s ** 2).sum() * G / (G - 1)) / n
