"""lb_stats recheck: shared loaders. CPU only, 12 threads, read-only on every prediction file; writes only under ./out.
Files = every submission whose final score can be rebuilt (0.5 z(ft) + 0.5 z(to) of preds/ens_*; old pools from the
landscape cache nn_x/audit/valpub/landscape/out). Public scores are data (docs/04_SUBMISSIONS.md + today's W8 / S16)."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
P = ROOT / "preds"
W5 = ROOT / "nn_x/w5/preds"
LAND = ROOT / "nn_x/audit/valpub/landscape/out"
HERE = ROOT / "nn_x/w12/recheck/lb_stats"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True, parents=True)
BOUND = 1790497920
MSK = 10800
NPUB, NPRIV, NVAL = 8562, 20033, 14423

PUBLIC = {"v3": 0.3739, "v4": 0.3772, "v5": 0.3802, "v10": 0.38786, "Cd": 0.39208, "G3": 0.392820, "PF7x": 0.392723,
          "M2": 0.392434, "S2": 0.392473, "TOonly": 0.392346, "Pall": 0.392436, "Q1a": 0.392886, "Q2a": 0.393205,
          "C3": 0.393454, "C1": 0.393394, "C2": 0.392754, "W8": 0.394486, "S16": 0.394078}
VAL_REC = {"v3": 0.3816, "v4": 0.3839, "v5": 0.3869, "v10": 0.39399, "Cd": 0.39901, "G3": 0.40032, "PF7x": 0.39974,
           "M2": 0.40094, "S2": 0.40088, "TOonly": 0.40035, "Pall": 0.39896, "Q1a": 0.40068, "Q2a": 0.40061,
           "C3": 0.40101, "C1": 0.40089, "C2": 0.40051, "W8": 0.40165, "S16": 0.40173}
ENS = {"Cd": "Cd", "G3": "G3", "PF7x": "PF7x", "Pall": "Pall", "Q1a": "Q1apopfree", "Q2a": "Q2apop1", "C1": "Q2a8pop1",
       "C3": "Q2a8s3", "C2": "G3s3", "W8": "W8kd", "S16": "S16kd", "W8C5": "W8C5kd", "KD4": "KDkd05", "K5": "K5kd05",
       "K2": "K2kd05"}
CACHED = ["M2", "S2", "TOonly", "v10", "v5"]        # landscape cache: score_<name>_{val,test}.npy
MODERN = ["Cd", "Pall", "PF7x", "G3", "M2", "S2", "TOonly", "Q1a", "Q2a", "C1", "C2", "C3", "W8", "S16"]   # scored, with arrays
UNSCORED = ["W8C5", "KD4", "K5", "K2"]
OLDARR = ["v5", "v10"]

F10 = [f"v23_gru_pre2s_s{s}" for s in range(501, 507)]
F7x = [f"r6_cd_m02_r{s}" for s in range(101, 108)]
F7x70 = [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)]
F5cd = ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"]
F6 = ["v13_lr1_mix1"] + [f"v14_s{i}" for i in range(92, 97)]
F7 = [f"v17_pre2s_s{i}" for i in range(101, 108)]
F9 = [f"v22_pre2s_lr1_s{i}" for i in range(401, 407)]
F12 = [f"v26_d192_pre2s_s{i}" for i in range(701, 708)]
G3M = F10 + F7x + F7x70
KD = {a: [f"w5_e10_{a}_s{s}" for s in (951, 952, 953, 954)] for a in ("kd05", "kd05t25", "kd05w192", "kd05t25w192")}
HOLD_MEMBERS = {
    "Cd": F5cd + F6 + F7, "Pall": F5cd + F6 + F7 + F9 + F10 + F12, "PF7x": F5cd + F6 + F7x + F9 + F10 + F12,
    "G3": G3M, "M2": F7x + F7x70, "TOonly": G3M, "C2": G3M,
    "Q1a": G3M + [f"abl_e10_xdrop_pop_s{s}" for s in range(951, 955)],
    "Q2a": G3M + [f"abl_e10_pop1_s{s}" for s in range(951, 955)],
    "C1": G3M + [f"abl_e10_pop1_s{s}" for s in range(951, 959)],
    "C3": G3M + [f"abl_e10_pop1_s{s}" for s in range(951, 959)],
    "W8": KD["kd05w192"] + KD["kd05t25w192"],
    "S16": KD["kd05"] + KD["kd05t25"] + KD["kd05w192"] + KD["kd05t25w192"],
    "KD4": KD["kd05"],
}

_META = {}


def meta(split):
    if split not in _META:
        f = pl.read_parquet(FLAT / split / "feeds.parquet")
        _META[split] = (f, f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64))
    return _META[split]


def zf(split, x):
    f, M, off = meta(split)
    return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)


def final_score(name, split):
    if name in ENS:
        t = ENS[name]
        return 0.5 * zf(split, np.load(P / f"ens_{t}_ft_{split}.npy")) + 0.5 * zf(split, np.load(P / f"ens_{t}_to_{split}.npy"))
    return np.load(LAND / f"score_{name}_{split}.npy").astype(np.float64)


def half_score(name, half, split):
    return zf(split, np.load(P / f"ens_{ENS[name]}_{half}_{split}.npy"))


def val_y():
    return np.load(FLAT / "val/cand_target.npy").astype(np.float64)


def nd_val(score):
    f, M, off = meta("val")
    return ndcg_per_feed(np.asarray(score, np.float64), val_y(), off, M)


_H = {}


def hold():
    if not _H:
        Mh, oh = np.load(P / "holdout/holdout_meta.npy")
        _H.update(M=Mh.astype(np.int64), off=oh.astype(np.int64), y=np.load(P / "holdout/holdout_y.npy").astype(np.float64))
    return _H


def hold_path(m):
    for d in (P / "holdout", W5 / "holdout"):
        p = d / f"nn_{m}_holdout.npy"
        if p.exists():
            return p
    raise FileNotFoundError(m)


def hold_pool_nd(members):
    h = hold()
    z = np.mean([feed_rank_z(h["off"], h["M"], np.load(hold_path(m)).astype(np.float32))[:, 1] for m in members], 0)
    return ndcg_per_feed(z.astype(np.float64), h["y"], h["off"], h["M"]), z


def val_day():
    f, M, off = meta("val")
    return (f["timestamp"].to_numpy().astype(np.int64) + MSK) // 86400


def test_parts():
    f, M, off = meta("test")
    ts = f["timestamp"].to_numpy().astype(np.int64)
    return ts < BOUND, (ts + MSK) // 86400, M


EDGES = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250])


def mbin(M):
    return np.searchsorted(EDGES, M, side="right")


def mix_weights(Msrc, Mdst):
    """weights on source feeds so that the M-bin mix equals the destination's (src/calib.py binning)."""
    bs, bd = mbin(Msrc), mbin(Mdst)
    ps = np.bincount(bs, minlength=len(EDGES) + 1) / len(bs)
    pd_ = np.bincount(bd, minlength=len(EDGES) + 1) / len(bd)
    w = (pd_ / np.maximum(ps, 1e-12))[bs]
    return w / w.mean()


def se(d):
    d = np.asarray(d, np.float64)
    return d.std(ddof=1) / np.sqrt(len(d))


def wmean_se(d, w):
    w = w / w.sum()
    m = (w * d).sum()
    return m, np.sqrt((w ** 2 * (d - m) ** 2).sum())
