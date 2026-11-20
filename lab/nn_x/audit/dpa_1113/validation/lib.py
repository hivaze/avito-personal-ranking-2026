"""Shared helpers for the validation-representativeness lens (dpa/validation). CPU only, 12 threads.
Read-only on data/, preds/, nn*/; writes only under this scratch directory (out/)."""
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

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
P = ROOT / "preds"
H = P / "holdout"
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
BOUND = 1790497920

FAM = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1"] + [f"v14_s{i}" for i in range(92, 97)],
    "F7": [f"v17_pre2s_s{i}" for i in range(101, 108)],
    "F9": [f"v22_pre2s_lr1_s{i}" for i in range(401, 407)],
    "F10": [f"v23_gru_pre2s_s{i}" for i in range(501, 507)],
    "F12": [f"v26_d192_pre2s_s{i}" for i in range(701, 708)],
    "F7x": [f"r6_cd_m02_r{i}" for i in range(101, 108)],
    "F7x70": [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)],
    "POPFREE": [f"abl_e10_xdrop_pop_s{i}" for i in range(951, 955)],
    "POP1": [f"abl_e10_pop1_s{i}" for i in range(951, 955)],
}
MEMBERS = {
    "Cd": FAM["F5cd"] + FAM["F6"] + FAM["F7"],
    "Pall": FAM["F5cd"] + FAM["F6"] + FAM["F7"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
    "PF7x": FAM["F5cd"] + FAM["F6"] + FAM["F7x"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
    "G3": FAM["F10"] + FAM["F7x"] + FAM["F7x70"],
    "M2": FAM["F7x"] + FAM["F7x70"],
    "TOonly": FAM["F10"] + FAM["F7x"] + FAM["F7x70"],
    "Q1a": FAM["F10"] + FAM["F7x"] + FAM["F7x70"] + FAM["POPFREE"],
    "Q2a": FAM["F10"] + FAM["F7x"] + FAM["F7x70"] + FAM["POP1"],
}
# exact public scores on record (CURRENT_STATUS log 11-12 / 11-13; VAL_PUBLIC_1113 table 1.2)
PUBLIC = {"Cd": 0.39208, "Pall": 0.392436, "PF7x": 0.392723, "G3": 0.392820, "M2": 0.392434, "S2": 0.392473,
          "TOonly": 0.392346, "Q1a": 0.392886, "Q2a": 0.393205}
OLDGEN = {"v3": (0.3816, 0.3739), "v4": (0.3839, 0.3772), "v5": (0.3869, 0.3802), "v10": (0.39399, 0.38786)}  # (val, public)
POOLS = ["Cd", "Pall", "PF7x", "G3", "M2", "S2", "TOonly", "Q1a", "Q2a"]
CONTRASTS = [p for p in POOLS if p != "G3"]
N_PUB, N_PRIV = 8562, 20033

_META = {}


def meta(split):
    if split not in _META:
        f = pl.read_parquet(FLAT / split / "feeds.parquet")
        _META[split] = (f, f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64))
    return _META[split]


def zf(x, M):
    """per-feed z-score (population sd), float64"""
    x = np.asarray(x, np.float64)
    n = len(M); fid = np.repeat(np.arange(n), M)
    mu = np.bincount(fid, x, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, x * x, n) / M - mu ** 2, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]


def zmean(arrs, M):
    acc = None
    for a in arrs:
        z = zf(a, M)
        acc = z if acc is None else acc + z
    return acc / len(arrs)


def day(ts):
    return (np.asarray(ts, np.int64) + 10800) // 86400


def hour(ts):
    return ((np.asarray(ts, np.int64) + 10800) % 86400) // 3600


def weekday(ts):
    return (day(ts) + 3) % 7  # 0 = Mon


ENS = {"Cd": "Cd", "Pall": "Pall", "PF7x": "PF7x", "G3": "G3", "M2": "M2", "Q1a": "Q1apopfree", "Q2a": "Q2apop1"}


def pool_score(name, split, half=None):
    """final per-candidate score (half=None) or the z of one half ('ft'/'to') of a scored pool."""
    f, M, off = meta(split)
    if half is not None:
        if name == "TOonly":
            name = "G3"
            if half == "ft":
                raise KeyError("TOonly has no ft half")
        if name == "S2":
            raise KeyError("S2 has no halves")
        return zf(np.load(P / f"ens_{ENS[name]}_{half}_{split}.npy"), M)
    if name == "S2":
        return np.load(P / f"nnstk_g3gate3r_{split}.npy").astype(np.float64)
    if name == "TOonly":
        return np.load(P / f"ens_G3_toonly_{split}.npy").astype(np.float64)
    return 0.5 * zf(np.load(P / f"ens_{ENS[name]}_ft_{split}.npy"), M) + 0.5 * zf(np.load(P / f"ens_{ENS[name]}_to_{split}.npy"), M)


def val_y():
    return np.load(FLAT / "val/cand_target.npy").astype(np.float64)


def perfeed_val(name, half=None):
    tag = name if half is None else f"{name}_{half}"
    c = OUT / f"pfv_{tag}.npy"
    if c.exists():
        return np.load(c)
    f, M, off = meta("val")
    nd = ndcg_per_feed(pool_score(name, "val", half).astype(np.float32), val_y(), off, M)
    np.save(c, nd)
    return nd


_HOLD = {}


def holdout_index():
    """train feed indices of the 30K holdout (identical to nn_x/pubdiag/shift/shiftlib.holdout_index)."""
    if "idx" in _HOLD:
        return _HOLD["idx"]
    ftr, Mtr, _ = meta("train")
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(len(ftr))).all()
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    p[Mtr < 50] = 0.0
    cand = np.where((p > 0.9) & (Mtr >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    Mh, offh = np.load(H / "holdout_meta.npy")
    assert (Mtr[hold] == Mh).all(), "holdout reconstruction mismatch"
    _HOLD.update(idx=hold, M=Mh.astype(np.int64), off=offh.astype(np.int64), y=np.load(H / "holdout_y.npy").astype(np.float64),
                 p_random=p, pool134=np.setdiff1d(cand, hold))
    return hold


def hold_member(m):
    for d in (H, ROOT / "nn_x/w5/preds/holdout", ROOT / "nn_x/abl/preds/holdout"):
        p = d / f"nn_{m}_holdout.npy"
        if p.exists():
            return np.load(p)
    raise FileNotFoundError(m)


def perfeed_holdout(name):
    """holdout per-feed NDCG of the pool's TRAIN-ONLY members (member-equal z-mean); TOonly = G3; S2 -> None."""
    if name == "S2":
        return None
    if name == "TOonly":
        name = "G3"
    c = OUT / f"pfh_{name}.npy"
    if c.exists():
        return np.load(c)
    holdout_index()
    s = zmean([hold_member(m) for m in MEMBERS[name]], _HOLD["M"])
    nd = ndcg_per_feed(s.astype(np.float32), _HOLD["y"], _HOLD["off"], _HOLD["M"])
    np.save(c, nd)
    return nd


def public_mask():
    f, M, off = meta("test")
    return f["timestamp"].to_numpy().astype(np.int64) < BOUND


def wmean_se(x, w=None):
    x = np.asarray(x, np.float64)
    if w is None:
        return x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    w = np.asarray(w, np.float64)
    m = np.average(x, weights=w)
    return m, np.sqrt((w ** 2 * (x - m) ** 2).sum()) / w.sum()


def neff(w):
    w = np.asarray(w, np.float64)
    return w.sum() ** 2 / (w ** 2).sum()


def psi(a, b, edges=None, nb=10, eps=1e-4, wa=None, wb=None):
    """population stability index of sample a vs reference b (bins = quantiles of b unless given)."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if edges is None:
        edges = np.unique(np.quantile(b, np.linspace(0, 1, nb + 1)[1:-1]))
    ia = np.searchsorted(edges, a, side="right"); ib = np.searchsorted(edges, b, side="right")
    K = len(edges) + 1
    pa = np.bincount(ia, weights=wa, minlength=K).astype(np.float64); pa = pa / pa.sum()
    pb = np.bincount(ib, weights=wb, minlength=K).astype(np.float64); pb = pb / pb.sum()
    pa = np.maximum(pa, eps); pb = np.maximum(pb, eps)
    return float(((pa - pb) * np.log(pa / pb)).sum())


def psi_cat(pa, pb, eps=1e-4):
    pa = np.maximum(np.asarray(pa, np.float64), eps); pb = np.maximum(np.asarray(pb, np.float64), eps)
    pa /= pa.sum(); pb /= pb.sum()
    return float(((pa - pb) * np.log(pa / pb)).sum())


def smd(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float((a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-18))
