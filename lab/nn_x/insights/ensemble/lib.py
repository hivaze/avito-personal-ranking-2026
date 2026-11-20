"""Shared helpers for the ensemble / FT-stage insight analysis (nn_x/insights/ensemble). CPU only, read-only on
preds/ and data/. Pool spec = the lead's build_pool.py: per-feed z of every member (population sd), branch = mean
of member z, final = 0.5*z(FT branch) + 0.5*z(TO branch)."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
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
P = ROOT / "preds"
OUT = ROOT / "nn_x/insights/ensemble"
CACHE = Path("/tmp/scratchpad/ens")
CACHE.mkdir(parents=True, exist_ok=True)
BOUND = 1790497920  # public = test feeds with ts < BOUND

FAM = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84",
             "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1"] + [f"v14_s{i}" for i in range(92, 97)],
    "F7": [f"v17_pre2s_s{i}" for i in range(101, 108)],
    "F9": [f"v22_pre2s_lr1_s{i}" for i in range(401, 407)],
    "F10": [f"v23_gru_pre2s_s{i}" for i in range(501, 507)],
    "F12": [f"v26_d192_pre2s_s{i}" for i in range(701, 708)],
    "F7b": [f"r6_cd_c0_r{i}" for i in range(101, 108)],
    "F7x": [f"r6_cd_m02_r{i}" for i in range(101, 108)],
    "F7x70": [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)],
    "V27g": ["v27_gru_m02_s511", "v27_gru_m02_s513", "v27_gru_m02_s515"],
    "V28l": [f"v28_lr1_m02_s{i}" for i in range(411, 418)],
    "V29d": ["v29_d192_m02_s711", "v29_d192_m02_s712", "v29_d192_m02_s713"],
}
# plateau (pre-cooldown) F7 snapshots with their own n2cpv3 fine-tune: the "no cooldown" end of the spectrum
FAM_EXTRA = {"F7p": [f"v17_pre2s_s{i}@77500" for i in range(101, 107)]}
OLD = ["F5cd", "F6", "F7", "F9", "F10", "F12"]
M02_RECOOL = ["F7x", "F7x70"]
NEW = ["V27g", "V28l", "V29d"]
POOLS = {
    "G3": FAM["F10"] + FAM["F7x"] + FAM["F7x70"],
    "PF7x": FAM["F5cd"] + FAM["F6"] + FAM["F7x"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
    "M2": FAM["F7x"] + FAM["F7x70"],
    "Cd": FAM["F5cd"] + FAM["F6"] + FAM["F7"],
}


def add_v30():
    """v30_f7_m02 (from-scratch F7 recipe + m02) members, if their preds are complete"""
    ms = [f"v30_f7_m02_s{i}" for i in range(901, 907)]
    ok = [m for m in ms if all((P / f"nn_{m}_{s}.npy").exists() for s in
                               ("trainonly_val", "trainonly_test", "n2cpv3_val_oof", "n2cpv3_test"))
          and (P / "holdout" / f"nn_{m}_holdout.npy").exists()]
    if ok:
        FAM["V30f"] = ok
        if "V30f" not in NEW:
            NEW.append("V30f")
    return ok


_META = {}


def meta(split):
    if split not in _META:
        if split == "holdout":
            m = np.load(P / "holdout/holdout_meta.npy")
            M, off = m[0].astype(np.int64), m[1].astype(np.int64)
            y = np.load(P / "holdout/holdout_y.npy").astype(np.float64)
            f = None
        else:
            f = pl.read_parquet(FLAT / split / "feeds.parquet")
            M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
            y = np.load(FLAT / split / "cand_target.npy").astype(np.float64) if split == "val" else None
        fid = np.repeat(np.arange(len(M)), M)
        _META[split] = dict(f=f, M=M, off=off, y=y, fid=fid, n=len(M))
    return _META[split]


def zfeed(s, split):
    m = meta(split); fid, M, n = m["fid"], m["M"], m["n"]
    s = np.asarray(s, np.float64)
    mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]


def load_member(m, split, branch):
    """branch 'to' or 'ft'; split val/test/holdout (holdout: to only)"""
    if split == "holdout":
        assert branch == "to"
        return np.load(P / "holdout" / f"nn_{m}_holdout.npy")
    suf = {("val", "to"): "trainonly_val", ("test", "to"): "trainonly_test",
           ("val", "ft"): "n2cpv3_val_oof", ("test", "ft"): "n2cpv3_test"}[(split, branch)]
    return np.load(P / f"nn_{m}_{suf}.npy")


def member_z(m, split, branch):
    """cached per-feed z of one member (float32)"""
    fn = CACHE / f"z_{m}_{split}_{branch}.npy"
    if fn.exists():
        return np.load(fn)
    z = zfeed(load_member(m, split, branch), split).astype(np.float32)
    np.save(fn, z)
    return z


def branch(members, split, br):
    return np.mean([member_z(m, split, br).astype(np.float64) for m in members], 0)


def final(members, split, w_ft=0.5):
    if split == "holdout":
        return branch(members, split, "to")
    return w_ft * zfeed(branch(members, split, "ft"), split) + (1 - w_ft) * zfeed(branch(members, split, "to"), split)


def pf(score, split="val"):
    m = meta(split)
    return ndcg_per_feed(score, m["y"], m["off"], m["M"])


def pse(d):
    d = np.asarray(d, np.float64)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def fmt(d, scale=1.0):
    mu, se = pse(d)
    return f"{mu * scale:+.5f} +- {se * scale:.5f}"


def cut(x, edges, labels):
    i = np.searchsorted(np.asarray(edges, np.float64), np.asarray(x, np.float64), side="left")
    return np.array(labels, dtype=object)[i]


_SEG = {}


def segments(split):
    """label-free per-feed segments (+ label-defined descriptive ones on val)"""
    if split in _SEG:
        return _SEG[split]
    d = pl.read_parquet(ROOT / f"nn_x/pubdiag/shift/desc_{split}.parquet")
    S = {}
    S["M"] = cut(d["M"], [19, 49, 99, 199], ["M<20", "M20-49", "M50-99", "M100-199", "M200+"])
    S["M50"] = np.where(d["M"].to_numpy() >= 50, "M>=50", "M<50")
    S["luo_cold"] = cut(d["luo_cold_share"], [0.2, 0.35, 0.5, 0.65], ["lc<20%", "lc20-35%", "lc35-50%", "lc50-65%", "lc>=65%"])
    S["cold_bitmap"] = cut(d["cold_share"], [0.15, 0.3, 0.45], ["c<15%", "c15-30%", "c30-45%", "c>=45%"])
    u0 = d["uid0"].to_numpy() > 0; ntr = d["n_train_feeds"].to_numpy()
    S["user"] = np.where(u0, "uid0", cut(ntr, [0, 1, 4], ["tr0", "tr1", "tr2-4", "tr5+"]))
    S["H"] = cut(d["H"], [0, 19, 63, 191, 255], ["H0", "H1-19", "H20-63", "H64-191", "H192-255", "H256"])
    S["n_cat"] = cut(d["n_cat"], [1, 4, 8, 12], ["ncat1", "ncat2-4", "ncat5-8", "ncat9-12", "ncat13+"])
    S["cat_in_hist"] = cut(d["cand_cat_in_hist_share"], [0.5, 0.8, 0.95], ["cih<=.5", "cih.5-.8", "cih.8-.95", "cih>.95"])
    S["max_cat_share"] = cut(d["max_cat_share"], [0.3, 0.6, 0.9], ["mcs<=.3", "mcs.3-.6", "mcs.6-.9", "mcs>.9"])
    ts = d["timestamp"].to_numpy()
    if split == "val":
        day = ((ts + 10800) // 86400).astype(np.int64)
        S["day"] = np.where(day == day.min(), "Tue", "Thu")
        ft = pl.read_parquet(ROOT / "eda/errors/segments/feed_table.parquet")
        S["pos_cold(label)"] = np.where(ft["pos_cold_all"].to_numpy(), "coldpos", np.where(ft["pos_cold_any"].to_numpy(), "somecold", "warmpos"))
        S["npos(label)"] = cut(ft["npos"], [1, 2], ["npos1", "npos2", "npos3+"])
    elif split == "test":
        S["day"] = np.where(ts < BOUND, "public", "private")
    _SEG[split] = S
    return S


def cand_cold(split):
    """per-candidate: item never seen in train candidates or train histories (bitmap def of the error lenses)"""
    fn = CACHE / f"candcold_{split}.npy"
    if fn.exists():
        return np.load(fn)
    NITEM = 111_400_000
    seen = np.zeros(NITEM, bool)
    seen[np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")[:]] = True
    seen[np.load(FLAT / "train/hist_item_id.npy", mmap_mode="r")[:]] = True
    out = {}
    for s in ("val", "test"):
        out[s] = ~seen[np.load(FLAT / s / "cand_item_id.npy")]
        np.save(CACHE / f"candcold_{s}.npy", out[s])
    return out[split]


def user_folds():
    """the FT fold assignment: GroupKFold(5) by user, user-0 feeds as singleton groups (nn/finetune.py --u0_singleton)"""
    from sklearn.model_selection import GroupKFold
    m = meta("val"); uid = m["f"]["user_id"].to_numpy().astype(np.int64)
    g = np.where(uid == 0, -1 - np.arange(len(uid)), uid)
    fold = np.zeros(len(uid), np.int64)
    for k, (_, ho) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(uid)), groups=g)):
        fold[ho] = k
    return fold
