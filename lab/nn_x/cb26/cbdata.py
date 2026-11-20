"""cb26 data helpers: the trH (30K random-like train holdout) index, ensemble features and tabular features.
Row orders: val/test = flat order (same as preds/), trH = holdout feeds in sorted order (same as preds/holdout/)."""
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT, FEATS  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402

ROOT = Path("<repo>")
P = ROOT / "preds"
D = ROOT / "nn_x/cb26/data"
D.mkdir(parents=True, exist_ok=True)
FNEW = ROOT / "data/feats_new"

FAM = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1"] + [f"v14_s{i}" for i in range(92, 97)],
    "F7x": [f"r6_cd_m02_r{i}" for i in range(101, 108)],
    "F7x70": [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)],
    "F9": [f"v22_pre2s_lr1_s{i}" for i in range(401, 407)],
    "F10": [f"v23_gru_pre2s_s{i}" for i in range(501, 507)],
    "F12": [f"v26_d192_pre2s_s{i}" for i in range(701, 708)],
}
POOLS = {"G3": ["F10", "F7x", "F7x70"], "PF7x": ["F5cd", "F6", "F7x", "F9", "F10", "F12"]}


def hold_index():
    """train-feed indices of the NN random-like holdout (nn/train.py:156-166 / nn/holdout_eval.py)."""
    f = pl.read_parquet(FLAT / "train/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(len(f))).all()
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    return hold, f


def meta(split):
    """(feeds df, M, off, y) for split in val/test/trH; trH rows are train flat rows of the holdout feeds."""
    if split == "trH":
        hold, f = hold_index()
        M = f["M"].to_numpy().astype(np.int64)[hold]
        off = np.concatenate([[0], np.cumsum(M)[:-1]])
        hm = np.load(P / "holdout/holdout_meta.npy")
        assert (hm[0] == M).all() and (hm[1] == off).all()
        y = np.load(P / "holdout/holdout_y.npy").astype(np.float32)
        return f[hold], M, off, y
    f, M, off = feed_meta(split)
    y = np.load(FLAT / f"{split}/cand_target.npy").astype(np.float32) if split == "val" else None
    return f, M, off, y


def trH_rows():
    hold, f = hold_index()
    M = f["M"].to_numpy().astype(np.int64); o = f["cand_off"].to_numpy().astype(np.int64)
    return np.concatenate([np.arange(o[i], o[i] + M[i]) for i in hold])


def zf(off, M, x):
    return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]


def member_file(m, split, branch):
    if split == "trH":
        assert branch == "to"
        return P / f"holdout/nn_{m}_holdout.npy"
    if branch == "to":
        return P / f"nn_{m}_trainonly_{split}.npy"
    return P / f"nn_{m}_n2cpv3_{'val_oof' if split == 'val' else 'test'}.npy"


def ens_feats(pool, split):
    """Per-row ensemble features (cached). Keys: to, to_sd, fam_<F>_to (minus pool to) [+ ft, ft_sd for val/test]."""
    cache = D / f"ens_{pool}_{split}.npz"
    if cache.exists():
        return dict(np.load(cache))
    _, M, off, _ = meta(split)
    out = {}
    branches = ["to"] if split == "trH" else ["to", "ft"]
    for br in branches:
        fz = {}
        allz = []
        for F in POOLS[pool]:
            zs = [zf(off, M, np.load(member_file(m, split, br))) for m in FAM[F]]
            allz += zs
            fz[F] = np.mean(zs, 0)
        A = np.stack(allz, 0)
        mu = A.mean(0)
        out[br] = mu.astype(np.float32)
        out[f"{br}_sd"] = A.std(0).astype(np.float32)
        for F in POOLS[pool]:
            out[f"fam_{F}_{br}"] = (fz[F] - mu).astype(np.float32)
    if split != "trH":  # consistency with the lead's pool arrays
        ref = np.load(P / f"ens_{pool}_to_{split}.npy")
        assert np.abs(ref - out["to"]).max() < 1e-4, np.abs(ref - out["to"]).max()
    np.savez(cache, **out)
    return out


# ---- tabular feature sets -------------------------------------------------------------------------------------
EXCL_PREFIX = ("it_tr_", "u_tr_", "it_cand_cnt_", "tte_", "pre_", "als_", "hour", "trd_skde")


def resid_feature_names():
    """Leak-safe, parity-OK, train-available features for the residual models (E1/E2)."""
    tr = {p.stem for p in (FEATS / "train").glob("*.npy")}
    va = {p.stem for p in (FEATS / "val").glob("*.npy")}
    names = []
    for n in sorted(tr & va):
        if n.startswith(EXCL_PREFIX):
            continue
        if n.startswith("it_hist_") and "_luo" not in n:  # non-LUO history counts: same-user later snapshots
            continue
        if n.startswith(("pre2_", "prew_")):  # model scores whose train rows have their own parity history
            continue
        names.append(n)
    return names


def load_tab(split, names, rows=None):
    sp = "train" if split == "trH" else split
    if split == "trH" and rows is None:
        rows = trH_rows()
    cols = []
    for n in names:
        p = FEATS / sp / f"{n}.npy"
        if not p.exists():
            p = FNEW / sp / f"{n}.npy"
        x = np.load(p, mmap_mode="r")
        cols.append(np.asarray(x if rows is None else x[rows], dtype=np.float32))
    return np.stack(cols, 1)
