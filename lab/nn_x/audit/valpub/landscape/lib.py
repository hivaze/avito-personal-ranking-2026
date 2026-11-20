"""Shared helpers for the solution-landscape analysis (nn_x/audit/valpub/landscape). CPU only, 12 threads.
Reads preds/, subs/, eda/errors/segments, nn_x/pubdiag/shift descriptors; writes only under this directory's out/.

Pools (final per-candidate score on val / test; spec = 0.5 z(ft) + 0.5 z(to), per-feed z = stack_val.feed_rank_z[:,1]):
  Cd, Pall, PF7x, G3, M2             : preds/ens_<name>_{ft,to}_{val,test}.npy
  S2    (neural stacker on G3)       : preds/nnstk_g3gate3r_{val,test}.npy (final scores)
  TOonly (G3 train-only half)        : preds/ens_G3_toonly_{val,test}.npy
  N1 (G3 + NEW12), NEW12, F10only    : nn_x/audit/publicprobe/out/cand_*_{val,test}.npy (final scores)
  U1 (G3 + W5base s951-954), U3 (U1 + v30_f7_m02_s901-906): rebuilt from member files (member-equal spec)
  S1 (hfit P11 W1to), PoP3           : preds/ens_hfitP11W1to_*, preds/agg_S3_PoP3_* (unscored stacking candidates)
  old generations v3/v4/v5/v10       : test ranks from subs/*.parquet (exact); val arrays rebuilt where the recipe is known
Public scores are CONTEXT (the user asked for this analysis); nothing here selects on them.
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "12"
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/src")
sys.path.insert(0, "<repo>/eda/errors/segments")
sys.path.insert(0, "<repo>/nn_x/pubdiag/shift")
from metrics import ndcg_per_feed  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402
import shiftlib  # noqa: E402

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
P = ROOT / "preds"
H = P / "holdout"
PP = ROOT / "nn_x/audit/publicprobe/out"
HERE = ROOT / "nn_x/audit/valpub/landscape"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True, parents=True)
BOUND = 1790497920

FAM = dict(shiftlib.FAM)
FAM["W5base"] = [f"w5_e10_base_s{s}" for s in range(951, 955)]
FAM["V30"] = [f"v30_f7_m02_s{s}" for s in range(901, 907)]
FAM["NEW12"] = ["v27_gru_m02_s511", "v27_gru_m02_s513", "v27_gru_m02_s515", "v28_lr1_m02_s411", "v28_lr1_m02_s413",
                "v28_lr1_m02_s414", "v28_lr1_m02_s415", "v28_lr1_m02_s417", "v29_d192_m02_s711", "v29_d192_m02_s712",
                "v30_f7_m02_s901", "v30_f7_m02_s902"]
MEMBERS = {
    "Cd": FAM["F5cd"] + FAM["F6"] + FAM["F7"],
    "Pall": FAM["F5cd"] + FAM["F6"] + FAM["F7"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
    "PF7x": FAM["F5cd"] + FAM["F6"] + FAM["F7x"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
    "G3": FAM["F10"] + FAM["F7x"] + FAM["F7x70"],
    "M2": FAM["F7x"] + FAM["F7x70"],
    "TOonly": FAM["F10"] + FAM["F7x"] + FAM["F7x70"],
    "S2": FAM["F10"] + FAM["F7x"] + FAM["F7x70"],
    "S1": FAM["F5cd"] + FAM["F6"] + FAM["F7x"] + FAM["F9"] + FAM["F10"] + FAM["F12"],  # hfit on the PF7x families (TO-weighted)
    "PoP3": FAM["F5cd"] + FAM["F6"] + FAM["F7"] + FAM["F7x"] + FAM["F7x70"] + FAM["F9"] + FAM["F10"] + FAM["F12"],  # union (approx.)
    "N1": FAM["F10"] + FAM["F7x"] + FAM["F7x70"] + FAM["NEW12"],
    "NEW12": FAM["NEW12"],
    "F10only": FAM["F10"],
    "U1": FAM["F10"] + FAM["F7x"] + FAM["F7x70"] + FAM["W5base"],
    "U3": FAM["F10"] + FAM["F7x"] + FAM["F7x70"] + FAM["W5base"] + FAM["V30"],
}
RECOOL = set(FAM["F7x"] + FAM["F7x70"])      # m02 re-cooled from F7 roots (val-period-adjacent cooldown)
M02 = RECOOL | set(FAM["NEW12"]) | set(FAM["W5base"]) | set(FAM["V30"])  # m02 recipe (re-cool or from scratch)
FAMILY_OF = {m: f for f, ms in FAM.items() for m in ms}
FAMILY_OF.update({m: "NEW12" for m in FAM["NEW12"]})

# exact public scores (context only) and recorded holdouts
PUBLIC = {"Cd": 0.39208, "Pall": 0.392436, "PF7x": 0.392723, "G3": 0.392820, "M2": 0.392434, "S2": 0.392473,
          "TOonly": 0.392346, "v3": 0.3739, "v4": 0.3772, "v5": 0.3802, "v10": 0.38786}
VAL_RECORDED = {"Cd": 0.39901, "Pall": 0.39896, "PF7x": 0.39974, "G3": 0.40032, "M2": 0.40094, "S2": 0.40088,
                "TOonly": 0.40035, "v3": 0.3816, "v4": 0.3839, "v5": 0.3869, "v10": 0.39399}
HOLD_RECORDED = {"Cd": 0.32319, "Pall": 0.32335, "PF7x": 0.32360, "G3": 0.32379, "M2": 0.32361}
SCORED = ["Cd", "Pall", "PF7x", "G3", "M2", "S2", "TOonly"]
UNSCORED = ["N1", "U1", "U3", "S1", "PoP3", "NEW12", "F10only"]
OLD = ["v3", "v4", "v5", "v10"]


def zf(off, M, x):
    return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)


_META = {}


def meta(split):
    if split not in _META:
        f = pl.read_parquet(FLAT / split / "feeds.parquet")
        _META[split] = (f, f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64))
    return _META[split]


def val_labels():
    return np.load(FLAT / "val/cand_target.npy").astype(np.float64)


def public_mask():
    f, M, off = meta("test")
    return f["timestamp"].to_numpy().astype(np.int64) < BOUND


def member_pool(members, split):
    """member-equal spec: ft = mean_m z(ft_m), to = mean_m z(to_m); final = 0.5 z(ft) + 0.5 z(to)."""
    f, M, off = meta(split)
    ftn, ton = ("n2cpv3_val_oof", "trainonly_val") if split == "val" else ("n2cpv3_test", "trainonly_test")
    ft = np.mean([zf(off, M, np.load(member_path(m, ftn))) for m in members], 0)
    to = np.mean([zf(off, M, np.load(member_path(m, ton))) for m in members], 0)
    return 0.5 * zf(off, M, ft) + 0.5 * zf(off, M, to), ft, to


def member_path(m, suf):
    for d in (P, ROOT / "nn_x/w5/preds"):
        p = d / f"nn_{m}_{suf}.npy"
        if p.exists():
            return p
    raise FileNotFoundError(m, suf)


def holdout_member_path(m):
    for d in (H, ROOT / "nn_x/w5/preds/holdout"):
        p = d / f"nn_{m}_holdout.npy"
        if p.exists():
            return p
    raise FileNotFoundError(m)


def subs_to_flat(name):
    """test score array in flat candidate order from a submission parquet (feed_id, item_id, score)."""
    f, M, off = meta("test")
    d = pl.read_parquet(ROOT / "subs" / name)
    fid_of_feed = dict(zip(f["feed_id"].to_numpy().tolist(), range(len(M))))
    feed_fidx = np.array([fid_of_feed[x] for x in d["feed_id"].to_numpy().tolist()], np.int64)
    item = d["item_id"].to_numpy().astype(np.int64)
    key_sub = feed_fidx * (1 << 32) + item
    flat_fid = np.repeat(np.arange(len(M)), M)
    flat_item = np.load(FLAT / "test/cand_item_id.npy").astype(np.int64)
    key_flat = flat_fid * (1 << 32) + flat_item
    order = np.argsort(key_sub)
    pos = np.searchsorted(key_sub[order], key_flat)
    assert (key_sub[order][pos] == key_flat).all(), "submission rows do not cover the flat candidates"
    return d["score"].to_numpy().astype(np.float32)[order][pos]


OLD_SUBS = {"v3": "sub_v3_nnens_stk_gb.parquet", "v4": "sub_v4_resstk_nnft.parquet", "v5": "sub_v5_nnfinal2_res2.parquet",
            "v10": "sub_v10_nnfinal4mix_pre.parquet"}


def pool_scores(name, split):
    """final per-candidate score of pool `name` on split (cached under out/)."""
    cache = OUT / f"score_{name}_{split}.npy"
    if cache.exists():
        return np.load(cache)
    f, M, off = meta(split)
    if name in ("Cd", "Pall", "PF7x", "G3", "M2"):
        s = 0.5 * zf(off, M, np.load(P / f"ens_{name}_ft_{split}.npy")) + 0.5 * zf(off, M, np.load(P / f"ens_{name}_to_{split}.npy"))
    elif name == "S2":
        s = np.load(P / f"nnstk_g3gate3r_{split}.npy").astype(np.float64)
    elif name == "TOonly":
        s = np.load(P / f"ens_G3_toonly_{split}.npy").astype(np.float64)
    elif name == "N1":
        s = np.load(PP / f"cand_N1_G3+NEW_{split}.npy").astype(np.float64)
    elif name in ("NEW12", "F10only"):
        s = np.load(PP / f"cand_{name}_{split}.npy").astype(np.float64)
    elif name in ("U1", "U3"):
        s = member_pool(MEMBERS[name], split)[0]
    elif name == "S1":
        s = 0.5 * zf(off, M, np.load(P / f"ens_hfitP11W1to_ft_{split}.npy")) + 0.5 * zf(off, M, np.load(P / f"ens_hfitP11W1to_to_{split}.npy"))
    elif name == "PoP3":
        s = np.load(P / f"agg_S3_PoP3_{split}.npy").astype(np.float64)
    elif name in OLD:
        if split == "test":
            s = subs_to_flat(OLD_SUBS[name]).astype(np.float64)
        elif name == "v10":   # 0.5 z(nn_final4 OOF) + 0.5 z(nn_final4 train-only) (+0.1 z(pre) was neutral; sub recipe per STATUS_LOG 11-11 3e)
            s = 0.5 * zf(off, M, np.load(P / "nn_final4_val_oof.npy")) + 0.5 * zf(off, M, np.load(P / "nn_final4_trainonly_val.npy"))
        elif name == "v5":    # 0.5 z(nn_final2 OOF) + 0.5 z(stk2_res2_r3) (quarantined stacker; the recipe of record)
            s = 0.5 * zf(off, M, np.load(P / "nn_final2_val_oof.npy")) + 0.5 * zf(off, M, np.load(ROOT / "preds_quarantine/stk2_res2_r3_val.npy"))
        else:
            raise FileNotFoundError(f"no val recipe for {name}")
    else:
        raise KeyError(name)
    s = s.astype(np.float32)
    np.save(cache, s)
    return s


def half_scores(name, half, split):
    """z of one half (ft or to) of a pool whose halves are stored (Cd, Pall, PF7x, G3, M2, U1, U3, S1)."""
    f, M, off = meta(split)
    if name in ("Cd", "Pall", "PF7x", "G3", "M2"):
        return zf(off, M, np.load(P / f"ens_{name}_{half}_{split}.npy"))
    if name == "S1":
        return zf(off, M, np.load(P / f"ens_hfitP11W1to_{half}_{split}.npy"))
    if name in ("U1", "U3", "N1", "NEW12", "F10only"):
        s, ft, to = member_pool(MEMBERS[name], split)
        return zf(off, M, ft if half == "ft" else to)
    raise KeyError(name)


def perfeed_val(name):
    cache = OUT / f"nd_{name}_val.npy"
    if cache.exists():
        return np.load(cache)
    f, M, off = meta("val")
    nd = ndcg_per_feed(pool_scores(name, "val"), val_labels(), off, M)
    np.save(cache, nd)
    return nd


_HOLD = {}


def holdout():
    """(train feed rows df, M, off, y, ts, uid) of the 30K holdout."""
    if not _HOLD:
        hold, fh, M, off = shiftlib.holdout_index()
        y = np.load(H / "holdout_y.npy").astype(np.float64)
        _HOLD.update(dict(idx=hold, f=fh, M=M, off=off, y=y, ts=fh["timestamp"].to_numpy().astype(np.int64),
                          uid=fh["user_id"].to_numpy().astype(np.int64)))
    return _HOLD


def perfeed_holdout(name):
    """holdout per-feed NDCG of the pool's TRAIN-ONLY members (z-mean); None if a member lacks holdout preds."""
    cache = OUT / f"nd_{name}_holdout.npy"
    if cache.exists():
        return np.load(cache)
    if name in ("S2", "S1", "PoP3") or name in OLD:
        return None
    hd = holdout()
    try:
        arrs = [np.load(holdout_member_path(m)) for m in MEMBERS[name]]
    except FileNotFoundError:
        return None
    s = shiftlib.zmean(arrs, hd["M"])
    nd = ndcg_per_feed(s.astype(np.float32), hd["y"], hd["off"], hd["M"])
    np.save(cache, nd)
    return nd


def within_feed_corr(a, b, M, off):
    """per-feed Pearson correlation of two flat score arrays (after per-feed z)."""
    fid = np.repeat(np.arange(len(M)), M)
    za, zb = zf(off, M, a), zf(off, M, b)
    return np.bincount(fid, za * zb, len(M)) / M


def top_item(s, M, off):
    """index (within feed) of the top-scored candidate per feed."""
    fid = np.repeat(np.arange(len(M)), M)
    order = np.lexsort((-np.asarray(s, np.float64), fid))
    return order[off] - off


# ----------------------------------------------------------------------------- descriptors
def descriptors(split):
    """label-free per-feed descriptors with common bins for val and test (+ public/private flag on test)."""
    f, M, off = meta(split)
    ts = f["timestamp"].to_numpy().astype(np.int64)
    seg = pl.read_parquet(ROOT / "eda/errors/segments" / ("feed_table.parquet" if split == "val" else "test_segments.parquet"))
    assert (seg["fidx"].to_numpy() == np.arange(len(M))).all()
    desc = pl.read_parquet(ROOT / f"nn_x/pubdiag/shift/desc_{split}.parquet")
    d = {
        "M": M, "H": f["H"].to_numpy().astype(np.int64), "uid0": f["user_id"].to_numpy() == 0,
        "n_train_feeds": seg["n_train_feeds"].to_numpy().astype(np.int64),
        "cold_share": seg["cold_share"].to_numpy(), "luo_cold_share": desc["luo_cold_share"].to_numpy(),
        "hour": shiftlib.msk_hour(ts), "dow": shiftlib.weekday(ts), "ts": ts,
        "n_cat": seg["n_cat"].to_numpy().astype(np.int64), "max_cat_share": seg["max_cat_share"].to_numpy(),
        "hist_top_cat_share": seg["hist_top_cat_share"].to_numpy(), "cand_cat_in_hist_share": seg["cand_cat_in_hist_share"].to_numpy(),
        "top_cat": desc["top_cat"].to_numpy().astype(np.int64),
    }
    if split == "test":
        d["public"] = ts < BOUND
    else:
        d["npos"] = seg["npos"].to_numpy().astype(np.int64)
        d["pos_cold_any"] = seg["pos_cold_any"].to_numpy(); d["pos_cold_all"] = seg["pos_cold_all"].to_numpy()
        d["n_pos_cat"] = seg["n_pos_cat"].to_numpy().astype(np.int64)
        d["pos_cat_in_hist"] = seg["pos_cat_in_hist"].to_numpy()
    return d


def cut(x, edges, labels):
    i = np.searchsorted(np.asarray(edges, np.float64), np.asarray(x, np.float64), side="left")
    return np.array(labels, dtype=object)[i]


def slicings(d, g3_pos_pct=None):
    """dict name -> per-feed label array; label-free ones exist for val and test, label ones only for val."""
    S = {}
    S["M"] = cut(d["M"], [10, 25, 50, 100, 200], ["M5-10", "M11-25", "M26-50", "M51-100", "M101-200", "M201-300"])
    S["H"] = cut(d["H"], [0, 19, 63, 191, 255], ["H0", "H1-19", "H20-63", "H64-191", "H192-255", "H256"])
    S["user"] = np.where(d["uid0"], "uid0", cut(d["n_train_feeds"], [0, 1, 4], ["tr0", "tr1", "tr2-4", "tr5+"]))
    S["cold_share"] = cut(d["cold_share"], [0.15, 0.3, 0.45], ["cold<15%", "cold15-30%", "cold30-45%", "cold>=45%"])
    S["luo_cold"] = cut(d["luo_cold_share"], [0.2, 0.35, 0.5, 0.65], ["luo<20%", "luo20-35%", "luo35-50%", "luo50-65%", "luo>=65%"])
    S["hour"] = cut(d["hour"], [5, 9, 13, 17], ["h00-05", "h06-09", "h10-13", "h14-17", "h18-23"])
    S["weekend"] = np.where(d["dow"] >= 5, "weekend", "weekday")
    S["n_cat"] = cut(d["n_cat"], [4, 8, 12, 18], ["ncat1-4", "ncat5-8", "ncat9-12", "ncat13-18", "ncat19+"])
    S["max_cat_share"] = cut(d["max_cat_share"], [0.3, 0.5, 0.7, 0.9], ["mcs<=.3", "mcs.3-.5", "mcs.5-.7", "mcs.7-.9", "mcs>.9"])
    S["cand_cat_in_hist"] = cut(d["cand_cat_in_hist_share"], [0.5, 0.8, 0.95], ["cih<=.5", "cih.5-.8", "cih.8-.95", "cih>.95"])
    top = [13, 2, 3, 50, 6, 7, 10, 15, 11, 48]
    tc = d["top_cat"]
    S["top_cat"] = np.where(np.isin(tc, top), np.char.add("cat", tc.astype(str)), "cat_other")
    if "npos" in d:
        S["day"] = np.where(d["dow"] == 1, "Tue", "Thu")
        S["npos"] = cut(d["npos"], [1, 2, 3, 5], ["npos1", "npos2", "npos3", "npos4-5", "npos6+"])
        S["pos_cold"] = np.where(d["pos_cold_all"], "pos_all_cold", np.where(d["pos_cold_any"], "pos_some_cold", "pos_warm"))
        S["pos_cat_in_hist"] = np.where(d["pos_cat_in_hist"], "poscat_in_hist", "poscat_NOT_in_hist")
        if g3_pos_pct is not None:
            S["g3_pos_rank"] = cut(g3_pos_pct, [0.0, 0.05, 0.15, 0.4], ["G3 top1", "G3 <=5%", "G3 5-15%", "G3 15-40%", "G3 >40%"])
    return S


def best_pos_pct(s, y, M, off):
    """rank percentile (0 = top) of the best-ranked positive under score s, per feed (NaN if no positive)."""
    fid = np.repeat(np.arange(len(M)), M)
    order = np.lexsort((y, -np.asarray(s, np.float64), fid))      # pessimistic ties: negatives first
    rank = np.arange(len(order)) - np.repeat(off, M)
    ys = y[order]
    out = np.full(len(M), np.nan)
    pos = ys > 0
    first = np.full(len(M), -1)
    # first positive position per feed
    idx = np.where(pos)[0]
    fids = fid[order][idx]
    uniq, firstidx = np.unique(fids, return_index=True)
    first[uniq] = rank[idx[firstidx]]
    m = first >= 0
    out[m] = first[m] / np.maximum(M[m] - 1, 1)
    return out


def mse(x, w=None):
    return shiftlib.mse(x, w)


def fmt(m, se):
    return f"{m:+.5f}±{se:.5f}"
