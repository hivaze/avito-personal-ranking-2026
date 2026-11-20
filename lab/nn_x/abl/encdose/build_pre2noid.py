"""ENCODER-DOSE (A): the 6 cvnd_pre2s encoder columns rebuilt from the ID-FREE scores (ledger nn_x/abl/ledger/ledger.md section 10.A).

The live pre2_* columns (reports/pretrain_report.md 8c) are: val / test by pretrain/assemble.py (tags v3At2, v4At2 moment-matched
to the test-side twins v3t2, v4t2 on val; tags v3t2, v4t2 on test), train by pretrain/assemble_train.py (--wtag "w{k}r_{c}"
--rtag "{c}t2", windows 1-3, configs v3 / v4, NaN outside the windows), then pre2_fe_s / pre2_new_s = (x + 20) / 6
(nn/feat_cache). This script re-implements that arithmetic line by line with the RAW KEYS swapped
    fe -> fe_noid, new -> new_noid, hmax32 -> hmax32_noid   (hmax32_noid from nn_x/abl/encdose/score_hmax_noid.py)
and writes pre2n_{fe,new,fe_z,new_z,fe_s,new_s,hmax32}.npy for train / val / test into data/feats/<split>/ (exclusive create)
with hard links in nn/feat_cache/<split>/ (the pop1/build_xp_imp_c50.py pattern; nothing existing is touched).

  --check   self-check: the SAME code on the LIVE keys (fe, new, hmax32) must reproduce the stored pre2_fe / pre2_fe_z / pre2_new /
            pre2_new_z / pre2_hmax32 (max |diff| printed per split; nothing written)
  --hmax32-fallback   pre-declared fallback (ledger 10.A): pre2n_hmax32 := a copy of pre2_hmax32 (the ID-aware column)

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/encdose/build_pre2noid.py [--check] [--hmax32-fallback]
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/src")
from metrics import ndcg  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402

ROOT = Path("<repo>")
P, F, FLAT, PT = ROOT / "preds", ROOT / "data/feats", ROOT / "data/flat", ROOT / "data/pretrain"
WS, CACHE = PT / "winscores", ROOT / "nn/feat_cache"
OUT = ROOT / "nn_x/abl/encdose/out"
RAW = ["fe", "new", "hmax32"]
CONFIGS = ["v3", "v4"]
VAL_TAGS, VAL_CALIB, TEST_TAGS = ["v3At2", "v4At2"], ["v3t2", "v4t2"], ["v3t2", "v4t2"]
WTAG, RTAG = "w{k}r_{c}", "{c}t2"
warnings.filterwarnings("ignore", category=RuntimeWarning)


def affine(x, ref):  # pretrain/assemble.affine, verbatim
    mx, mr = (x > -2) & np.isfinite(x), (ref > -2) & np.isfinite(ref)
    a = ref[mr].astype(np.float64).std() / x[mx].astype(np.float64).std()
    b = ref[mr].astype(np.float64).mean() - a * x[mx].astype(np.float64).mean()
    fn = lambda v: np.where(v > -2, a * v + b, v).astype(np.float32)  # noqa: E731
    return fn, (a, b)


def moments(x):  # pretrain/assemble_train.moments, verbatim
    m = np.isfinite(x) & (x > -2)
    return x[m].astype(np.float64).mean(), x[m].astype(np.float64).std()


def build_valtest(split, tags, key, calib=None):
    """pretrain/assemble.build restricted to fe / fe_z / new / new_z / hmax32, with the raw file key given by `key`."""
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    M, off = f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64)
    acc = {k: [] for k in ("fe", "fe_z", "new", "new_z", "hmax32")}
    cal = {}
    for i, t in enumerate(tags):
        L0 = lambda k: np.load(P / f"pre_{t}_{key[k]}_{split}.npy").astype(np.float32)  # noqa: E731
        if calib:
            fns = {}
            for k in RAW:
                fns[k], ab = affine(L0(k), np.load(P / f"pre_{calib[i]}_{key[k]}_{split}.npy").astype(np.float32))
                cal[f"{t}->{calib[i]} {k}"] = (round(ab[0], 4), round(ab[1], 4))
            L = lambda k: fns[k](L0(k)) if k in fns else L0(k)  # noqa: E731
        else:
            L = L0
        fe, new = L("fe"), L("new")
        zf, zn = feed_rank_z(off, M, fe)[:, 1], feed_rank_z(off, M, new)[:, 1]
        vals = dict(fe=fe, fe_z=zf, new=new, new_z=zn, hmax32=L("hmax32"))
        for k, v in vals.items():
            acc[k].append(v)
    return {k: np.mean(v, axis=0) for k, v in acc.items()}, cal


def window_feats(k, M_all, key):
    """pretrain/assemble_train.window_feats restricted to the five columns, with the raw file key given by `key`."""
    feeds = np.load(PT / f"win{k}/window_feeds.npy")
    rows = np.load(PT / f"win{k}/cand_rows.npy")
    s_rows = np.load(PT / f"win{k}/cand_rows_s60000.npy")
    pos = np.searchsorted(rows, s_rows)
    assert (rows[pos] == s_rows).all()
    M = M_all[feeds]
    off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    acc = {kk: [] for kk in ("fe", "fe_z", "new", "new_z", "hmax32")}
    calib = {}
    for c in CONFIGS:
        L0 = lambda kk: np.load(WS / f"pre_{WTAG.format(k=k, c=c)}_{key[kk]}_win{k}.npy").astype(np.float32)  # noqa: E731
        fns = {}
        for kk in RAW:
            x = L0(kk)
            mx, sx = moments(x[pos])
            mr, sr = moments(np.load(WS / f"pre_{RTAG.format(c=c)}_{key[kk]}_win{k}_s60000.npy"))
            aa = sr / sx; bb = mr - aa * mx
            fns[kk] = (aa, bb)
            calib[f"{c}_{kk}"] = (round(aa, 4), round(bb, 4))
        L = lambda kk: np.where(L0(kk) > -2, fns[kk][0] * L0(kk) + fns[kk][1], L0(kk)).astype(np.float32) if kk in fns else L0(kk)  # noqa: E731
        fe, new = L("fe"), L("new")
        zf, zn = feed_rank_z(off, M, fe)[:, 1], feed_rank_z(off, M, new)[:, 1]
        vals = dict(fe=fe, fe_z=zf, new=new, new_z=zn, hmax32=L("hmax32"))
        for kk, v in vals.items():
            acc[kk].append(v)
    return rows, {kk: np.mean(v, axis=0) for kk, v in acc.items()}, calib


def build_train(key):
    f = pl.read_parquet(FLAT / "train/feeds.parquet")
    M_all = f["M"].to_numpy().astype(np.int64)
    N = len(np.load(FLAT / "train/cand_fidx.npy", mmap_mode="r"))
    full = {kk: np.full(N, np.nan, np.float32) for kk in ("fe", "fe_z", "new", "new_z", "hmax32")}
    cal = {}
    for k in (1, 2, 3):
        rows, out, calib = window_feats(k, M_all, key)
        for kk, v in out.items():
            full[kk][rows] = v
        cal[k] = calib
        print(f"  window {k}: {len(rows)} rows; calib {calib}", flush=True)
    return full, cal


def psi(a, b, edges):
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def save_new(split, name, x):
    dst, lnk = F / split / f"{name}.npy", CACHE / split / f"{name}.npy"
    assert not dst.exists() and not lnk.exists(), f"{dst} / {lnk} exists: refusing to overwrite"
    with open(dst, "xb") as fh:  # exclusive create
        np.save(fh, x)
    os.link(dst, lnk)
    st_d, st_l = os.stat(dst), os.stat(lnk)
    assert st_d.st_ino == st_l.st_ino and st_d.st_nlink == 2
    back = np.load(lnk)
    assert back.dtype == np.float32 and np.array_equal(back, x, equal_nan=True)
    print(f"  {split}: wrote {dst} ({x.shape[0]:,} rows), hard link {lnk} (inode {st_d.st_ino}, nlink 2)", flush=True)


def main():
    check = "--check" in sys.argv
    fallback = "--hmax32-fallback" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    key = ({k: k for k in RAW} if check else {"fe": "fe_noid", "new": "new_noid", "hmax32": "hmax32" if fallback else "hmax32_noid"})
    cols = {}
    cal = {}
    cols["val"], cal["val"] = build_valtest("val", VAL_TAGS, key, VAL_CALIB)
    cols["test"], cal["test"] = build_valtest("test", TEST_TAGS, key)
    cols["train"], cal["train"] = build_train(key)
    print(json.dumps({"key": key, "calib": cal}, indent=1, default=str), flush=True)
    res = {"key": key, "calib": cal, "check": check, "fallback": fallback}
    if check:  # the live construction must come back exactly
        for sp in ("val", "test", "train"):
            for kk in ("fe", "fe_z", "new", "new_z", "hmax32"):
                ref = np.load(F / sp / f"pre2_{kk}.npy")
                x = cols[sp][kk].astype(np.float32)
                same_nan = bool((np.isnan(ref) == np.isnan(x)).all())
                d = float(np.nanmax(np.abs(x - ref))) if same_nan else float("nan")
                res[f"check_{sp}_{kk}"] = dict(same_nan=same_nan, max_abs_diff=d, exact=bool(same_nan and np.array_equal(x, ref, equal_nan=True)))
                print(f"  CHECK {sp} pre2_{kk}: same NaN mask {same_nan}, max |diff| {d:.3e}, exact {res[f'check_{sp}_{kk}']['exact']}", flush=True)
        json.dump(res, open(OUT / "build_pre2noid_check.json", "w"), indent=1, default=str)
        return
    # noid columns: write + parity
    for sp in ("val", "test", "train"):
        c = cols[sp]
        fe_mask, ref_mask = np.isnan(c["fe"]), np.isnan(np.load(F / sp / "pre2_fe.npy"))
        assert (fe_mask == ref_mask).all(), f"{sp}: NaN mask of pre2n_fe differs from pre2_fe's"
        for kk in ("fe", "new"):
            assert (np.isnan(c[kk]) == fe_mask).all() and (np.isnan(c[kk + "_z"]) == fe_mask).all()
        assert (np.isnan(c["hmax32"]) == fe_mask).all()
        c["fe_s"] = ((c["fe"] + 20) / 6).astype(np.float32)
        c["new_s"] = ((c["new"] + 20) / 6).astype(np.float32)
        for kk in ("fe", "new", "fe_z", "new_z", "fe_s", "new_s", "hmax32"):
            save_new(sp, f"pre2n_{kk}", c[kk].astype(np.float32))
    # parity / information statistics
    stats = {}
    for sp in ("val", "test", "train"):
        c = cols[sp]
        m = ~np.isnan(c["fe"])
        live = {kk: np.load(F / sp / f"pre2_{kk}.npy") for kk in ("fe", "new", "hmax32")}
        stats[sp] = {kk: dict(mean=float(np.nanmean(c[kk])), sd=float(np.nanstd(c[kk])), corr_live=float(np.corrcoef(c[kk][m], live[kk][m])[0, 1]) if kk in live else None)
                     for kk in ("fe", "new", "hmax32", "fe_s", "new_s", "fe_z", "new_z")}
        stats[sp]["nan_share"] = float((~m).mean())
    for kk in ("fe_s", "new_s", "hmax32", "fe_z", "new_z"):
        tr = cols["train"][kk]; tr = tr[~np.isnan(tr)]
        q = np.unique(np.quantile(tr, np.linspace(0, 1, 11)[1:-1]))
        edges = np.r_[-1e9, q + 1e-6, 1e9]
        stats[f"psi_train_val_{kk}"] = psi(tr, cols["val"][kk], edges)
        stats[f"psi_train_test_{kk}"] = psi(tr, cols["test"][kk], edges)
    f = pl.read_parquet(FLAT / "val/feeds.parquet")
    y, off, M = np.load(FLAT / "val/cand_target.npy"), f["cand_off"].to_numpy(), f["M"].to_numpy()
    stats["zero_shot_val_ndcg"] = {"pre2n_fe": float(ndcg(cols["val"]["fe"], y, off, M)), "pre2_fe": float(ndcg(np.load(F / "val/pre2_fe.npy"), y, off, M)),
                                   "pre2n_new": float(ndcg(cols["val"]["new"], y, off, M)), "pre2_new": float(ndcg(np.load(F / "val/pre2_new.npy"), y, off, M)),
                                   "pre2n_hmax32": float(ndcg(cols["val"]["hmax32"], y, off, M)), "pre2_hmax32": float(ndcg(np.load(F / "val/pre2_hmax32.npy"), y, off, M))}
    res["stats"] = stats
    print(json.dumps(stats, indent=1), flush=True)
    json.dump(res, open(OUT / "pre2noid_parity.json", "w"), indent=1, default=str)
    for kk in ("fe_s", "new_s", "hmax32"):
        assert stats[f"psi_train_val_{kk}"] < 0.5, f"PSI train-vs-val of pre2n_{kk} >= 0.5: parity rule broken"


if __name__ == "__main__":
    main()
