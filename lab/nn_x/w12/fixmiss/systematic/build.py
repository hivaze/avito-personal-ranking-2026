"""Full-validation per-candidate table for the systematic-miss scan (CPU only).

Reuses lab/nn_x/errfix_1117/code/build_tables.feed_rows (66 columns, own-feed data only) with the CURRENT best
score: s = 0.5*ft + 0.5*to from preds/ens_FD16P20_m04_02kd_{ft,to}_val.npy. Adds label-free history columns
(explorer signals) and the user's earlier-labelled-feed novelty click rate (train feeds only: all train feeds
are earlier than every val feed; never a later snapshot).

Out: data/val_table.parquet (one row per val candidate, stored order).
"""
from __future__ import annotations

import os
for _k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_k, "1")
import sys
from pathlib import Path

import numpy as np
import polars as pl
import numba as nb

ROOT = Path("<repo>")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lab/nn_x/errfix_1117/code"))
import build_tables as bt  # noqa: E402

FLAT = ROOT / "artifacts/data/flat"


@nb.njit(cache=True)
def hist_feats(hoff, H, hc, K):
    n = len(H)
    ent = np.full(n, np.nan); ncat = np.zeros(n); nov = np.full(n, np.nan); novr = np.full(n, np.nan)
    brk10 = np.zeros(n); topsh = np.full(n, np.nan); nov_cnt = np.zeros(n)
    for f in range(n):
        o = hoff[f]; h = H[f]
        if h == 0:
            continue
        cnt = np.zeros(64)
        for i in range(h):
            cnt[hc[o + i]] += 1
        e = 0.0; nc = 0; mx = 0.0
        for c in range(64):
            if cnt[c] > 0:
                p = cnt[c] / h; e -= p * np.log(p); nc += 1
                if cnt[c] > mx:
                    mx = cnt[c]
        ent[f] = e; ncat[f] = nc; topsh[f] = mx / h
        seen = np.zeros(64)
        b = 0
        for i in range(min(10, h)):
            if seen[hc[o + i]] == 0:
                seen[hc[o + i]] = 1; b += 1
        brk10[f] = b
        # novelty: event i's category absent from the K older events i+1..i+K (needs full context)
        tot = 0; nv = 0; totr = 0; nvr = 0
        for i in range(0, h - K):
            c = hc[o + i]
            hit = False
            for j in range(i + 1, i + K + 1):
                if hc[o + j] == c:
                    hit = True
                    break
            tot += 1
            if not hit:
                nv += 1
            if i < 32:
                totr += 1
                if not hit:
                    nvr += 1
        if tot >= 8:
            nov[f] = nv / tot
        if totr >= 8:
            novr[f] = nvr / totr
        nov_cnt[f] = tot
    return ent, ncat, nov, novr, brk10, topsh, nov_cnt


@nb.njit(cache=True)
def past_novel(sel, coff, M, hoff, H, ccat, y, hc):
    """For each selected train feed: (#positives, #positives whose category is absent from that feed's history)."""
    npos = np.zeros(len(sel)); nnew = np.zeros(len(sel))
    for k in range(len(sel)):
        f = sel[k]
        seen = np.zeros(64)
        for i in range(H[f]):
            seen[hc[hoff[f] + i]] = 1
        for r in range(coff[f], coff[f] + M[f]):
            if y[r] == 1:
                npos[k] += 1
                if seen[ccat[r]] == 0:
                    nnew[k] += 1
    return npos, nnew


def main():
    F = pl.read_parquet(FLAT / "val/feeds.parquet")
    M = F["M"].to_numpy().astype(np.int64); off = F["cand_off"].to_numpy()
    ft = np.load(ROOT / "preds/ens_FD16P20_m04_02kd_ft_val.npy").astype(np.float64)
    to = np.load(ROOT / "preds/ens_FD16P20_m04_02kd_to_val.npy").astype(np.float64)
    sc = 0.5 * ft + 0.5 * to
    y = np.load(FLAT / "val/cand_target.npy")
    df = bt.build("val", np.arange(len(M)), {"score": sc, "s_ft": ft, "s_to": to}, off, M, y)
    assert df.height == len(sc)
    # label-free history explorer signals
    hc = np.load(FLAT / "val/hist_category_id.npy").astype(np.int64)
    H = F["H"].to_numpy().astype(np.int64); hoff = F["hist_off"].to_numpy().astype(np.int64)
    ent, ncat, nov, novr, brk10, topsh, novn = hist_feats(hoff, H, hc, 32)
    ffe = pl.DataFrame({"fidx": np.arange(len(M)), "timestamp": F["timestamp"].to_numpy(), "h_ent": ent,
                        "h_ncat": ncat, "h_nov32": nov, "h_nov32r": novr, "h_brk10": brk10, "h_topsh": topsh})
    # earlier labelled feeds of the same user (train only; user 0 never linked)
    T = pl.read_parquet(FLAT / "train/feeds.parquet")
    vu = np.unique(F["user_id"].to_numpy()); vu = vu[vu != 0]
    tu = T["user_id"].to_numpy()
    sel = np.nonzero(np.isin(tu, vu))[0].astype(np.int64)
    ty = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
    tcc = np.load(FLAT / "train/cand_category_id.npy", mmap_mode="r")
    thc = np.load(FLAT / "train/hist_category_id.npy", mmap_mode="r")
    tvmax = T["timestamp"].max(); assert tvmax < F["timestamp"].min(), "train must be earlier than val"
    npos, nnew = past_novel(sel, T["cand_off"].to_numpy().astype(np.int64), T["M"].to_numpy().astype(np.int64),
                            T["hist_off"].to_numpy().astype(np.int64), T["H"].to_numpy().astype(np.int64),
                            np.asarray(tcc), np.asarray(ty), np.asarray(thc))
    pu = pl.DataFrame({"user_id": tu[sel], "npos": npos, "nnew": nnew}).group_by("user_id").agg(
        pl.col("npos").sum().alias("past_npos"), pl.col("nnew").sum().alias("past_nnew"), pl.len().alias("past_nfeeds"))
    ffe = ffe.join(pl.DataFrame({"fidx": np.arange(len(M)), "user_id": F["user_id"].to_numpy()}), on="fidx")
    ffe = ffe.join(pu, on="user_id", how="left").with_columns(
        pl.when(pl.col("user_id") == 0).then(None).otherwise(pl.col("past_npos")).alias("past_npos")).drop("user_id")
    df = df.join(ffe, on="fidx", how="left")
    assert (df["row"].to_numpy() == np.arange(len(sc))).all()
    (HERE / "data").mkdir(exist_ok=True)
    df.write_parquet(HERE / "data/val_table.parquet")
    print("rows", df.height, "cols", len(df.columns))
    print("feeds with past train positives", int(ffe["past_npos"].fill_null(0).gt(0).sum()))


if __name__ == "__main__":
    main()
