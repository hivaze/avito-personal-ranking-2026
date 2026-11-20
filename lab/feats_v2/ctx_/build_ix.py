"""ctx_ix_: item cross-feed context from the train feeds the item appeared in (BlackPearl-style "feed-size stats").

Per item, over its train candidate appearances (val/test rows: all of train; train rows: strictly earlier train days):
  ctx_ix_mlogM    mean log(M) of the feeds containing the item
  ctx_ix_slogM    std of log(M)
  ctx_ix_dlogM    log(M of this feed) - mean log(M)  (is this feed bigger than the item's usual feeds)
  ctx_ix_mrpos    mean within-feed pct rank of the production display position (0 = shown first); a proxy of the
                  production ranker's opinion of the item, averaged over users
  ctx_ix_mlpos    mean log1p(pos)
  ctx_ix_mf1      mean f1 of the item in its train feeds; ctx_ix_df1 = f1 - mean f1 (this user's match vs usual)
  *_rk            within-feed pct rank of mlogM, mrpos, mf1
NaN when the item never appeared as a train candidate (before the row's day for train rows).
Usage: python build_ix.py <splits>
"""
import sys, time
from pathlib import Path
import numba as nb
import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/feats_v2/ctx_")
ROOT = Path("<repo>/data")
FLAT = ROOT / "flat"
FEATS = ROOT / "feats"
NI = 111_400_000


def day_of(ts):
    return (ts.astype(np.int64) + 3 * 3600) // 86400


@nb.njit(parallel=True, cache=True)
def _feed_pct_rank_asc(cand_off, M, x, out):
    """pct rank within feed, 0 = smallest, 1 = largest, average ties, NaN kept NaN; 0.5 for a single valid value."""
    for f in nb.prange(len(M)):
        co = cand_off[f]; m = M[f]
        v = x[co:co + m]
        nv = 0
        for i in range(m):
            if not np.isnan(v[i]):
                nv += 1
        order = np.argsort(v)  # NaN last
        i = 0
        while i < nv:
            j = i
            while j + 1 < nv and v[order[j + 1]] == v[order[i]]:
                j += 1
            r = ((i + j) / 2.0) / (nv - 1) if nv > 1 else 0.5
            for k in range(i, j + 1):
                out[co + order[k]] = r
            i = j + 1
        for k in range(nv, m):
            out[co + order[k]] = np.nan


def train_row_values():
    f = pl.read_parquet(FLAT / "train/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    co = f["cand_off"].to_numpy().astype(np.int64)
    fidx = np.load(FLAT / "train/cand_fidx.npy")
    logM = np.log(M.astype(np.float32))[fidx]
    pos = np.load(FLAT / "train/cand_pos.npy").astype(np.float32)
    rpos = np.empty(len(pos), np.float32)
    _feed_pct_rank_asc(co, M, pos, rpos)
    lpos = np.log1p(pos)
    f1 = np.load(FLAT / "train/cand_f1.npy").astype(np.float32)
    day = day_of(f["timestamp"].to_numpy())[fidx]
    return fidx, day, logM, rpos, lpos, f1


VALS = ["logM", "logM2", "rpos", "lpos", "f1"]


def main(splits):
    t0 = time.time()
    nb.set_num_threads(24)
    fidx, day, logM, rpos, lpos, f1 = train_row_values()
    item = np.load(FLAT / "train/cand_item_id.npy")
    vals = dict(logM=logM, logM2=logM * logM, rpos=rpos, lpos=lpos, f1=f1)
    print("train loaded", f"{time.time()-t0:.0f}s", flush=True)
    cnt = np.zeros(NI, np.float64)
    sums = {k: np.zeros(NI, np.float64) for k in VALS}
    out_tr = None
    if "train" in splits:
        out_tr = {k: np.full(len(item), np.nan, np.float32) for k in ["n"] + VALS}
    # rows are sorted by timestamp -> each day is a contiguous block
    bounds = np.flatnonzero(np.diff(day)) + 1
    starts = np.r_[0, bounds]; ends = np.r_[bounds, len(day)]
    for s, e in zip(starts, ends):
        it = item[s:e]
        if out_tr is not None:
            c = cnt[it]
            out_tr["n"][s:e] = c
            for k in VALS:
                out_tr[k][s:e] = np.where(c > 0, sums[k][it] / np.maximum(c, 1), np.nan)
        cnt += np.bincount(it, minlength=NI)
        for k in VALS:
            sums[k] += np.bincount(it, weights=vals[k][s:e], minlength=NI)
        print("day", day[s], e - s, f"{time.time()-t0:.0f}s", flush=True)
    del vals, logM, rpos, lpos, f1
    for split in splits:
        fe = pl.read_parquet(FLAT / split / "feeds.parquet")
        M = fe["M"].to_numpy().astype(np.int64)
        co = fe["cand_off"].to_numpy().astype(np.int64)
        fx = np.load(FLAT / split / "cand_fidx.npy")
        if split == "train":
            o = out_tr
        else:
            it = np.load(FLAT / split / "cand_item_id.npy")
            c = cnt[it]
            o = {"n": c.astype(np.float32)}
            for k in VALS:
                o[k] = np.where(c > 0, sums[k][it] / np.maximum(c, 1), np.nan).astype(np.float32)
        own_logM = np.log(M.astype(np.float32))[fx]
        own_f1 = np.load(FLAT / split / "cand_f1.npy").astype(np.float32)
        feats = {
            "ctx_ix_mlogM": o["logM"],
            "ctx_ix_slogM": np.sqrt(np.maximum(o["logM2"] - o["logM"] ** 2, 0)),
            "ctx_ix_dlogM": own_logM - o["logM"],
            "ctx_ix_mrpos": o["rpos"],
            "ctx_ix_mlpos": o["lpos"],
            "ctx_ix_mf1": o["f1"],
            "ctx_ix_df1": own_f1 - o["f1"],
        }
        for k in ["ctx_ix_mlogM", "ctx_ix_mrpos", "ctx_ix_mf1"]:
            r = np.empty(len(fx), np.float32)
            _feed_pct_rank_asc(co, M, feats[k].astype(np.float32), r)
            feats[k + "_rk"] = r
        for k, v in feats.items():
            np.save(FEATS / split / f"{k}.npy", np.asarray(v, dtype=np.float32))
        print(split, "saved", {k: float(np.isnan(v).mean()) for k, v in feats.items() if k.endswith("mlogM")},
              f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
