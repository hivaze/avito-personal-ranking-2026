"""Family "ud": user-vector drift relative to the same user's strictly earlier snapshot.

VAL/TEST-ONLY fine-tune extra. The written files are data/feats_new/{val,test}/ud_*.npy. Train files are NOT written:
on train a production user's previous feed is minutes old (val/test: ~10-15 days), so train-stage parity cannot hold.
The same build() code path is used, on train feed subsets, only by the parity diagnostics (scratch, not saved here).

Definition, for a feed g of non-zero user u at quantised timestamp t (feeds.parquet timestamps are multiples of 128 s;
rounding is monotone, so ts_P < t on the stored values implies a truly earlier request):
  P      = the same user's feed with the largest ts strictly below t, over train + val + test feeds.parquet.
           Ties among several feeds at that ts are broken by the largest (split code, fidx); deterministic.
  u_now  = user_emb[g] (own split), u_prev = user_emb[P] from data/flat/<split of P>/user_emb.npy.
  The feed is "covered" iff P exists, u != 0, and neither vector is all zero.
  c      = own-split item-table row (data/competition/<split>/item_embeddings) via cand_emb_idx; NaN if -1.
  cos(a, b) = a.b / ((|a| + 1e-6)(|b| + 1e-6))   (same convention as src/features.py cos_u)
Files (float32, aligned with data/flat/<split>/cand_*.npy rows; NaN on uncovered feeds or missing candidate vectors):
  ud_cos_prev = cos(c, u_prev)
  ud_drift    = cos(c, u_now) - ud_cos_prev
  ud_drift_fz = within-feed z of ud_drift over its non-NaN rows (population sd, +1e-9)
  ud_gap_l    = log1p((t - ts_P) / 86400), feed-level value broadcast to the feed's rows (NaN on uncovered feeds)
  ud_has      = 1 where ud_drift is non-NaN, else 0 (row-level; needed because attach_extra maps NaN -> 0 = val mean)
Leak rules honoured: only a strictly earlier snapshot of the same user (asserted), no labels anywhere, user 0 never
gets a P, identical construction for every split (P may come from any split; for val rows the strict-ts rule makes
test feeds impossible, asserted).
Usage: CUDA_VISIBLE_DEVICES="" python feat_new/ud.py [--splits val,test] [--threads 16]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, polars as pl, numba as nb

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
COMP = ROOT / "data/competition"
OUT = ROOT / "data/feats_new"
SPLITS = ("train", "val", "test")
NAMES = ["ud_cos_prev", "ud_drift", "ud_drift_fz", "ud_gap_l", "ud_has"]


def item_table(split):
    meta = json.loads((COMP / split / "item_embeddings/data.meta").read_text())
    return np.memmap(COMP / split / "item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


def prev_feeds():
    """For every feed of every split: (prev_split, prev_fidx, prev_ts), -1 if none. Returns dict split -> arrays."""
    parts = []
    for k, s in enumerate(SPLITS):
        f = pl.read_parquet(FLAT / s / "feeds.parquet", columns=["fidx", "timestamp", "user_id"])
        assert (f["fidx"].to_numpy() == np.arange(len(f))).all()
        parts.append((np.full(len(f), k, np.int64), f["fidx"].to_numpy().astype(np.int64),
                      f["timestamp"].to_numpy().astype(np.int64), f["user_id"].to_numpy().astype(np.int64)))
    sp, fx, ts, us = (np.concatenate([p[i] for p in parts]) for i in range(4))
    o = np.lexsort((fx, sp, ts, us))  # sort by user, then ts, then (split, fidx)
    u_s, t_s = us[o], ts[o]
    n = len(o)
    new_run = np.r_[True, (u_s[1:] != u_s[:-1]) | (t_s[1:] != t_s[:-1])]
    run_start = np.maximum.accumulate(np.where(new_run, np.arange(n), 0))
    pp = run_start - 1  # last element of the previous (user, ts) run = largest ts < t, max (split, fidx) on ties
    ok = (pp >= 0) & (u_s != 0)
    ok[ok] &= u_s[pp[ok]] == u_s[ok]
    assert (t_s[pp[ok]] < t_s[ok]).all(), "P must be strictly earlier"
    psplit = np.full(n, -1, np.int64); pfidx = np.full(n, -1, np.int64); pts = np.full(n, -1, np.int64)
    psplit[o[ok]] = sp[o[pp[ok]]]; pfidx[o[ok]] = fx[o[pp[ok]]]; pts[o[ok]] = ts[o[pp[ok]]]
    res, a = {}, 0
    for k, s in enumerate(SPLITS):
        m = len(parts[k][0])
        res[s] = dict(psplit=psplit[a:a + m], pfidx=pfidx[a:a + m], pts=pts[a:a + m], ts=ts[a:a + m], user=us[a:a + m])
        a += m
    assert (res["val"]["psplit"] != 2).all(), "val feeds can never see a test feed"
    return res


@nb.njit(parallel=True, cache=True)
def _cos_rows(C, ok_c, fid, cov, un, up, out_now, out_prev):
    n = C.shape[0]
    for r in nb.prange(n):
        f = fid[r]
        if not cov[f] or not ok_c[r]:
            out_now[r] = np.nan; out_prev[r] = np.nan
            continue
        d1 = 0.0; d2 = 0.0; cc = 0.0
        for k in range(C.shape[1]):
            v = np.float64(C[r, k])
            cc += v * v
            d1 += v * un[f, k]
            d2 += v * up[f, k]
        cn = np.sqrt(cc) + 1e-6
        out_now[r] = d1 / cn
        out_prev[r] = d2 / cn


@nb.njit(cache=True)
def _feed_z_nan(x, off, M, out):
    for f in range(len(M)):
        o = off[f]; m = M[f]
        mu = 0.0; c = 0
        for k in range(o, o + m):
            if not np.isnan(x[k]):
                mu += x[k]; c += 1
        if c == 0:
            for k in range(o, o + m):
                out[k] = np.nan
            continue
        mu /= c
        sd = 0.0
        for k in range(o, o + m):
            if not np.isnan(x[k]):
                sd += (x[k] - mu) ** 2
        sd = np.sqrt(sd / c)
        for k in range(o, o + m):
            out[k] = (x[k] - mu) / (sd + 1e-9) if not np.isnan(x[k]) else np.nan


def _unit(U):
    U = U.astype(np.float64)
    return U / (np.sqrt((U * U).sum(1, keepdims=True)) + 1e-6), (U != 0).any(1)


def build(split, fids=None, pf=None, chunk_rows=8_000_000):
    """Features for the candidate rows of `fids` (sorted feed indices of `split`; None = all feeds, contiguous rows).
    Returns (dict name -> float32 [rows], feed-level info dict, rows, offsets, M)."""
    pf = prev_feeds() if pf is None else pf
    f = pl.read_parquet(FLAT / split / "feeds.parquet", columns=["M", "cand_off"])
    Mall = f["M"].to_numpy().astype(np.int64); offall = f["cand_off"].to_numpy().astype(np.int64)
    fids = np.arange(len(f)) if fids is None else np.asarray(fids, np.int64)
    assert (np.diff(fids) > 0).all()
    M = Mall[fids]; off = np.r_[0, np.cumsum(M)[:-1]].astype(np.int64)
    rows = np.repeat(offall[fids] - off, M) + np.arange(M.sum())
    P = pf[split]
    psplit, pfidx = P["psplit"][fids], P["pfidx"][fids]
    UE = np.load(FLAT / split / "user_emb.npy", mmap_mode="r")
    un, nz_now = _unit(np.asarray(UE[fids]))
    Up = np.zeros((len(fids), UE.shape[1]), np.int8)
    for k, s in enumerate(SPLITS):
        sel = np.where(psplit == k)[0]
        if len(sel):
            src = pfidx[sel]; o = np.argsort(src, kind="stable")
            Up[sel[o]] = np.asarray(np.load(FLAT / s / "user_emb.npy", mmap_mode="r")[src[o]])
    up, nz_prev = _unit(Up)
    cov = (psplit >= 0) & nz_now & nz_prev
    gap_d = np.where(psplit >= 0, (P["ts"][fids] - P["pts"][fids]) / 86400.0, np.nan)
    n = len(rows)
    ei_all = np.load(FLAT / split / "cand_emb_idx.npy", mmap_mode="r")
    T = item_table(split)
    fid = np.repeat(np.arange(len(fids)), M)
    cnow = np.empty(n, np.float64); cprev = np.empty(n, np.float64)
    for a in range(0, n, chunk_rows):
        b = min(n, a + chunk_rows)
        ei = np.asarray(ei_all[rows[a:b]]).astype(np.int64)
        okc = ei >= 0
        need = np.where(okc & cov[fid[a:b]])[0]
        C = np.zeros((b - a, T.shape[1]), np.int8)
        if len(need):
            src = ei[need]; o = np.argsort(src, kind="stable")
            C[need[o]] = np.asarray(T[src[o]])
        okc = okc & (C != 0).any(1)  # an all-zero item vector has no direction
        _cos_rows(C, okc, fid[a:b], cov, un, up, cnow[a:b], cprev[a:b])
    drift = cnow - cprev
    fz = np.empty(n, np.float64)
    _feed_z_nan(drift, off, M, fz)
    gl = np.where(cov, np.log1p(np.maximum(gap_d, 0.0)), np.nan)[fid]
    out = dict(ud_cos_prev=cprev.astype(np.float32), ud_drift=drift.astype(np.float32), ud_drift_fz=fz.astype(np.float32),
               ud_gap_l=gl.astype(np.float32), ud_has=(~np.isnan(drift)).astype(np.float32))
    info = dict(fids=fids, cov=cov, gap_d=gap_d, psplit=psplit, cos_uu=(un * up).sum(1), cnow=cnow.astype(np.float32))
    return out, info, rows, off, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    nb.set_num_threads(min(a.threads, 16))
    t0 = time.time()
    pf = prev_feeds()
    print(f"prev-feed table {time.time() - t0:.0f}s", flush=True)
    for s in a.splits.split(","):
        assert s in ("val", "test"), "ud is a val/test-only family (see docstring)"
        out, info, rows, off, M = build(s, pf=pf)
        (OUT / s).mkdir(parents=True, exist_ok=True)
        for k in NAMES:
            np.save(OUT / s / f"{k}.npy", out[k])
        g = info["gap_d"][info["cov"]]
        print(f"{s}: rows {len(rows)} feeds {len(M)} covered {info['cov'].mean():.4f} rows-has {out['ud_has'].mean():.4f} "
              f"gap median {np.median(g):.2f} d; P split counts {np.bincount(info['psplit'][info['cov']], minlength=3)} "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
