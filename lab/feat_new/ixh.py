"""ixh_*: horizon-matched item exposure context from OTHER train production candidate lists
(feats_v2/ctx_/build_ix.py revived with a shared train-row lag L).

For a candidate row with cutoff day k = day(c), let A = all TRAIN candidate rows of the same item_id that belong to
train feeds whose MSK day is < k.  The own feed is never in A (its day >= k).
  cutoff c:
    train rows : c0 = start of the latest full train day <= own day (days with > 1000 feeds; the seq_cv_* /
                 feat_new/research_expo.py convention), then c = c0 - L days, with the SHARED lag of
                 feat_new/hz_common.py (L in {0, 4, 8} by a fixed hash of feed_id, one third each; the 30K
                 nn/train.py --holdout feeds get L = 0).
    val / test : c = 2026-09-20 00:00 MSK (all train days before Sep 20; the 34 train feeds stamped Sep 20 are excluded
                 for every split alike).
  n = |A|, rpos = within-feed pct rank of the production display position (build_ix._feed_pct_rank_asc, 0 = shown
  first, ties averaged, 0.5 for a 1-candidate feed).

Files (float32, aligned with data/flat/<split>/cand_*.npy rows) in data/feats_new/<split>/:
  ixh_good   1 - mean_A(rpos)            production ranker's average opinion of the item (NaN if n = 0)
  ixh_lpos   mean_A log1p(pos)           (NaN if n = 0)
  ixh_mlogM  mean_A log M                (NaN if n = 0)
  ixh_dlogM  log M_own - ixh_mlogM       (NaN if n = 0)
  ixh_mf1    mean_A f1                   (NaN if n = 0)
  ixh_df1    f1_own - ixh_mf1            (NaN if n = 0)
  ixh_ln     log1p(n) / 2                (0 when unseen, never NaN)
  ixh_has    1[n > 0]                    explicit availability flag for nn/finetune.py attach_extra (no has: support)
Side output (scratch): per-train-feed lag and cutoff day, for the gates.

Leak argument: only train candidate rows of days strictly before the row's own cutoff day are read; no target, nothing
from the own feed, no later snapshot of anybody (train rows see only earlier train days, val/test rows see only train).
pos is the production display position (decided before any interaction).  User-0 feeds are ordinary feeds here
(item-level statistic over feeds, no user-level statistic is formed).  Identical code path for every split: the
cumulative state "all train rows of days < k" is queried at k = row cutoff.

Usage: CUDA_VISIBLE_DEVICES="" python feat_new/ixh.py [--verify N]
"""
import argparse, sys, time
from pathlib import Path
import numba as nb
import numpy as np
import polars as pl

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
OUT = ROOT / "data/feats_new"
SCR = Path("/tmp/scratchpad/featrev/ixh")
MSK = 3 * 3600
DAY_VT = 20716          # 2026-09-20 (MSK day index): val/test cutoff day
NI = 111_400_000        # > max item_id (111,319,563)
SHARED_LAGS = np.array([0, 4, 8], np.int64)


def day_of(ts):
    return (np.asarray(ts).astype(np.int64) + MSK) // 86400


def holdout_feeds(M):
    """Exactly nn/train.py:149-157 (--holdout 30000)."""
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(len(M))).all()
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    return np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))


def train_cutoffs(f):
    """Per train feed: (base cutoff day c0, lag L, cutoff day k = c0 - L).
    The lag is the SHARED one of feat_new/hz_common.py (lag_per_feed / cutoff_per_feed, also used by hzcv and
    it_hist_hz); the asserts re-derive c0 and the holdout override independently."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hz_common as hz
    ts = f["timestamp"].to_numpy()
    M = f["M"].to_numpy().astype(np.int64)
    d = day_of(ts)
    u, c = np.unique(d, return_counts=True)
    days = u[c > 1000]  # feats_v2/seq_/sources.py::train_days rule (as in research_expo.cutoff_per_feed)
    idx = np.searchsorted(days, d, side="right") - 1
    assert idx.min() >= 0
    c0 = days[idx]
    ts_h, cut_h, L = hz.cutoff_per_feed("train", lag=True)
    assert (ts_h == ts.astype(np.int64)).all() and set(np.unique(L).tolist()) <= set(SHARED_LAGS.tolist())
    assert (L[holdout_feeds(M)] == 0).all()
    k = (cut_h + MSK) // 86400
    assert ((cut_h + MSK) % 86400 == 0).all() and (k == c0 - L).all()
    return c0, L.astype(np.int64), k.astype(np.int64)


@nb.njit(parallel=True, cache=True)
def _feed_pct_rank_asc(cand_off, M, x, out):
    """copy of feats_v2/ctx_/build_ix.py::_feed_pct_rank_asc: pct rank within feed, 0 = smallest, ties averaged,
    NaN kept NaN; 0.5 for a single valid value."""
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


@nb.njit(cache=True)
def _add_rows(item, v0, v1, v2, v3, s, e, cnt, a0, a1, a2, a3):
    for r in range(s, e):
        i = item[r]
        cnt[i] += 1.0
        a0[i] += v0[r]; a1[i] += v1[r]; a2[i] += v2[r]; a3[i] += v3[r]


@nb.njit(parallel=True, cache=True)
def _answer(feeds, cand_off, M, item, cnt, a0, a1, a2, a3, on, o0, o1, o2, o3):
    for j in nb.prange(len(feeds)):
        f = feeds[j]
        for r in range(cand_off[f], cand_off[f] + M[f]):
            i = item[r]
            c = cnt[i]
            on[r] = c
            if c > 0:
                o0[r] = a0[i] / c; o1[r] = a1[i] / c; o2[r] = a2[i] / c; o3[r] = a3[i] / c
            else:
                o0[r] = np.nan; o1[r] = np.nan; o2[r] = np.nan; o3[r] = np.nan


def empty_out(n):
    return [np.zeros(n, np.float32)] + [np.full(n, np.nan, np.float32) for _ in range(4)]


def finalize(split, raw, M_feed, co, fidx):
    on, m_rpos, m_lpos, m_logM, m_f1 = raw
    own_logM = np.log(M_feed.astype(np.float64))[fidx]
    own_f1 = np.load(FLAT / split / "cand_f1.npy").astype(np.float64)
    feats = {
        "ixh_good": 1.0 - m_rpos.astype(np.float64),
        "ixh_lpos": m_lpos,
        "ixh_mlogM": m_logM,
        "ixh_dlogM": own_logM - m_logM,
        "ixh_mf1": m_f1,
        "ixh_df1": own_f1 - m_f1,
        "ixh_ln": np.log1p(on.astype(np.float64)) / 2.0,
        "ixh_has": (on > 0).astype(np.float32),
    }
    (OUT / split).mkdir(parents=True, exist_ok=True)
    for k, v in feats.items():
        v = np.asarray(v, np.float32)
        assert len(v) == len(fidx)
        np.save(OUT / split / f"{k}.npy", v)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    nb.set_num_threads(min(a.threads, 16))
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    f = pl.read_parquet(FLAT / "train/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    co = f["cand_off"].to_numpy().astype(np.int64)
    ts = f["timestamp"].to_numpy().astype(np.int64)
    assert (np.diff(ts) >= 0).all() and (co == np.concatenate([[0], np.cumsum(M)[:-1]])).all()
    fday = day_of(ts)
    c0, L, kf = train_cutoffs(f)
    np.savez(SCR / "train_lag.npz", c0=c0, L=L, k=kf, fday=fday)
    print("lags:", {int(l): int((L == l).sum()) for l in SHARED_LAGS}, "holdout L=0 forced", flush=True)

    fidx = np.load(FLAT / "train/cand_fidx.npy")
    item = np.load(FLAT / "train/cand_item_id.npy")
    assert item.max() < NI
    pos = np.load(FLAT / "train/cand_pos.npy").astype(np.float32)
    rpos = np.empty(len(pos), np.float32)
    _feed_pct_rank_asc(co, M, pos, rpos)
    assert not np.isnan(rpos).any()
    v_rpos = rpos
    v_lpos = np.log1p(pos).astype(np.float32); del pos
    v_logM = np.log(M.astype(np.float32))[fidx]
    v_f1 = np.load(FLAT / "train/cand_f1.npy").astype(np.float32)
    print("train row values", f"{time.time()-t0:.0f}s", flush=True)

    cnt = np.zeros(NI, np.float64)
    acc = [np.zeros(NI, np.float64) for _ in range(4)]
    raw_tr = empty_out(len(item))
    feed_bounds = {}  # actual MSK day -> contiguous feed range
    for d in np.unique(fday):
        w = np.flatnonzero(fday == d)
        feed_bounds[int(d)] = (int(w[0]), int(w[-1]) + 1)
        assert w[-1] - w[0] + 1 == len(w)
    k_lo = int(min(kf.min(), fday.min()))
    vt = {}
    for k in range(k_lo, DAY_VT + 1):
        q = np.flatnonzero(kf == k).astype(np.int64)
        if len(q):
            _answer(q, co, M, item, cnt, *acc, *raw_tr)
        if k == DAY_VT:
            for split in ["val", "test"]:
                fs = pl.read_parquet(FLAT / split / "feeds.parquet")
                Ms = fs["M"].to_numpy().astype(np.int64); cos_ = fs["cand_off"].to_numpy().astype(np.int64)
                its = np.load(FLAT / split / "cand_item_id.npy")
                raw = empty_out(len(its))
                _answer(np.arange(len(Ms), dtype=np.int64), cos_, Ms, its, cnt, *acc, *raw)
                vt[split] = (raw, Ms, cos_)
        if k in feed_bounds:
            fs_, fe_ = feed_bounds[k]
            s, e = int(co[fs_]), int(co[fe_ - 1] + M[fe_ - 1])
            _add_rows(item, v_rpos, v_lpos, v_logM, v_f1, s, e, cnt, *acc)
        print(f"k={k} answered {len(q)} train feeds; added day rows {feed_bounds.get(k, (0, 0))}; "
              f"items seen {int((cnt > 0).sum()) if k % 4 == 0 else -1}  {time.time()-t0:.0f}s", flush=True)
    del cnt, acc, v_rpos, v_lpos, v_logM, v_f1
    for split, (raw, Ms, cos_) in vt.items():
        fx = np.load(FLAT / split / "cand_fidx.npy")
        ft = finalize(split, raw, Ms, cos_, fx)
        print(split, "saved; NaN share ixh_good", float(np.isnan(ft["ixh_good"]).mean()), f"{time.time()-t0:.0f}s", flush=True)
    ft = finalize("train", raw_tr, M, co, fidx)
    print("train saved; NaN share ixh_good", float(np.isnan(ft["ixh_good"]).mean()), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
