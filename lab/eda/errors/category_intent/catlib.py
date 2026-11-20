"""Shared helpers for the CATEGORY-INTENT error lens.

Groups = (feed, category) cells. Groups of a feed are contiguous and ordered by category id.
History is stored newest-first (hpos 0 = most recent event).
"""
import sys
from pathlib import Path
import numpy as np
import numba as nb
import polars as pl

ROOT = Path("<repo>")
FLAT = ROOT / "data" / "flat"
FEATS = ROOT / "data" / "feats"
PREDS = ROOT / "preds"
OUT = ROOT / "eda" / "errors" / "category_intent" / "out"
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed  # noqa: E402

N_CAT = 64
N_ET = 10
N_SRC = 27


# ----------------------------------------------------------------------------- data views
def leak_mask_val():
    """True for the val feeds whose non-zero user has a LATER feed in val or test (to be excluded)."""
    fv = pl.read_parquet(FLAT / "val" / "feeds.parquet").select("fidx", "user_id", "timestamp")
    ft = pl.read_parquet(FLAT / "test" / "feeds.parquet").select("user_id", "timestamp")
    allf = pl.concat([fv.select("user_id", "timestamp"), ft]).rename({"timestamp": "ts2"})
    j = fv.filter(pl.col("user_id") != 0).join(allf, on="user_id").filter(pl.col("ts2") > pl.col("timestamp"))
    bad = np.unique(j["fidx"].to_numpy())
    m = np.zeros(len(fv), bool)
    m[bad] = True
    return m


def load_view(split, feeds=None, cand_feats=(), with_hist=True):
    """Compact view of a split restricted to `feeds` (sorted feed indices; None = all)."""
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    M_all = f["M"].to_numpy().astype(np.int64)
    H_all = f["H"].to_numpy().astype(np.int64)
    off_all = f["cand_off"].to_numpy().astype(np.int64)
    hoff_all = f["hist_off"].to_numpy().astype(np.int64)
    if feeds is None:
        feeds = np.arange(len(f))
        cidx = None
        hidx = None
    else:
        feeds = np.sort(np.asarray(feeds))
        cidx = _ranges(off_all[feeds], M_all[feeds])
        hidx = _ranges(hoff_all[feeds], H_all[feeds])
    M = M_all[feeds]
    H = H_all[feeds]
    v = dict(split=split, feeds=feeds, M=M, H=H,
             off=np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64),
             hoff=np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64),
             user_id=f["user_id"].to_numpy()[feeds], timestamp=f["timestamp"].to_numpy()[feeds])

    def take(path, idx):
        x = np.load(path, mmap_mode="r")
        return np.asarray(x) if idx is None else np.asarray(x[idx])

    for n in ["category_id", "f1", "f2", "f3", "f4", "f5"]:
        v["c_" + n] = take(FLAT / split / f"cand_{n}.npy", cidx)
    if (FLAT / split / "cand_target.npy").exists():
        v["c_target"] = take(FLAT / split / "cand_target.npy", cidx).astype(np.float32)
    for n in cand_feats:
        v["x_" + n] = take(FEATS / split / f"{n}.npy", cidx).astype(np.float32)
    if with_hist:
        for n in ["category_id", "event_type", "event_source"]:
            v["h_" + n] = take(FLAT / split / f"hist_{n}.npy", hidx)
    ue = np.load(FLAT / split / "user_emb.npy", mmap_mode="r")
    v["user_emb"] = np.asarray(ue[feeds]).astype(np.float32)
    v["cidx"] = cidx
    return v


def _ranges(starts, lens):
    tot = int(lens.sum())
    out = np.empty(tot, np.int64)
    _fill_ranges(starts, lens, np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64), out)
    return out


@nb.njit(parallel=True, cache=True)
def _fill_ranges(starts, lens, o, out):
    for i in nb.prange(len(starts)):
        for k in range(lens[i]):
            out[o[i] + k] = starts[i] + k


# ----------------------------------------------------------------------------- groups
@nb.njit(parallel=True, cache=True)
def _count_groups(off, M, cat):
    n = len(M)
    G = np.zeros(n, np.int64)
    for f in nb.prange(n):
        seen = np.zeros(N_CAT, np.bool_)
        for i in range(M[f]):
            seen[cat[off[f] + i]] = True
        G[f] = seen.sum()
    return G


@nb.njit(parallel=True, cache=True)
def _fill_groups(off, M, cat, goff, cand_g, g_cat, g_n, g_feed):
    n = len(M)
    for f in nb.prange(n):
        cnt = np.zeros(N_CAT, np.int64)
        for i in range(M[f]):
            cnt[cat[off[f] + i]] += 1
        loc = np.full(N_CAT, -1, np.int64)
        k = 0
        for c in range(N_CAT):
            if cnt[c] > 0:
                loc[c] = goff[f] + k
                g_cat[goff[f] + k] = c
                g_n[goff[f] + k] = cnt[c]
                g_feed[goff[f] + k] = f
                k += 1
        for i in range(M[f]):
            cand_g[off[f] + i] = loc[cat[off[f] + i]]


def build_groups(v):
    off, M, cat = v["off"], v["M"], v["c_category_id"].astype(np.int64)
    G = _count_groups(off, M, cat)
    goff = np.concatenate([[0], np.cumsum(G)[:-1]]).astype(np.int64)
    ng = int(G.sum())
    cand_g = np.empty(len(cat), np.int64)
    g_cat = np.empty(ng, np.int64); g_n = np.empty(ng, np.int64); g_feed = np.empty(ng, np.int64)
    _fill_groups(off, M, cat, goff, cand_g, g_cat, g_n, g_feed)
    v.update(G=G, goff=goff, cand_g=cand_g, g_cat=g_cat, g_n=g_n, g_feed=g_feed)
    if "c_target" in v:
        v["g_npos"] = np.bincount(cand_g, weights=v["c_target"], minlength=ng).astype(np.float32)
        v["g_y"] = (v["g_npos"] > 0).astype(np.float32)
    return v


def gmax(v, x):
    out = np.full(len(v["g_cat"]), -np.inf)
    np.maximum.at(out, v["cand_g"], x.astype(np.float64))
    return out


def gmean(v, x):
    return np.bincount(v["cand_g"], weights=x.astype(np.float64), minlength=len(v["g_cat"])) / v["g_n"]


def gmin(v, x):
    return -gmax(v, -x)


@nb.njit(parallel=True, cache=True)
def _feed_lse(off, M, x, out):
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        mx = -1e300
        for i in range(m):
            if x[o + i] > mx:
                mx = x[o + i]
        s = 0.0
        for i in range(m):
            s += np.exp(x[o + i] - mx)
        out[f] = mx + np.log(s)


def feed_lse(off, M, x):
    out = np.empty(len(M))
    _feed_lse(off, M, x.astype(np.float64), out)
    return out


def glse(v, x):
    """log-sum-exp of x within each group."""
    mx = gmax(v, x)
    s = np.bincount(v["cand_g"], weights=np.exp(x - mx[v["cand_g"]]), minlength=len(mx))
    return mx + np.log(s)


@nb.njit(parallel=True, cache=True)
def feed_z(off, M, x):
    out = np.empty(len(x), np.float64)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        mu = 0.0
        for i in range(m):
            mu += x[o + i]
        mu /= m
        sd = 0.0
        for i in range(m):
            sd += (x[o + i] - mu) ** 2
        sd = np.sqrt(sd / m) + 1e-9
        for i in range(m):
            out[o + i] = (x[o + i] - mu) / sd
    return out


@nb.njit(parallel=True, cache=True)
def feed_pctrank(off, M, x):
    """within-feed percentile rank, average ties, 0 = lowest, 1 = highest"""
    out = np.empty(len(x), np.float64)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        if m == 1:
            out[o] = 0.5
            continue
        v = x[o:o + m]
        idx = np.argsort(v)
        i = 0
        while i < m:
            j = i
            while j + 1 < m and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            r = 0.5 * (i + j) / (m - 1)
            for k in range(i, j + 1):
                out[o + idx[k]] = r
            i = j + 1
    return out


# ----------------------------------------------------------------------------- history per group
HIST_NAMES = (["h_cnt", "h_cnt5", "h_cnt20", "h_minh", "h_exp3", "h_exp10", "h_exp30", "h_recrank"]
              + [f"h_et{e}" for e in range(1, N_ET + 1)]
              + [f"h_et{e}_exp10" for e in (1, 2, 3, 5)] + ["h_src_top_" + str(s) for s in range(6)])
TOP_SRC = np.array([0, 1, 2, 3, 4, 5], np.int64)  # replaced at runtime by set_top_src


@nb.njit(parallel=True, cache=True)
def _hist_group_feats(hoff, H, hcat, het, hsrc, goff, G, g_cat, top_src, out):
    n = len(H)
    for f in nb.prange(n):
        cnt = np.zeros(N_CAT); c5 = np.zeros(N_CAT); c20 = np.zeros(N_CAT)
        minh = np.full(N_CAT, 999.0)
        e3 = np.zeros(N_CAT); e10 = np.zeros(N_CAT); e30 = np.zeros(N_CAT)
        cet = np.zeros((N_CAT, N_ET + 1))
        cete = np.zeros((N_CAT, N_ET + 1))
        csrc = np.zeros((N_CAT, 6))
        ho = hoff[f]
        for j in range(H[f]):
            c = hcat[ho + j]
            if c < 0 or c >= N_CAT:
                continue
            cnt[c] += 1
            if j < 5:
                c5[c] += 1
            if j < 20:
                c20[c] += 1
            if minh[c] > j:
                minh[c] = j
            w10 = np.exp(-j / 10.0)
            e3[c] += np.exp(-j / 3.0); e10[c] += w10; e30[c] += np.exp(-j / 30.0)
            e = het[ho + j]
            if 0 < e <= N_ET:
                cet[c, e] += 1
                cete[c, e] += w10
            s = hsrc[ho + j]
            for k in range(6):
                if top_src[k] == s:
                    csrc[c, k] += 1
        # recency rank of each category among the history categories (0 = most recent)
        rr = np.full(N_CAT, 99.0)
        for c in range(N_CAT):
            if minh[c] < 999.0:
                r = 0
                for c2 in range(N_CAT):
                    if minh[c2] < minh[c]:
                        r += 1
                rr[c] = r
        for k in range(G[f]):
            g = goff[f] + k
            c = g_cat[g]
            out[g, 0] = cnt[c]; out[g, 1] = c5[c]; out[g, 2] = c20[c]; out[g, 3] = minh[c]
            out[g, 4] = e3[c]; out[g, 5] = e10[c]; out[g, 6] = e30[c]; out[g, 7] = rr[c]
            for e in range(1, N_ET + 1):
                out[g, 7 + e] = cet[c, e]
            out[g, 18] = cete[c, 1]; out[g, 19] = cete[c, 2]; out[g, 20] = cete[c, 3]; out[g, 21] = cete[c, 5]
            for k2 in range(6):
                out[g, 22 + k2] = csrc[c, k2]


def hist_group_feats(v, top_src):
    out = np.zeros((len(v["g_cat"]), len(HIST_NAMES)), np.float32)
    _hist_group_feats(v["hoff"], v["H"], v["h_category_id"].astype(np.int64), v["h_event_type"].astype(np.int64),
                      v["h_event_source"].astype(np.int64), v["goff"], v["G"], v["g_cat"], top_src, out)
    return out


# ----------------------------------------------------------------------------- metrics
def cat_ndcg(v, gscore):
    """per-feed category-level NDCG (binary relevance = category holds >= 1 positive), pessimistic ties."""
    return ndcg_per_feed(gscore, v["g_y"], v["goff"], v["G"])


def item_ndcg(v, score):
    return ndcg_per_feed(score, v["c_target"], v["off"], v["M"])


def paired(a, b, mask):
    d = (a - b)[mask]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def mse(x, mask):
    x = x[mask]
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
