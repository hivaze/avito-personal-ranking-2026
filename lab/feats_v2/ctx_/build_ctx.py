"""ctx_ family: feed context and competition among candidates (FeatureSalad / BlackPearl-style contextual normalization).

Usage: python build_ctx.py <splits comma-separated> <groups comma-separated> [threads]
groups:
  fs  feed-level stats (constant within a feed): mean/std/skew of f1..f5, maxsim32, cos_u, cos_h10; category entropy,
      #distinct categories, max category share; feed embedding diversity; share of feed in history categories
  ct  candidate contrasts: x - median, x/max (f) or min-max (sims), x/sum (f)
  cr  within-category (same feed, same category_id): pct rank, z-score, gap to category max, for the 8 columns
  cl  category-level: rank of the candidate's category among the feed's categories by its best maxsim32 / cos_u / f1,
      category rank by count, history share / feed share ratio, category mean maxsim32 minus feed mean
  nd  similarity graph among candidates (item embeddings): max cos to other candidates, #near-duplicates (cos > .8/.9/.95),
      mean cos to top-5 candidates by cos_u / by maxsim32, kNN-smoothed cos_u / maxsim32 / f1 of the neighbours,
      #near-duplicates (cos > .9) with a better maxsim32 / f1 / cos_u, user / history direction after removing the feed mean
  ix  item cross-feed context: stats of feed sizes M / normalized production position of the item's train feeds
      (expanding window by day for train rows), exploration (val+test) appearances within +-48h; ranked within feed
All features: float32, flat, aligned with data/flat/<split>/cand_*.npy, written to data/feats/<split>/ctx_*.npy
"""
import json, sys, time
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl

ROOT = Path("<repo>/data")
FLAT = ROOT / "flat"
FEATS = ROOT / "feats"
N_CAT = 54
COLS = ["f1", "f2", "f3", "f4", "f5", "maxsim32", "cos_u", "cos_h10"]
NF = 5  # first NF columns are the f-features (non-negative)


def load_item_emb(split):
    d = ROOT / "competition" / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.memmap(d / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


# ----------------------------------------------------------------------------- helpers (numba)
@nb.njit(cache=True)
def _pct_rank_desc(v, idx, out_rank):
    """pct rank of v[idx] within the group (1 = highest, 0 = lowest, average ties, NaN -> NaN); 0.5 if group size 1."""
    m = len(idx)
    vals = np.empty(m, np.float64)
    nv = 0
    for k in range(m):
        x = v[idx[k]]
        vals[k] = x
        if not np.isnan(x):
            nv += 1
    for k in range(m):
        x = vals[k]
        if np.isnan(x):
            out_rank[k] = np.nan
            continue
        if nv <= 1:
            out_rank[k] = 0.5
            continue
        lo = 0; eq = 0
        for j in range(m):
            y = vals[j]
            if np.isnan(y):
                continue
            if y < x:
                lo += 1
            elif y == x:
                eq += 1
        out_rank[k] = (lo + (eq - 1) / 2.0) / (nv - 1)


@nb.njit(cache=True)
def _median(vals):
    s = np.sort(vals[~np.isnan(vals)])
    n = len(s)
    if n == 0:
        return np.nan
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


# ----------------------------------------------------------------------------- fs + ct + cr + cl
@nb.njit(parallel=True, cache=True)
def _ctx_kernel(cand_off, M, X, cat, hist_cat_share, feed_out, cand_out):
    """X: N x K (K=8). cat: N int64. hist_cat_share: N float (share of user's history in the candidate's category).
    feed_out (n_feeds x (3K + 4)): [mean_k, std_k, skew_k]*K, cat_entropy, n_cat, max_cat_share, share_in_hist_cats
    cand_out (N x C) columns per k: 0 med, 1 rmax/minmax, 2 rsum, 3 cat_rk, 4 cat_z, 5 cat_gap  (6 per k) then
       6K+0 cl_best_h_rk, +1 cl_best_u_rk, +2 cl_best_f1_rk, +3 cl_cnt_rk, +4 cl_hist_ratio, +5 cl_mean_h_dev, +6 cat_n"""
    n = len(M)
    K = X.shape[1]
    for f in nb.prange(n):
        co = cand_off[f]
        m = M[f]
        # ---------------- feed stats
        for k in range(K):
            s1 = 0.0; cnt = 0
            for i in range(m):
                x = X[co + i, k]
                if not np.isnan(x):
                    s1 += x; cnt += 1
            mu = s1 / cnt if cnt > 0 else np.nan
            s2 = 0.0; s3 = 0.0
            for i in range(m):
                x = X[co + i, k]
                if not np.isnan(x):
                    d = x - mu
                    s2 += d * d; s3 += d * d * d
            sd = np.sqrt(s2 / cnt) if cnt > 0 else np.nan
            feed_out[f, 3 * k] = mu
            feed_out[f, 3 * k + 1] = sd
            feed_out[f, 3 * k + 2] = (s3 / cnt) / (sd ** 3) if (cnt > 2 and sd > 1e-9) else 0.0
        cc = np.zeros(N_CAT, np.int64)
        inh = 0
        for i in range(m):
            cc[cat[co + i]] += 1
            if hist_cat_share[co + i] > 0:
                inh += 1
        ent = 0.0; ncat = 0; mx = 0
        for c in range(N_CAT):
            if cc[c] > 0:
                p = cc[c] / m
                ent -= p * np.log(p)
                ncat += 1
                if cc[c] > mx:
                    mx = cc[c]
        feed_out[f, 3 * K] = ent
        feed_out[f, 3 * K + 1] = ncat
        feed_out[f, 3 * K + 2] = mx / m
        feed_out[f, 3 * K + 3] = inh / m
        # ---------------- candidate contrasts
        for k in range(K):
            v = X[co:co + m, k].astype(np.float64)
            med = _median(v)
            vmax = -1e30; vmin = 1e30; vs = 0.0
            for i in range(m):
                x = v[i]
                if not np.isnan(x):
                    if x > vmax:
                        vmax = x
                    if x < vmin:
                        vmin = x
                    vs += x
            for i in range(m):
                x = v[i]
                cand_out[co + i, 6 * k] = x - med
                if k < NF:
                    cand_out[co + i, 6 * k + 1] = x / vmax if vmax > 0 else np.nan
                    cand_out[co + i, 6 * k + 2] = x / vs if vs > 0 else np.nan
                else:
                    cand_out[co + i, 6 * k + 1] = (x - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    cand_out[co + i, 6 * k + 2] = np.nan
        # ---------------- within category
        # group members by category (stable order)
        order = np.argsort(cat[co:co + m], kind="mergesort")
        best_h = np.full(N_CAT, -1e30); best_u = np.full(N_CAT, -1e30); best_f1 = np.full(N_CAT, -1e30)
        sum_h = np.zeros(N_CAT); cnt_h = np.zeros(N_CAT)
        fh_s = 0.0; fh_c = 0
        a = 0
        while a < m:
            c = cat[co + order[a]]
            b = a
            while b + 1 < m and cat[co + order[b + 1]] == c:
                b += 1
            g = order[a:b + 1] + co
            gm = b - a + 1
            rk = np.empty(gm, np.float64)
            for k in range(K):
                _pct_rank_desc(X[:, k], g, rk)
                s1 = 0.0; cnt = 0; gmax = -1e30
                for t in range(gm):
                    x = X[g[t], k]
                    if not np.isnan(x):
                        s1 += x; cnt += 1
                        if x > gmax:
                            gmax = x
                mu = s1 / cnt if cnt > 0 else 0.0
                s2 = 0.0
                for t in range(gm):
                    x = X[g[t], k]
                    if not np.isnan(x):
                        s2 += (x - mu) ** 2
                sd = np.sqrt(s2 / cnt) if cnt > 0 else 0.0
                for t in range(gm):
                    x = X[g[t], k]
                    cand_out[g[t], 6 * k + 3] = rk[t]
                    cand_out[g[t], 6 * k + 4] = np.nan if np.isnan(x) else ((x - mu) / sd if sd > 1e-9 else 0.0)
                    cand_out[g[t], 6 * k + 5] = x - gmax if cnt > 0 else np.nan
                if k == 0:
                    best_f1[c] = gmax
                elif k == 5:
                    best_h[c] = gmax
                    sum_h[c] = s1; cnt_h[c] = cnt
                    fh_s += s1; fh_c += cnt
                elif k == 6:
                    best_u[c] = gmax
            a = b + 1
        fmean_h = fh_s / fh_c if fh_c > 0 else np.nan
        # ---------------- category-level ranks (0 = best category), normalized by (ncat - 1)
        for i in range(m):
            c = cat[co + i]
            r_h = 0; r_u = 0; r_f = 0; r_c = 0
            for c2 in range(N_CAT):
                if cc[c2] == 0 or c2 == c:
                    continue
                if best_h[c2] > best_h[c]:
                    r_h += 1
                if best_u[c2] > best_u[c]:
                    r_u += 1
                if best_f1[c2] > best_f1[c]:
                    r_f += 1
                if cc[c2] > cc[c]:
                    r_c += 1
            den = ncat - 1 if ncat > 1 else 1
            cand_out[co + i, 6 * K] = r_h / den if ncat > 1 else 0.0
            cand_out[co + i, 6 * K + 1] = r_u / den if ncat > 1 else 0.0
            cand_out[co + i, 6 * K + 2] = r_f / den if ncat > 1 else 0.0
            cand_out[co + i, 6 * K + 3] = r_c / den if ncat > 1 else 0.0
            cand_out[co + i, 6 * K + 4] = hist_cat_share[co + i] / (cc[c] / m)
            cand_out[co + i, 6 * K + 5] = (sum_h[c] / cnt_h[c] - fmean_h) if cnt_h[c] > 0 else np.nan
            cand_out[co + i, 6 * K + 6] = cc[c]


def ctx_names():
    K = len(COLS)
    cand = []
    for k, c in enumerate(COLS):
        cand += [f"ctx_ct_{c}_med", f"ctx_ct_{c}_rmax" if k < NF else f"ctx_ct_{c}_mm", f"ctx_ct_{c}_rsum",
                 f"ctx_cr_{c}_rk", f"ctx_cr_{c}_z", f"ctx_cr_{c}_gap"]
    cand += ["ctx_cl_best_h_rk", "ctx_cl_best_u_rk", "ctx_cl_best_f1_rk", "ctx_cl_cnt_rk", "ctx_cl_hist_ratio",
             "ctx_cl_mean_h_dev", "ctx_cr_n"]
    feed = []
    for c in COLS:
        feed += [f"ctx_fs_{c}_mean", f"ctx_fs_{c}_std", f"ctx_fs_{c}_skew"]
    feed += ["ctx_fs_cat_ent", "ctx_fs_n_cat", "ctx_fs_max_cat_share", "ctx_fs_in_hist_share"]
    return feed, cand


# ----------------------------------------------------------------------------- nd: similarity graph
@nb.njit(parallel=True, fastmath=True, cache=True)
def _nd_kernel(cand_off, M, hist_off, H, cemb, hemb, table, user_emb, cos_u, maxsim, f1, feed_out, out):
    """out columns:
    0 nd_max, 1 nd_c80, 2 nd_c90, 3 nd_c95, 4 nd_mean, 5 nd_top5u, 6 nd_top5h,
    7 nd_knn_u, 8 nd_knn_h, 9 nd_knn_f1, 10 nd_dup_better_h, 11 nd_dup_better_f1, 12 nd_dup_better_u,
    13 nd_ucent (cos(c,u) - cos(c,feedmean)), 14 nd_hcent (cos(c,h10) - cos(c,feedmean)),
    15 nd_knn_u_dev (cos_u - knn_u), 16 nd_knn_h_dev, 17 nd_max_samecat_gap (max cos to other-cat minus same-cat)
    feed_out: 0 mean pairwise cos, 1 share of pairs with cos > .9, 2 cos(user, feed mean), 3 cos(h10, feed mean)"""
    n = len(M)
    D = table.shape[1]
    for f in nb.prange(n):
        co = cand_off[f]
        m = M[f]
        V = np.zeros((m, D), np.float32)
        ok = np.zeros(m, np.bool_)
        fm = np.zeros(D, np.float32)
        for i in range(m):
            ei = cemb[co + i]
            if ei < 0:
                continue
            v = table[ei].astype(np.float32)
            nv = np.sqrt((v * v).sum()) + 1e-6
            V[i] = v / nv
            ok[i] = True
            fm += V[i]
        fmn = np.sqrt((fm * fm).sum())
        if fmn > 0:
            fm = fm / fmn
        u = user_emb[f].astype(np.float32)
        un = np.sqrt((u * u).sum())
        uok = un > 0
        if uok:
            u = u / un
        h = np.zeros(D, np.float32)
        ho = hist_off[f]
        nh = 0
        for j in range(H[f]):
            ei = hemb[ho + j]
            if ei < 0:
                continue
            v = table[ei].astype(np.float32)
            nv = np.sqrt((v * v).sum()) + 1e-6
            h += v / nv
            nh += 1
            if nh >= 10:
                break
        hn = np.sqrt((h * h).sum())
        hok = hn > 0
        if hok:
            h = h / hn
        S = V @ V.T
        # top-5 by cos_u and by maxsim
        cu = np.empty(m); ch = np.empty(m)
        for i in range(m):
            # verifier fix: a zero user vector gives cos_u == 0 (not NaN) for every candidate -> treat as missing
            # (otherwise top-5-by-cos_u is an arbitrary tie order and knn_u / knn_u_dev collapse to a fake 0)
            cu[i] = cos_u[co + i] if (ok[i] and uok and not np.isnan(cos_u[co + i])) else -9.0
            ch[i] = maxsim[co + i] if (ok[i] and not np.isnan(maxsim[co + i])) else -9.0
        ou = np.argsort(-cu)
        oh = np.argsort(-ch)
        npair = 0; spair = 0.0; p90 = 0
        for i in range(m):
            if not ok[i]:
                for k in range(18):
                    out[co + i, k] = np.nan
                continue
            mx = -2.0; c80 = 0; c90 = 0; c95 = 0; sm = 0.0; no = 0
            wsu = 0.0; wsh = 0.0; wsf = 0.0; wu = 0.0; wh = 0.0; wf = 0.0
            bh = 0; bf = 0; bu = 0
            mx_same = -2.0; mx_oth = -2.0
            for j in range(m):
                if j == i or not ok[j]:
                    continue
                s = S[i, j]
                no += 1
                sm += s
                if s > mx:
                    mx = s
                if s > 0.8:
                    c80 += 1
                if s > 0.9:
                    c90 += 1
                    if maxsim[co + j] > maxsim[co + i]:
                        bh += 1
                    if f1[co + j] > f1[co + i]:
                        bf += 1
                    if cos_u[co + j] > cos_u[co + i]:
                        bu += 1
                if s > 0.95:
                    c95 += 1
                w = np.exp(20.0 * (s - 1.0))
                if uok and not np.isnan(cos_u[co + j]):
                    wsu += w * cos_u[co + j]; wu += w
                if not np.isnan(maxsim[co + j]):
                    wsh += w * maxsim[co + j]; wh += w
                if not np.isnan(f1[co + j]):
                    wsf += w * f1[co + j]; wf += w
                if j > i:
                    npair += 1; spair += s
                    if s > 0.9:
                        p90 += 1
            out[co + i, 0] = mx if no > 0 else np.nan
            out[co + i, 1] = c80
            out[co + i, 2] = c90
            out[co + i, 3] = c95
            out[co + i, 4] = sm / no if no > 0 else np.nan
            # top-5 by cos_u excluding self
            t = 0; st = 0.0
            for r in range(m):
                j = ou[r]
                if j == i or cu[j] < -5:
                    continue
                st += S[i, j]; t += 1
                if t >= 5:
                    break
            out[co + i, 5] = st / t if t > 0 else np.nan
            t = 0; st = 0.0
            for r in range(m):
                j = oh[r]
                if j == i or ch[j] < -5:
                    continue
                st += S[i, j]; t += 1
                if t >= 5:
                    break
            out[co + i, 6] = st / t if t > 0 else np.nan
            out[co + i, 7] = wsu / wu if wu > 0 else np.nan
            out[co + i, 8] = wsh / wh if wh > 0 else np.nan
            out[co + i, 9] = wsf / wf if wf > 0 else np.nan
            out[co + i, 10] = bh
            out[co + i, 11] = bf
            out[co + i, 12] = bu
            cfm = (V[i] * fm).sum()
            out[co + i, 13] = (V[i] * u).sum() - cfm if uok else np.nan
            out[co + i, 14] = (V[i] * h).sum() - cfm if hok else np.nan
            out[co + i, 15] = cos_u[co + i] - out[co + i, 7]
            out[co + i, 16] = maxsim[co + i] - out[co + i, 8]
            out[co + i, 17] = np.nan
        feed_out[f, 0] = spair / npair if npair > 0 else np.nan
        feed_out[f, 1] = p90 / npair if npair > 0 else np.nan
        feed_out[f, 2] = (u * fm).sum() if (uok and fmn > 0) else np.nan
        feed_out[f, 3] = (h * fm).sum() if (hok and fmn > 0) else np.nan


ND_NAMES = ["ctx_nd_max", "ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_mean", "ctx_nd_top5u", "ctx_nd_top5h",
            "ctx_nd_knn_u", "ctx_nd_knn_h", "ctx_nd_knn_f1", "ctx_nd_dupb_h", "ctx_nd_dupb_f1", "ctx_nd_dupb_u",
            "ctx_nd_ucent", "ctx_nd_hcent", "ctx_nd_knn_u_dev", "ctx_nd_knn_h_dev", None]
ND_FEED = ["ctx_fs_div", "ctx_fs_dup_pairs", "ctx_fs_u_fm", "ctx_fs_h_fm"]


# ----------------------------------------------------------------------------- driver
def load_meta(split):
    feeds = pl.read_parquet(FLAT / split / "feeds.parquet")
    M = feeds["M"].to_numpy().astype(np.int64)
    H = feeds["H"].to_numpy().astype(np.int64)
    co = feeds["cand_off"].to_numpy().astype(np.int64)
    ho = feeds["hist_off"].to_numpy().astype(np.int64)
    return feeds, M, H, co, ho


import os
_KEEP = set(os.environ["CTX_KEEP"].split(",")) if os.environ.get("CTX_KEEP") else None  # optional whitelist of names


def save(split, name, arr):
    if _KEEP is not None and name not in _KEEP:
        return
    np.save(FEATS / split / f"{name}.npy", np.ascontiguousarray(arr, dtype=np.float32))


def build(split, groups):
    t = time.time()
    feeds, M, H, co, ho = load_meta(split)
    fidx = np.load(FLAT / split / "cand_fidx.npy")
    N = len(fidx)
    if {"fs", "ct", "cr", "cl"} & set(groups):
        X = np.empty((N, len(COLS)), np.float32)
        for k, c in enumerate(COLS):
            X[:, k] = np.load(FEATS / split / f"{c}.npy")
        cat = np.load(FLAT / split / "cand_category_id.npy").astype(np.int64)
        hcs = np.load(FEATS / split / "cat_share_all.npy")
        hcs = np.nan_to_num(hcs, nan=0.0).astype(np.float64)
        fnames, cnames = ctx_names()
        fo = np.empty((len(M), len(fnames)), np.float32)
        cand_out = np.empty((N, len(cnames)), np.float32)
        _ctx_kernel(co, M, X, cat, hcs, fo, cand_out)
        del X
        for i, nm in enumerate(cnames):
            g = nm.split("_")[1]
            if g in groups:
                if nm.endswith("_rsum") and np.all(np.isnan(cand_out[:, i])):
                    continue
                if nm.startswith("ctx_cr_f3_"):  # f3 is constant within (feed, category): useless
                    continue
                save(split, nm, cand_out[:, i])
        if "fs" in groups:
            for i, nm in enumerate(fnames):
                save(split, nm, fo[fidx, i])
        del cand_out
        print(split, "ctx kernel", f"{time.time()-t:.0f}s", flush=True)
    if "nd" in groups:
        table = load_item_emb(split)
        table = np.ascontiguousarray(table)
        ue = np.load(FLAT / split / "user_emb.npy")
        cemb = np.load(FLAT / split / "cand_emb_idx.npy").astype(np.int64)
        hemb = np.load(FLAT / split / "hist_emb_idx.npy").astype(np.int64)
        cos_u = np.load(FEATS / split / "cos_u.npy").astype(np.float32)
        maxsim = np.load(FEATS / split / "maxsim32.npy").astype(np.float32)
        f1 = np.load(FEATS / split / "f1.npy").astype(np.float32)
        print(split, "nd loaded", f"{time.time()-t:.0f}s", flush=True)
        out = np.empty((N, len(ND_NAMES)), np.float32)
        fo = np.empty((len(M), len(ND_FEED)), np.float32)
        _nd_kernel(co, M, ho, H, cemb, hemb, table, ue, cos_u, maxsim, f1, fo, out)
        del table
        for i, nm in enumerate(ND_NAMES):
            if nm is not None:
                save(split, nm, out[:, i])
        for i, nm in enumerate(ND_FEED):
            save(split, nm, fo[fidx, i])
        print(split, "nd", f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    nb.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 24)
    for s in sys.argv[1].split(","):
        build(s, sys.argv[2].split(","))
