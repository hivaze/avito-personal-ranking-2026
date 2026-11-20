"""Candidate-level feature builder.

build_split(split) -> writes data/feats/<split>/<name>.npy (float32, flat, aligned with data/flat/<split>/cand_*.npy)
Feature groups:
  base      f1..f5, category, M, H, emb_missing
  rel       within-feed pct-rank / z-score / gap-to-max of f1..f5 and embedding scores
  emb       user-item dot/cos, item norm, candidate vs history-pooled cosines (all/last10/last50/per event type),
            max-sim and soft attention over last 32 history items, centrality within the feed
  cat       category affinity with history (counts, shares, recency, per event type), feed category share
  item      global item popularity from all histories (transductive), train impressions/positives/CTR/COEC
  user      user-level train stats
"""
import json, sys, time
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl

ROOT = Path("<repo>/data")
FLAT = ROOT / "flat"
FEATS = ROOT / "feats"
N_ET = 10       # history event types 1..10
N_CAT = 54      # categories 0..53
LASTK = 32      # history items used for max-sim / attention


def load_split(split):
    d = FLAT / split
    feeds = pl.read_parquet(d / "feeds.parquet")
    a = {k: np.load(d / f"cand_{k}.npy", mmap_mode="r") for k in
         ["fidx", "item_id", "category_id", "f1", "f2", "f3", "f4", "f5", "emb_idx"]}
    for k in ["target", "pos"]:
        if (d / f"cand_{k}.npy").exists():
            a[k] = np.load(d / f"cand_{k}.npy", mmap_mode="r")
    h = {k: np.load(d / f"hist_{k}.npy", mmap_mode="r") for k in
         ["fidx", "item_id", "category_id", "event_type", "event_source", "emb_idx", "hpos"]}
    return feeds, a, h


def load_item_emb(split):
    d = ROOT / "competition" / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.fromfile(d / "data.memmap", dtype=np.int8).reshape(meta["shape"])


# ----------------------------------------------------------------------------- embedding features
@nb.njit(parallel=True, fastmath=True, cache=True)
def _emb_feats(cand_off, M, hist_off, H, cemb_idx, hemb_idx, het, user_emb, table, out):
    """out columns:
    0 dot_u, 1 cos_u, 2 item_norm, 3 cos_hall, 4 cos_h10, 5 cos_h50, 6..15 cos_het[1..10],
    16 maxsim32, 17 attn32 (logsumexp(5*cos)/5), 18 mean_sim32, 19 centrality (cos to feed mean),
    20 dot_hall (unnormalized dot with mean hist vec)"""
    n = len(M)
    D = table.shape[1]
    for f in nb.prange(n):
        u = user_emb[f].astype(np.float32)
        un = np.sqrt((u * u).sum()) + 1e-6
        hall = np.zeros(D, np.float32); h10 = np.zeros(D, np.float32); h50 = np.zeros(D, np.float32)
        het_m = np.zeros((N_ET + 1, D), np.float32)
        c_all = 0; c10 = 0; c50 = 0
        het_c = np.zeros(N_ET + 1, np.int64)
        ho = hist_off[f]
        nk = 0
        lastk = np.zeros((LASTK, D), np.float32)
        for j in range(H[f]):
            ei = hemb_idx[ho + j]
            if ei < 0:
                continue
            v = table[ei].astype(np.float32)
            vn = np.sqrt((v * v).sum()) + 1e-6
            v = v / vn
            hall += v; c_all += 1
            if j < 10:
                h10 += v; c10 += 1
            if j < 50:
                h50 += v; c50 += 1
            et = het[ho + j]
            if 0 < et <= N_ET:
                het_m[et] += v; het_c[et] += 1
            if nk < LASTK:
                lastk[nk] = v; nk += 1
        # normalize pooled vectors
        hn = np.sqrt((hall * hall).sum()); h10n = np.sqrt((h10 * h10).sum()); h50n = np.sqrt((h50 * h50).sum())
        hetn = np.zeros(N_ET + 1, np.float32)
        for e in range(1, N_ET + 1):
            hetn[e] = np.sqrt((het_m[e] * het_m[e]).sum())
        co = cand_off[f]
        m = M[f]
        cmean = np.zeros(D, np.float32)
        cvecs = np.zeros((m, D), np.float32)
        for i in range(m):
            ei = cemb_idx[co + i]
            if ei < 0:
                for k in range(21):
                    out[co + i, k] = np.nan
                continue
            v = table[ei].astype(np.float32)
            vn = np.sqrt((v * v).sum()) + 1e-6
            dot = (v * u).sum()
            out[co + i, 0] = dot
            out[co + i, 1] = dot / (vn * un)
            out[co + i, 2] = vn
            v = v / vn
            cvecs[i] = v
            cmean += v
            out[co + i, 3] = (v * hall).sum() / hn if hn > 0 else np.nan
            out[co + i, 4] = (v * h10).sum() / h10n if h10n > 0 else np.nan
            out[co + i, 5] = (v * h50).sum() / h50n if h50n > 0 else np.nan
            for e in range(1, N_ET + 1):
                out[co + i, 5 + e] = (v * het_m[e]).sum() / hetn[e] if hetn[e] > 0 else np.nan
            if nk > 0:
                mx = -2.0; se = 0.0; sm = 0.0
                for k in range(nk):
                    s = (v * lastk[k]).sum()
                    if s > mx:
                        mx = s
                    se += np.exp(5.0 * s)
                    sm += s
                out[co + i, 16] = mx
                out[co + i, 17] = np.log(se) / 5.0
                out[co + i, 18] = sm / nk
            else:
                out[co + i, 16] = np.nan; out[co + i, 17] = np.nan; out[co + i, 18] = np.nan
            out[co + i, 20] = (v * hall).sum() / c_all if c_all > 0 else np.nan
        cmn = np.sqrt((cmean * cmean).sum())
        for i in range(m):
            if cemb_idx[co + i] >= 0 and cmn > 0:
                out[co + i, 19] = (cvecs[i] * cmean).sum() / cmn
            else:
                out[co + i, 19] = np.nan


EMB_NAMES = ["dot_u", "cos_u", "item_norm", "cos_hall", "cos_h10", "cos_h50"] + \
            [f"cos_het{e}" for e in range(1, N_ET + 1)] + ["maxsim32", "attn32", "meansim32", "centrality", "dot_hall"]


# ----------------------------------------------------------------------------- category affinity
@nb.njit(parallel=True, cache=True)
def _cat_feats(cand_off, M, hist_off, H, ccat, hcat, het, out):
    """0 cat_cnt_all, 1 cat_share_all, 2 cat_cnt_20, 3 cat_min_hpos (recency; 999 if none),
    4..13 cat_cnt_et[1..10], 14 feed_cat_share, 15 hist_n_unique_cat, 16 cat_cnt_5"""
    n = len(M)
    for f in nb.prange(n):
        cnt = np.zeros(N_CAT, np.float32); c20 = np.zeros(N_CAT, np.float32); c5 = np.zeros(N_CAT, np.float32)
        minh = np.full(N_CAT, 999.0, np.float32)
        cet = np.zeros((N_CAT, N_ET + 1), np.float32)
        ho = hist_off[f]
        h = H[f]
        for j in range(h):
            c = hcat[ho + j]
            if c < 0 or c >= N_CAT:
                continue
            cnt[c] += 1
            if j < 20:
                c20[c] += 1
            if j < 5:
                c5[c] += 1
            if minh[c] > j:
                minh[c] = j
            e = het[ho + j]
            if 0 < e <= N_ET:
                cet[c, e] += 1
        nu = 0
        for c in range(N_CAT):
            if cnt[c] > 0:
                nu += 1
        fc = np.zeros(N_CAT, np.float32)
        co = cand_off[f]
        m = M[f]
        for i in range(m):
            fc[ccat[co + i]] += 1
        for i in range(m):
            c = ccat[co + i]
            out[co + i, 0] = cnt[c]
            out[co + i, 1] = cnt[c] / h if h > 0 else np.nan
            out[co + i, 2] = c20[c]
            out[co + i, 3] = minh[c]
            for e in range(1, N_ET + 1):
                out[co + i, 3 + e] = cet[c, e]
            out[co + i, 14] = fc[c] / m
            out[co + i, 15] = nu
            out[co + i, 16] = c5[c]


CAT_NAMES = ["cat_cnt_all", "cat_share_all", "cat_cnt_20", "cat_min_hpos"] + \
            [f"cat_cnt_et{e}" for e in range(1, N_ET + 1)] + ["feed_cat_share", "hist_n_uniq_cat", "cat_cnt_5"]


# ----------------------------------------------------------------------------- within-feed relative
@nb.njit(parallel=True, cache=True)
def _rel_feats(cand_off, M, x, out):
    """per column of x: pct rank (0=lowest,1=highest, avg ties), z-score, gap to max"""
    n = len(M)
    ncol = x.shape[1]
    for f in nb.prange(n):
        co = cand_off[f]
        m = M[f]
        for c in range(ncol):
            v = x[co:co + m, c]
            order = np.argsort(v)
            r = np.empty(m, np.float32)
            i = 0
            while i < m:
                j = i
                while j + 1 < m and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2.0
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            mu = 0.0; cntv = 0; mx = -1e30
            for k in range(m):
                if not np.isnan(v[k]):
                    mu += v[k]; cntv += 1
                    if v[k] > mx:
                        mx = v[k]
            mu = mu / cntv if cntv > 0 else 0.0
            sd = 0.0
            for k in range(m):
                if not np.isnan(v[k]):
                    sd += (v[k] - mu) ** 2
            sd = np.sqrt(sd / cntv) if cntv > 0 else 0.0
            for k in range(m):
                out[co + k, 3 * c] = r[k] / (m - 1) if m > 1 else 0.5
                out[co + k, 3 * c + 1] = (v[k] - mu) / (sd + 1e-6)
                out[co + k, 3 * c + 2] = v[k] - mx


# ----------------------------------------------------------------------------- global item stats
def global_item_stats():
    """Transductive item popularity from ALL histories + candidate counts per split + train target stats."""
    path = FEATS / "item_stats.npz"
    if path.exists():
        return dict(np.load(path))
    t = time.time()
    NI = 111_400_000
    st = {}
    hist_all = np.zeros(NI, np.float32)
    hist_et = np.zeros((N_ET + 1, NI), np.float32)
    for s in ["train", "val", "test"]:
        hi = np.load(FLAT / s / "hist_item_id.npy")
        he = np.load(FLAT / s / "hist_event_type.npy")
        hist_all += np.bincount(hi, minlength=NI).astype(np.float32)
        for e in range(1, N_ET + 1):
            hist_et[e] += np.bincount(hi[he == e], minlength=NI).astype(np.float32)
        ci = np.load(FLAT / s / "cand_item_id.npy")
        st[f"cand_cnt_{s}"] = np.bincount(ci, minlength=NI).astype(np.float32)
    st["hist_all"] = hist_all
    for e in range(1, N_ET + 1):
        st[f"hist_et{e}"] = hist_et[e]
    # train target stats + position-debiased expected clicks (COEC)
    ci = np.load(FLAT / "train" / "cand_item_id.npy")
    tg = np.load(FLAT / "train" / "cand_target.npy").astype(np.float32)
    pos = np.load(FLAT / "train" / "cand_pos.npy").astype(np.int64)
    pos_ctr = np.bincount(pos, weights=tg, minlength=301) / np.maximum(np.bincount(pos, minlength=301), 1)
    st["pos_ctr"] = pos_ctr.astype(np.float32)
    st["tr_pos"] = np.bincount(ci, weights=tg, minlength=NI).astype(np.float32)
    st["tr_exp"] = np.bincount(ci, weights=pos_ctr[pos], minlength=NI).astype(np.float32)
    np.savez(path, **st)
    print("item stats", f"{time.time()-t:.0f}s", flush=True)
    return st


def user_stats():
    path = FEATS / "user_stats.npz"
    if path.exists():
        return dict(np.load(path))
    feeds = pl.read_parquet(FLAT / "train" / "feeds.parquet")
    tg = np.load(FLAT / "train" / "cand_target.npy").astype(np.float32)
    fidx = np.load(FLAT / "train" / "cand_fidx.npy")
    fpos = np.bincount(fidx, weights=tg, minlength=len(feeds))
    uid = feeds["user_id"].to_numpy()
    NU = int(max(uid.max(), *[pl.read_parquet(FLAT / s / "feeds.parquet")["user_id"].max() for s in ["val", "test"]])) + 1
    st = {"u_feeds": np.bincount(uid, minlength=NU).astype(np.float32),
          "u_cands": np.bincount(uid, weights=feeds["M"].to_numpy(), minlength=NU).astype(np.float32),
          "u_pos": np.bincount(uid, weights=fpos, minlength=NU).astype(np.float32)}
    np.savez(path, **st)
    return st


# ----------------------------------------------------------------------------- driver
def build_split(split, groups=("base", "emb", "cat", "item", "user", "rel")):
    out_dir = FEATS / split
    out_dir.mkdir(parents=True, exist_ok=True)
    feeds, a, h = load_split(split)
    M = feeds["M"].to_numpy().astype(np.int64)
    H = feeds["H"].to_numpy().astype(np.int64)
    co = feeds["cand_off"].to_numpy()
    ho = feeds["hist_off"].to_numpy()
    fidx = np.asarray(a["fidx"])
    N = len(fidx)
    names = []

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", np.asarray(arr, dtype=np.float32))
        names.append(name)

    t = time.time()
    if "base" in groups:
        for k in ["f1", "f2", "f3", "f4", "f5"]:
            save(k, a[k])
        save("category_id", a["category_id"])
        save("M", M[fidx]); save("H", H[fidx])
        save("emb_missing", np.asarray(a["emb_idx"]) < 0)
        ts = feeds["timestamp"].to_numpy()
        save("hour", ((ts + 3 * 3600) // 3600 % 24)[fidx])  # Moscow time
        het = np.asarray(h["event_type"])
        hf = np.asarray(h["fidx"])
        for e in range(1, N_ET + 1):
            save(f"u_het{e}_share", (np.bincount(hf, weights=(het == e), minlength=len(M)) / np.maximum(H, 1))[fidx])
        print(split, "base", f"{time.time()-t:.0f}s", flush=True)
    if "emb" in groups:
        table = load_item_emb(split)
        ue = np.load(FLAT / split / "user_emb.npy")
        out = np.empty((N, len(EMB_NAMES)), np.float32)
        _emb_feats(co, M, ho, H, np.asarray(a["emb_idx"]), np.asarray(h["emb_idx"]), np.asarray(h["event_type"]),
                   ue, table, out)
        del table
        for i, nm in enumerate(EMB_NAMES):
            save(nm, out[:, i])
        print(split, "emb", f"{time.time()-t:.0f}s", flush=True)
    if "cat" in groups:
        out = np.empty((N, len(CAT_NAMES)), np.float32)
        _cat_feats(co, M, ho, H, np.asarray(a["category_id"]).astype(np.int64), np.asarray(h["category_id"]).astype(np.int64),
                   np.asarray(h["event_type"]).astype(np.int64), out)
        for i, nm in enumerate(CAT_NAMES):
            save(nm, out[:, i])
        print(split, "cat", f"{time.time()-t:.0f}s", flush=True)
    if "item" in groups:
        st = global_item_stats()
        ci = np.asarray(a["item_id"])
        save("it_hist_all", st["hist_all"][ci])
        for e in range(1, N_ET + 1):
            save(f"it_hist_et{e}", st[f"hist_et{e}"][ci])
        cc = st[f"cand_cnt_{split}"][ci]
        save("it_cand_cnt_split", cc / len(M) * 1e4)    # per 10k feeds of this split (transductive)
        save("it_cand_cnt_train", st["cand_cnt_train"][ci])
        trp, tre = st["tr_pos"][ci], st["tr_exp"][ci]
        if split == "train":  # leave-own-feed-out is approximated by removing own label
            tg = np.asarray(a["target"]).astype(np.float32)
            pos_ctr = st["pos_ctr"]
            trp = trp - tg
            tre = tre - pos_ctr[np.asarray(a["pos"]).astype(np.int64)]
            cnt = st["cand_cnt_train"][ci] - 1
        else:
            cnt = st["cand_cnt_train"][ci]
        prior = 0.0159
        save("it_tr_ctr", (trp + prior * 20) / (cnt + 20))
        save("it_tr_coec", (trp + 2.0) / (tre + 2.0))
        save("it_tr_pos", trp)
        print(split, "item", f"{time.time()-t:.0f}s", flush=True)
    if "user" in groups:
        us = user_stats()
        uid = feeds["user_id"].to_numpy()[fidx]
        save("u_tr_feeds", us["u_feeds"][uid])
        save("u_tr_ctr", (us["u_pos"][uid] + 0.0159 * 50) / (us["u_cands"][uid] + 50))
        print(split, "user", f"{time.time()-t:.0f}s", flush=True)
    if "rel" in groups:
        cols = ["f1", "f2", "f3", "f4", "f5", "cos_u", "cos_hall", "cos_h10", "maxsim32", "attn32", "cat_cnt_all",
                "it_hist_all", "it_cand_cnt_split", "it_tr_coec"]
        x = np.stack([np.load(out_dir / f"{c}.npy") for c in cols], axis=1)
        out = np.empty((N, 3 * len(cols)), np.float32)
        _rel_feats(co, M, x, out)
        for i, c in enumerate(cols):
            save(f"{c}_rk", out[:, 3 * i]); save(f"{c}_z", out[:, 3 * i + 1]); save(f"{c}_gap", out[:, 3 * i + 2])
        print(split, "rel", f"{time.time()-t:.0f}s", flush=True)
    return names


if __name__ == "__main__":
    nb.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 64)
    for s in sys.argv[1].split(","):
        build_split(s)
