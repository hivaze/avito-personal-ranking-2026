"""Restore the near-duplicate context features that were pruned from data/feats while the NN fine-tune preset
n2c_pre still listed them (featrev inventory, INVENTORY-3):
    ctx_nd_c80, ctx_nd_c90, ctx_nd_c95   # other candidates of the same feed with item-embedding cosine > .8/.9/.95
    ctx_nd_mean                          mean cosine to the other candidates of the feed
    ctx_nd_dupb_h / _f1 / _u             # near-duplicates (cos > .9) with a higher maxsim32 / f1 / cos_u
The kernel is copied verbatim from feats_v2/ctx_/build_ctx.py:_nd_kernel (only numba cache disabled); every input is
feed-internal (the feed's own candidates, its history, the user embedding and the split's own item table; no labels,
no other feeds, no time windows), so train / val / test are constructed identically.
Self-check: the 10 ctx_nd_* columns that still exist in data/feats are recomputed and compared (max abs diff).
Output: data/feats_new/<split>/<name>.npy (float32, aligned with data/flat/<split>/cand_*).
Run: CUDA_VISIBLE_DEVICES="" NUMBA_NUM_THREADS=16 python feat_new/inventory_ctx_nd_restore.py val,test [threads]
"""
import json, sys, time
from pathlib import Path
import numba as nb
import numpy as np
import polars as pl

ROOT = Path("<repo>/data")
FLAT, FEATS, OUT = ROOT / "flat", ROOT / "feats", ROOT / "feats_new"
ND_NAMES = ["ctx_nd_max", "ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_mean", "ctx_nd_top5u", "ctx_nd_top5h",
            "ctx_nd_knn_u", "ctx_nd_knn_h", "ctx_nd_knn_f1", "ctx_nd_dupb_h", "ctx_nd_dupb_f1", "ctx_nd_dupb_u",
            "ctx_nd_ucent", "ctx_nd_hcent", "ctx_nd_knn_u_dev", "ctx_nd_knn_h_dev", None]
RESTORE = ["ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_mean", "ctx_nd_dupb_h", "ctx_nd_dupb_f1", "ctx_nd_dupb_u"]


def load_item_emb(split):
    d = ROOT / "competition" / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.memmap(d / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


@nb.njit(parallel=True, fastmath=True, cache=False)
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


def build(split):
    t = time.time()
    feeds = pl.read_parquet(FLAT / split / "feeds.parquet")
    M = feeds["M"].to_numpy().astype(np.int64); H = feeds["H"].to_numpy().astype(np.int64)
    co = feeds["cand_off"].to_numpy().astype(np.int64); ho = feeds["hist_off"].to_numpy().astype(np.int64)
    table = np.ascontiguousarray(load_item_emb(split))
    ue = np.load(FLAT / split / "user_emb.npy")
    cemb = np.load(FLAT / split / "cand_emb_idx.npy").astype(np.int64)
    hemb = np.load(FLAT / split / "hist_emb_idx.npy").astype(np.int64)
    cos_u = np.load(FEATS / split / "cos_u.npy").astype(np.float32)
    maxsim = np.load(FEATS / split / "maxsim32.npy").astype(np.float32)
    f1 = np.load(FEATS / split / "f1.npy").astype(np.float32)
    N = len(cemb)
    out = np.empty((N, len(ND_NAMES)), np.float32)
    fo = np.empty((len(M), 4), np.float32)
    _nd_kernel(co, M, ho, H, cemb, hemb, table, ue, cos_u, maxsim, f1, fo, out)
    del table
    for i, nm in enumerate(ND_NAMES):
        if nm is None:
            continue
        if nm in RESTORE:
            (OUT / split).mkdir(parents=True, exist_ok=True)
            np.save(OUT / split / f"{nm}.npy", np.ascontiguousarray(out[:, i], dtype=np.float32))
        elif (FEATS / split / f"{nm}.npy").exists():  # self-check against the live file
            ref = np.load(FEATS / split / f"{nm}.npy", mmap_mode="r")
            a, b = out[:, i], np.asarray(ref)
            both = ~np.isnan(a) & ~np.isnan(b)
            print(f"  check {split} {nm}: NaN mismatch {int((np.isnan(a) != np.isnan(b)).sum())}, "
                  f"max abs diff {float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0:.2e}", flush=True)
    print(split, "rows", N, f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    nb.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 16)
    for s in sys.argv[1].split(","):
        build(s)
