"""ctx_ extras (second batch).
groups:
  zs  unsupervised score fusion inside the feed: z-sum and Borda (mean pct rank) of f1 / maxsim32 / cos_u (+cos_h10, attn32),
      and their within-category pct rank / gap to the feed max
  sg  "source signature" of a candidate (which of f1..f5 sit on their minimum / exact-zero value) and its share in the feed,
      feed share of f1-at-minimum candidates
  ce  category-level extras: category mean f1 / cos_u in the feed minus the feed mean; similarity to the top-5 candidates by f1
Usage: python build_ctx2.py <splits> <groups> [threads]
"""
import json, sys, time
from pathlib import Path
import numba as nb
import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/feats_v2/ctx_")
from build_ctx import load_meta, load_item_emb, save, FLAT, FEATS, N_CAT

FMIN = dict(f1=1 / 56, f2=0.0, f3=1 / 62, f4=1 / 64, f5=0.0)


@nb.njit(parallel=True, cache=True)
def _group_rank_gap(cand_off, M, cat, x, out_crk, out_gap):
    """within-category pct rank (1 = best) of x, and x - feed max."""
    for f in nb.prange(len(M)):
        co = cand_off[f]; m = M[f]
        mx = -1e30
        for i in range(m):
            if x[co + i] > mx:
                mx = x[co + i]
        for i in range(m):
            out_gap[co + i] = x[co + i] - mx
            c = cat[co + i]
            lo = 0; eq = 0; n = 0
            for j in range(m):
                if cat[co + j] != c:
                    continue
                n += 1
                if x[co + j] < x[co + i]:
                    lo += 1
                elif x[co + j] == x[co + i]:
                    eq += 1
            out_crk[co + i] = (lo + (eq - 1) / 2.0) / (n - 1) if n > 1 else 0.5


@nb.njit(parallel=True, cache=True)
def _sig_kernel(cand_off, M, sig, f1min, out_share, out_feed_f1min):
    for f in nb.prange(len(M)):
        co = cand_off[f]; m = M[f]
        cnt = np.zeros(32, np.int64)
        nmin = 0
        for i in range(m):
            cnt[sig[co + i]] += 1
            nmin += f1min[co + i]
        for i in range(m):
            out_share[co + i] = cnt[sig[co + i]] / m
            out_feed_f1min[co + i] = nmin / m


@nb.njit(parallel=True, fastmath=True, cache=True)
def _ce_kernel(cand_off, M, cat, f1, cos_u, cemb, table, out):
    """0 cat mean f1 - feed mean f1, 1 cat mean cos_u - feed mean, 2 mean cos to top-5 candidates by f1 (excl. self)"""
    D = table.shape[1]
    for f in nb.prange(len(M)):
        co = cand_off[f]; m = M[f]
        sf = np.zeros(N_CAT); su = np.zeros(N_CAT); nc = np.zeros(N_CAT)
        tf = 0.0; tu = 0.0
        for i in range(m):
            c = cat[co + i]
            sf[c] += f1[co + i]; su[c] += cos_u[co + i]; nc[c] += 1
            tf += f1[co + i]; tu += cos_u[co + i]
        V = np.zeros((m, D), np.float32)
        ok = np.zeros(m, np.bool_)
        for i in range(m):
            ei = cemb[co + i]
            if ei >= 0:
                v = table[ei].astype(np.float32)
                V[i] = v / (np.sqrt((v * v).sum()) + 1e-6)
                ok[i] = True
        S = V @ V.T
        o = np.argsort(-f1[co:co + m])
        for i in range(m):
            c = cat[co + i]
            out[co + i, 0] = sf[c] / nc[c] - tf / m
            out[co + i, 1] = su[c] / nc[c] - tu / m
            if not ok[i]:
                out[co + i, 2] = np.nan
                continue
            t = 0; st = 0.0
            for r in range(m):
                j = o[r]
                if j == i or not ok[j]:
                    continue
                st += S[i, j]; t += 1
                if t >= 5:
                    break
            out[co + i, 2] = st / t if t > 0 else np.nan


def ld(split, n):
    return np.load(FEATS / split / f"{n}.npy").astype(np.float32)


def build(split, groups):
    t = time.time()
    feeds, M, H, co, ho = load_meta(split)
    cat = np.load(FLAT / split / "cand_category_id.npy").astype(np.int64)
    if "zs" in groups:
        z3 = np.nan_to_num(ld(split, "f1_z")) + np.nan_to_num(ld(split, "maxsim32_z")) + np.nan_to_num(ld(split, "cos_u_z"))
        z5 = z3 + np.nan_to_num(ld(split, "cos_h10_z")) + np.nan_to_num(ld(split, "attn32_z"))
        b3 = (np.nan_to_num(ld(split, "f1_rk"), nan=0.0) + np.nan_to_num(ld(split, "maxsim32_rk"), nan=0.0)
              + np.nan_to_num(ld(split, "cos_u_rk"), nan=0.0)) / 3
        for nm, x in [("ctx_zs_z3", z3), ("ctx_zs_z5", z5), ("ctx_zs_b3", b3)]:
            save(split, nm, x)
            crk = np.empty_like(x); gap = np.empty_like(x)
            _group_rank_gap(co, M, cat, x.astype(np.float32), crk, gap)
            save(split, nm + "_crk", crk)
            save(split, nm + "_gap", gap)
        print(split, "zs", f"{time.time()-t:.0f}s", flush=True)
    if "sg" in groups:
        sig = np.zeros(len(cat), np.int64)
        for b, (k, mn) in enumerate(FMIN.items()):
            v = np.load(FLAT / split / f"cand_{k}.npy")
            sig |= (np.abs(v - mn) < 1e-7).astype(np.int64) << b
        f1min = (sig & 1).astype(np.int64)
        sh = np.empty(len(cat), np.float32); fm = np.empty(len(cat), np.float32)
        _sig_kernel(co, M, sig, f1min, sh, fm)
        save(split, "ctx_sg_share", sh)
        save(split, "ctx_sg_feed_f1min", fm)
        print(split, "sg", f"{time.time()-t:.0f}s", flush=True)
    if "ce" in groups:
        table = np.ascontiguousarray(load_item_emb(split))
        cemb = np.load(FLAT / split / "cand_emb_idx.npy").astype(np.int64)
        f1 = ld(split, "f1"); cu = np.nan_to_num(ld(split, "cos_u"))
        out = np.empty((len(cat), 3), np.float32)
        _ce_kernel(co, M, cat, f1, cu, cemb, table, out)
        for i, nm in enumerate(["ctx_ce_cat_f1_dev", "ctx_ce_cat_u_dev", "ctx_ce_top5f1"]):
            save(split, nm, out[:, i])
        print(split, "ce", f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    nb.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 24)
    for s in sys.argv[1].split(","):
        build(s, sys.argv[2].split(","))
