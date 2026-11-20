"""Learned transition (Markov) models over history sequences: category level and embedding-cluster level.

Counts C[a, b] over pairs (older event a -> newer event b) within W=3 positions in de-duplicated train user
sequences (see sources.py), weight 1/distance, same-item pairs skipped (a candidate never appears in its own
feed's history, so same-item repeats would only inflate self-transitions). Two target sets: b of any event
type ("all") and b of event_type 1 ("t1", the event a feed click turns into).
Smoothed P(b|a) = (C[a,b] + alpha p(b)) / (C[a,.] + alpha); lift(a,b) = log P(b|a) / p(b).
Candidate score with the feed's last KMAX coded history events (newest first, weights 0.5^(hpos/hl)):
  *_lift  = sum_k w_k lift(a_k, b_c) / sum_k w_k                 (t1 matrix unless noted)
  *_mix   = log( sum_k w_k P(b_c|a_k) / sum_k w_k )              (Markov mixture, includes popularity of b)
  *_last  = lift(a_0, b_c)                                       (newest event only)
  *_xlift = like lift but only over history events with a_k != b_c (cross-cluster intent transitions)
Matrices for train rows come from sources strictly before the row's day; val/test use all of train.
Usage: python feat_trans.py <split>[,..] [threads]
"""
import sys
import numba as nb
import numpy as np
from seq_lib import *
from sources import train_days, source_fidx, feed_day_index

W = 3
KMAX = 10
HL = 3.0
GRAN = {"cat": 54, "c1024": 1024, "c4096": 4096}
ALPHA = {"cat": 100.0, "c1024": 50.0, "c4096": 20.0}


@nb.njit(parallel=True, cache=True)
def _acc(src, hist_off, H, hitem, hcode, het, K, T, out_all, out_t1):
    n = len(src)
    for t in nb.prange(T):
        for q in range(t, n, T):
            f = src[q]; ho = hist_off[f]; h = H[f]
            for j in range(h):
                b = hcode[ho + j]
                if b < 0:
                    continue
                is1 = het[ho + j] == 1
                for d in range(1, W + 1):
                    if j + d >= h:
                        break
                    a = hcode[ho + j + d]
                    if a < 0 or hitem[ho + j + d] == hitem[ho + j]:
                        continue
                    w = 1.0 / d
                    out_all[t, a, b] += w
                    if is1:
                        out_t1[t, a, b] += w


def hist_codes(split, g):
    if g == "cat":
        return np.asarray(arr(split, "hist_category_id")).astype(np.int16)
    cl = np.load(CLU / f"item_cl_{g[1:]}_{split}.npy")
    e = np.asarray(arr(split, "hist_emb_idx"))
    return np.where(e >= 0, cl[np.maximum(e, 0)], -1).astype(np.int16)


def cand_codes(split, g):
    if g == "cat":
        return np.asarray(arr(split, "cand_category_id")).astype(np.int16)
    cl = np.load(CLU / f"item_cl_{g[1:]}_{split}.npy")
    e = np.asarray(arr(split, "cand_emb_idx"))
    return np.where(e >= 0, cl[np.maximum(e, 0)], -1).astype(np.int16)


def build_matrices(g, threads, need_cutoffs):
    """Returns float32 arrays [n_cut+1, 2(all,t1), K, K] of counts; last index = all-train source.
    Each (granularity, cutoff) is cached separately in data/seq_work/."""
    K = GRAN[g]
    f = feeds("train")
    ho = f["hist_off"].to_numpy(); H = f["H"].to_numpy().astype(np.int64)
    days = train_days()
    cuts = list(days) + [None]
    res = np.zeros((len(cuts), 2, K, K), np.float32)
    T = threads if K <= 1024 else min(threads, 12)
    loaded = None
    for i, c in enumerate(cuts):
        if c is not None and not need_cutoffs:
            continue
        path = WORK / f"trans_{g}_{'all' if c is None else int(c)}.npy"
        if path.exists():
            res[i] = np.load(path); continue
        if loaded is None:
            loaded = (np.asarray(arr("train", "hist_item_id")), np.asarray(arr("train", "hist_event_type")),
                      hist_codes("train", g))
        hitem, het, hcode = loaded
        src = source_fidx(c)
        if len(src) > 0:
            oa = np.zeros((T, K, K), np.float64); o1 = np.zeros((T, K, K), np.float64)
            _acc(src, ho, H, hitem, hcode, het, K, T, oa, o1)
            res[i, 0] = oa.sum(0); res[i, 1] = o1.sum(0)
            del oa, o1
        np.save(path, res[i])
        log(g, "cutoff", c, "src", len(src), "pairs", res[i, 0].sum())
    return res


def to_tables(C, alpha):
    """C [K,K] counts -> (logP [K,K], lift [K,K]) float32."""
    C = C.astype(np.float64)
    pb = (C.sum(0) + 1.0) / (C.sum() + C.shape[0])
    P = (C + alpha * pb[None, :]) / (C.sum(1, keepdims=True) + alpha)
    return np.log(P).astype(np.float32), np.log(P / pb[None, :]).astype(np.float32)


@nb.njit(parallel=True, fastmath=True, cache=True)
def _score(cand_off, M, hist_off, H, ccode, hcode, fmat, LPa, LIa, LP1, LI1, out):
    """out cols: 0 lift_all, 1 lift_t1, 2 mix_t1, 3 last_t1, 4 xlift_t1"""
    n = len(M)
    for f in nb.prange(n):
        co = cand_off[f]; m = M[f]; ho = hist_off[f]; h = H[f]
        mi = fmat[f]
        a = np.empty(KMAX, np.int64); w = np.empty(KMAX, np.float64)
        na = 0
        for j in range(h):
            c = hcode[ho + j]
            if c < 0:
                continue
            a[na] = c; w[na] = 0.5 ** (j / HL); na += 1
            if na >= KMAX:
                break
        sw = 0.0
        for k in range(na):
            sw += w[k]
        for i in range(m):
            r = co + i
            b = ccode[r]
            if b < 0 or na == 0 or mi < 0:
                for q in range(5):
                    out[r, q] = np.nan
                continue
            la = 0.0; l1 = 0.0; mx = 0.0; xl = 0.0; xw = 0.0
            for k in range(na):
                la += w[k] * LIa[mi, a[k], b]
                l1 += w[k] * LI1[mi, a[k], b]
                mx += w[k] * np.exp(LP1[mi, a[k], b])
                if a[k] != b:
                    xl += w[k] * LI1[mi, a[k], b]; xw += w[k]
            out[r, 0] = la / sw
            out[r, 1] = l1 / sw
            out[r, 2] = np.log(mx / sw)
            out[r, 3] = LI1[mi, a[0], b]
            out[r, 4] = xl / xw if xw > 0 else np.nan


def run(split, threads):
    days = train_days()
    fm = feed_day_index(split, days)
    f = feeds(split)
    M = f["M"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
    co = f["cand_off"].to_numpy(); ho = f["hist_off"].to_numpy()
    N = int(M.sum())
    for g in GRAN:
        R = build_matrices(g, threads, split == 'train')
        nm = R.shape[0]
        if split != "train":
            sel = [nm - 1]
        else:
            sel = list(range(nm - 1))
        idx_map = -np.ones(nm, np.int64)
        tabs = [[], [], [], []]
        for k, i in enumerate(sel):
            if R[i, 0].sum() == 0:
                continue
            idx_map[i] = len(tabs[0])
            lpa, lia = to_tables(R[i, 0], ALPHA[g]); lp1, li1 = to_tables(R[i, 1], ALPHA[g])
            for t, x in zip(tabs, (lpa, lia, lp1, li1)):
                t.append(x)
        LPa, LIa, LP1, LI1 = (np.stack(t) for t in tabs)
        fmat = idx_map[fm]
        out = np.empty((N, 5), np.float32)
        _score(co, M, ho, H, cand_codes(split, g), hist_codes(split, g), fmat, LPa, LIa, LP1, LI1, out)
        for q, s in enumerate(["lift_all", "lift", "mix", "last", "xlift"]):
            save(split, f"seq_tr_{g}_{s}", out[:, q])
        log(split, g, "done")


if __name__ == "__main__":
    th = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    nb.set_num_threads(th)
    for s in sys.argv[1].split(","):
        run(s, th)
