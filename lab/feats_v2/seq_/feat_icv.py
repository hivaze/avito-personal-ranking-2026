"""Item -> cluster co-visitation (dense variant of item co-vis that also works for cold candidates).

For the feed's NH most recent unique history items x (weight 0.5^(hpos/10)), scan x's occurrences in the source
sequences (same Index/source rules as feat_covis.py: causal val/test, expanding-window train, same user skipped) and
accumulate the embedding-cluster ids of x's neighbours within +-3 positions (weight 1/d, same item skipped) into a
histogram over clusters. The candidate is scored by its own cluster:
  seq_icv_share4096  hist4096[cl(c)] / total
  seq_icv_lift4096   log((hist4096[cl] + a p(cl)) / (total + a) / p(cl))   (p = cluster prior over source events)
  seq_icv_lift1024   same at K=1024
  seq_icv_tot        total histogram mass (per feed; 0 when no history item is known in the source)
Only the history items must be known in the source; the candidate only needs an embedding.
"""
import sys
import numba as nb
import numpy as np
from seq_lib import *
from sources import train_days, source_fidx, day_of
from feat_covis import Index

NH = 40
CAPI = 500
NAMES = ["seq_icv_share4096", "seq_icv_lift4096", "seq_icv_lift1024", "seq_icv_tot"]


@nb.njit(parallel=True, cache=True)
def _icv(sel, coff, M, hoff, H, fuser, fts, cc1, cc2, hitem, hpos, items, codes1, codes2, starts, lens, susers, sts,
         scons, uitems, ustart, ucnt, p_seq, p_pos, prior1, prior2, out):
    K1 = len(prior1); K2 = len(prior2)
    for q in nb.prange(len(sel)):
        fi = sel[q]
        h = H[fi]; ho = hoff[fi]; u = fuser[fi]; t = fts[fi]
        h1 = np.zeros(K1, np.float64); h2 = np.zeros(K2, np.float64); tot = 0.0
        seen = np.empty(NH, np.int32); ns = 0
        for j in range(h):
            x = hitem[ho + j]
            dup = False
            for k in range(ns):
                if seen[k] == x:
                    dup = True
                    break
            if dup:
                continue
            seen[ns] = x; ns += 1
            w = 0.5 ** (hpos[ho + j] / 10.0)
            k = np.searchsorted(uitems, x)
            if k < len(uitems) and uitems[k] == x:
                pc = ucnt[k]
                ps = ustart[k] + max(0, pc - CAPI); pe = ustart[k] + pc
                for e in range(ps, pe):
                    s = p_seq[e]
                    if u != 0 and susers[s] == u:
                        continue
                    if scons[s] and sts[s] >= t:
                        continue
                    p = p_pos[e]; st = starts[s]; L = lens[s]
                    for qq in range(max(0, p - 3), min(L, p + 4)):
                        if qq == p or items[st + qq] == x:
                            continue
                        ww = w / abs(qq - p)
                        c1 = codes1[st + qq]
                        if c1 >= 0:
                            h1[c1] += ww; tot += ww
                            h2[codes2[st + qq]] += ww
            if ns >= NH:
                break
        co = coff[fi]
        for i in range(M[fi]):
            r = co + i
            b1 = cc1[r]; b2 = cc2[r]
            out[r, 3] = tot
            if b1 < 0 or tot == 0:
                out[r, 0] = np.nan; out[r, 1] = np.nan; out[r, 2] = np.nan
                continue
            out[r, 0] = h2[b2] / tot
            out[r, 1] = np.log((h2[b2] + 20.0 * prior2[b2]) / (tot + 20.0) / prior2[b2])
            out[r, 2] = np.log((h1[b1] + 20.0 * prior1[b1]) / (tot + 20.0) / prior1[b1])


def cand_codes(split, K):
    cl = np.load(CLU / f"item_cl_{K}_{split}.npy")
    e = np.asarray(arr(split, "cand_emb_idx"))
    return np.where(e >= 0, cl[np.maximum(e, 0)], -1).astype(np.int16)


def run_queries(idx, codes1, codes2, split, sel, out):
    f = feeds(split)
    p1 = np.bincount(codes1[codes1 >= 0], minlength=1024).astype(np.float64) + 1.0
    p2 = np.bincount(codes2[codes2 >= 0], minlength=4096).astype(np.float64) + 1.0
    _icv(np.asarray(sel, np.int64), f["cand_off"].to_numpy(), f["M"].to_numpy().astype(np.int64),
         f["hist_off"].to_numpy(), f["H"].to_numpy().astype(np.int64), f["user_id"].to_numpy().astype(np.int64),
         f["timestamp"].to_numpy().astype(np.int64), cand_codes(split, 1024), cand_codes(split, 4096),
         np.asarray(arr(split, "hist_item_id")), np.asarray(arr(split, "hist_hpos")),
         idx.items, codes1, codes2, idx.starts, idx.lens, idx.users, idx.ts, idx.cons,
         idx.uitems, idx.ustart, idx.ucnt, idx.p_seq, idx.p_pos, p1 / p1.sum(), p2 / p2.sum(), out)


if __name__ == "__main__":
    nb.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 24)
    sp = sys.argv[1].split(",")
    parts = [("train", source_fidx(None), False)] + [(s, np.arange(len(feeds(s))), True) for s in ("val", "test")]
    idx = Index(parts)
    c1, c2 = idx.cluster_codes(1024), idx.cluster_codes(4096)
    for split in sp:
        out = np.zeros((len(arr(split, "cand_fidx")), len(NAMES)), np.float32)
        run_queries(idx, c1, c2, split, np.arange(len(feeds(split))), out)
        log(split, "done")
        for i, nm in enumerate(NAMES):
            save(split, nm, out[:, i])
