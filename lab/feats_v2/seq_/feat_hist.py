"""Per-feed history sequence features (no learning; depend only on the feed's own history + embeddings).

Group E (embedding x sequence), group C (category sequence structure).
Usage: python feat_hist.py <split>[,<split>] [threads]
"""
import sys
import numba as nb
import numpy as np
from seq_lib import *

DEEP = np.zeros(N_ET + 1, np.bool_)
for e in (5, 6, 8, 9):
    DEEP[e] = True

E_NAMES = ["seq_dcos_hl2", "seq_dcos_hl5", "seq_dcos_hl15", "seq_dcos_t1_hl5", "seq_dcos_nt_hl10",
           "seq_dcos_deep", "seq_dcos_rep", "seq_datt_b10", "seq_dmax_hl10", "seq_top3", "seq_argmax_hpos",
           "seq_nd90_cnt", "seq_nd80_dec", "seq_cos_last1", "seq_cos_last_t1", "seq_max_all", "seq_seg_cos",
           "seq_nd95_t1", "seq_max_t1", "seq_max_nt", "seq_max_old", "seq_dcos_s1", "seq_dcos_s2", "seq_max_s1",
           "seq_max2_gap", "seq_max_s2", "seq_max_s0", "seq_max_deep", "seq_max_rep", "seq_max_new8", "seq_max_t4",
           "seq_max_t23", "seq_dcos_s0"]


@nb.njit(parallel=True, fastmath=True, cache=True)
def _emb_kernel(cand_off, M, hist_off, H, cemb, hemb, hitem, het, hsrc, deep, table, out):
    n = len(M)
    D = table.shape[1]
    NO = out.shape[1]
    for f in nb.prange(n):
        co = cand_off[f]; m = M[f]; ho = hist_off[f]; h = H[f]
        # history vectors (normalized), keep only rows with embeddings
        V = np.zeros((max(h, 1), D), np.float32)
        hp = np.zeros(max(h, 1), np.int64)
        ty = np.zeros(max(h, 1), np.int64)
        sr = np.zeros(max(h, 1), np.int64)
        cntit = np.zeros(max(h, 1), np.float32)
        nv = 0
        for j in range(h):
            ei = hemb[ho + j]
            if ei < 0:
                continue
            s = 0.0
            for d in range(D):
                x = np.float32(table[ei, d])
                V[nv, d] = x
                s += x * x
            s = np.sqrt(s) + 1e-6
            for d in range(D):
                V[nv, d] /= s
            hp[nv] = j
            ty[nv] = het[ho + j]
            sr[nv] = hsrc[ho + j]
            # repeat count of this item in the history
            c = 0
            it = hitem[ho + j]
            for k in range(h):
                if hitem[ho + k] == it:
                    c += 1
            cntit[nv] = c
            nv += 1
        # pooled vectors: 0 hl2, 1 hl5, 2 hl15, 3 t1 hl5, 4 nt hl10, 5 deep hl30, 6 rep hl30, 7 seg
        P = np.zeros((11, D), np.float32)
        last1 = -1; last_t1 = -1
        for k in range(nv):
            j = hp[k]
            w2 = 0.5 ** (j / 2.0); w5 = 0.5 ** (j / 5.0); w15 = 0.5 ** (j / 15.0)
            w10 = 0.5 ** (j / 10.0); w30 = 0.5 ** (j / 30.0)
            for d in range(D):
                v = V[k, d]
                P[0, d] += w2 * v; P[1, d] += w5 * v; P[2, d] += w15 * v
            if ty[k] == 1:
                for d in range(D):
                    P[3, d] += w5 * V[k, d]
                if last_t1 < 0:
                    last_t1 = k
            else:
                for d in range(D):
                    P[4, d] += w10 * V[k, d]
            if deep[ty[k]]:
                for d in range(D):
                    P[5, d] += w30 * V[k, d]
            if cntit[k] >= 2:
                for d in range(D):
                    P[6, d] += w30 * cntit[k] * V[k, d]
            if last1 < 0:
                last1 = k
            if sr[k] == 1 and ty[k] == 1:
                for d in range(D):
                    P[8, d] += w5 * V[k, d]
            if sr[k] == 2:
                for d in range(D):
                    P[9, d] += w5 * V[k, d]
            if sr[k] == 0:
                for d in range(D):
                    P[10, d] += w5 * V[k, d]
        # current segment: newest events while cos to running centroid >= 0.5 (one miss tolerated)
        nseg = 0; miss = 0
        for k in range(nv):
            if nseg == 0:
                for d in range(D):
                    P[7, d] += V[k, d]
                nseg = 1
                continue
            cn = 0.0; dt = 0.0
            for d in range(D):
                cn += P[7, d] * P[7, d]; dt += P[7, d] * V[k, d]
            c = dt / (np.sqrt(cn) + 1e-6)
            if c >= 0.5:
                for d in range(D):
                    P[7, d] += V[k, d]
                nseg += 1; miss = 0
            else:
                miss += 1
                if miss >= 2:
                    break
        Pn = np.zeros(11, np.float32)
        for q in range(11):
            s = 0.0
            for d in range(D):
                s += P[q, d] * P[q, d]
            Pn[q] = np.sqrt(s)
        sims = np.zeros(max(nv, 1), np.float32)
        for i in range(m):
            r = co + i
            ei = cemb[r]
            if ei < 0 or nv == 0:
                for q in range(NO):
                    out[r, q] = np.nan
                if ei >= 0 and nv == 0:
                    out[r, 11] = 0.0; out[r, 12] = 0.0; out[r, 17] = 0.0
                continue
            c = np.zeros(D, np.float32)
            s = 0.0
            for d in range(D):
                x = np.float32(table[ei, d]); c[d] = x; s += x * x
            s = np.sqrt(s) + 1e-6
            for d in range(D):
                c[d] /= s
            for q in range(11):
                if Pn[q] > 0:
                    dt = 0.0
                    for d in range(D):
                        dt += c[d] * P[q, d]
                    val = dt / Pn[q]
                else:
                    val = np.nan
                if q < 7:
                    out[r, q] = val
                elif q == 7:
                    out[r, 16] = val
                elif q == 8:
                    out[r, 21] = val
                elif q == 9:
                    out[r, 22] = val
                else:
                    out[r, 32] = val
            # per-event sims
            mx = -2.0; amx = 0; att = 0.0; attw = 0.0; dmx = 0.0; nd90 = 0.0; nd80 = 0.0; nd95t1 = 0.0
            t1 = -2.0; t2 = -2.0; t3 = -2.0
            mt1 = -2.0; mnt = -2.0; mold = -2.0; ms1 = -2.0
            ms2 = -2.0; ms0 = -2.0; mdeep = -2.0; mrep = -2.0; mnew = -2.0; mt4 = -2.0; mt23 = -2.0
            for k in range(nv):
                dt = 0.0
                for d in range(D):
                    dt += c[d] * V[k, d]
                sims[k] = dt
                j = hp[k]
                if dt > mx:
                    mx = dt; amx = j
                w20 = 0.5 ** (j / 20.0)
                att += w20 * np.exp(10.0 * (dt - 1.0)); attw += w20
                w10 = 0.5 ** (j / 10.0)
                v = max(dt, 0.0) * w10
                if v > dmx:
                    dmx = v
                if dt >= 0.9:
                    nd90 += 1.0
                if dt >= 0.8:
                    nd80 += w20
                if dt >= 0.95 and ty[k] == 1:
                    nd95t1 += 1.0
                if ty[k] == 1:
                    if dt > mt1:
                        mt1 = dt
                    if sr[k] == 1 and dt > ms1:
                        ms1 = dt
                elif dt > mnt:
                    mnt = dt
                if j >= 32 and dt > mold:
                    mold = dt
                if sr[k] == 2 and dt > ms2:
                    ms2 = dt
                if sr[k] == 0 and dt > ms0:
                    ms0 = dt
                if deep[ty[k]] and dt > mdeep:
                    mdeep = dt
                if cntit[k] >= 2 and dt > mrep:
                    mrep = dt
                if j < 8 and dt > mnew:
                    mnew = dt
                if ty[k] == 4 and dt > mt4:
                    mt4 = dt
                if (ty[k] == 2 or ty[k] == 3) and dt > mt23:
                    mt23 = dt
                if dt > t1:
                    t3 = t2; t2 = t1; t1 = dt
                elif dt > t2:
                    t3 = t2; t2 = dt
                elif dt > t3:
                    t3 = dt
            out[r, 7] = 1.0 + np.log(att / attw + 1e-30) / 10.0
            out[r, 8] = dmx
            if nv >= 3:
                out[r, 9] = (t1 + t2 + t3) / 3.0
            elif nv == 2:
                out[r, 9] = (t1 + t2) / 2.0
            else:
                out[r, 9] = t1
            out[r, 10] = amx
            out[r, 11] = nd90
            out[r, 12] = nd80
            out[r, 13] = sims[last1]
            out[r, 14] = sims[last_t1] if last_t1 >= 0 else np.nan
            out[r, 15] = mx
            out[r, 17] = nd95t1
            out[r, 18] = mt1 if mt1 > -2.0 else np.nan
            out[r, 19] = mnt if mnt > -2.0 else np.nan
            out[r, 20] = mold if mold > -2.0 else np.nan
            out[r, 23] = ms1 if ms1 > -2.0 else np.nan
            out[r, 24] = t1 - t2 if nv >= 2 else np.nan
            out[r, 25] = ms2 if ms2 > -2.0 else np.nan
            out[r, 26] = ms0 if ms0 > -2.0 else np.nan
            out[r, 27] = mdeep if mdeep > -2.0 else np.nan
            out[r, 28] = mrep if mrep > -2.0 else np.nan
            out[r, 29] = mnew if mnew > -2.0 else np.nan
            out[r, 30] = mt4 if mt4 > -2.0 else np.nan
            out[r, 31] = mt23 if mt23 > -2.0 else np.nan


C_NAMES = ["seq_cdec_hl3", "seq_cdec_hl20", "seq_cdec_t1_hl10", "seq_cmom", "seq_c_runs", "seq_c_headrun",
           "seq_c_rank", "seq_c_nitems", "seq_c_rep_items", "seq_entropy", "seq_last_et", "seq_n_t1_10",
           "seq_c_min_hpos_t1", "seq_c_max_hpos", "seq_c_runs_share", "seq_c_deep_dec"]


@nb.njit(parallel=True, cache=True)
def _cat_kernel(cand_off, M, hist_off, H, ccat, hcat, het, hitem, deep, out):
    n = len(M)
    for f in nb.prange(n):
        co = cand_off[f]; m = M[f]; ho = hist_off[f]; h = H[f]
        d3 = np.zeros(N_CAT, np.float64); d20 = np.zeros(N_CAT, np.float64); dt1 = np.zeros(N_CAT, np.float64)
        ddeep = np.zeros(N_CAT, np.float64)
        s3 = 0.0; s20 = 0.0; st1 = 0.0; sdeep = 0.0
        c10 = np.zeros(N_CAT, np.float64); call = np.zeros(N_CAT, np.float64)
        runs = np.zeros(N_CAT, np.float64)
        minh1 = np.full(N_CAT, 999.0); maxh = np.full(N_CAT, -1.0)
        nitems = np.zeros(N_CAT, np.float64); repit = np.zeros(N_CAT, np.float64)
        nt1_10 = 0.0
        prev = -1
        for j in range(h):
            c = hcat[ho + j]
            e = het[ho + j]
            w3 = 0.5 ** (j / 3.0); w20 = 0.5 ** (j / 20.0)
            d3[c] += w3; s3 += w3; d20[c] += w20; s20 += w20
            if e == 1:
                w10 = 0.5 ** (j / 10.0)
                dt1[c] += w10; st1 += w10
                if minh1[c] > j:
                    minh1[c] = j
                if j < 10:
                    nt1_10 += 1
            if deep[e]:
                w30 = 0.5 ** (j / 30.0)
                ddeep[c] += w30; sdeep += w30
            if j < 10:
                c10[c] += 1
            call[c] += 1
            if c != prev:
                runs[c] += 1
            prev = c
            if j > maxh[c]:
                maxh[c] = j
            # first occurrence (newest) of this item -> distinct item; count repeats
            it = hitem[ho + j]
            first = True
            for k in range(j):
                if hitem[ho + k] == it:
                    first = False
                    break
            if first:
                nitems[c] += 1
                cnt = 0
                for k in range(j + 1, h):
                    if hitem[ho + k] == it:
                        cnt += 1
                if cnt >= 1:
                    repit[c] += 1
        # head run
        headc = hcat[ho] if h > 0 else -1
        headrun = 0
        for j in range(h):
            if hcat[ho + j] == headc:
                headrun += 1
            else:
                break
        # entropy and rank
        ent = 0.0
        for c in range(N_CAT):
            if call[c] > 0:
                p = call[c] / h
                ent -= p * np.log(p)
        rank = np.full(N_CAT, 99.0)
        for c in range(N_CAT):
            if call[c] > 0:
                rr = 0
                for c2 in range(N_CAT):
                    if call[c2] > call[c]:
                        rr += 1
                rank[c] = rr
        totruns = 0.0
        for c in range(N_CAT):
            totruns += runs[c]
        last_et = het[ho] if h > 0 else 0
        for i in range(m):
            r = co + i
            c = ccat[r]
            if h == 0:
                for q in range(out.shape[1]):
                    out[r, q] = np.nan
                continue
            out[r, 0] = d3[c] / s3
            out[r, 1] = d20[c] / s20
            out[r, 2] = dt1[c] / st1 if st1 > 0 else np.nan
            out[r, 3] = ((c10[c] + 0.5) / (min(h, 10) + 1.0)) / ((call[c] + 0.5) / (h + 1.0))
            out[r, 4] = runs[c]
            out[r, 5] = headrun if c == headc else 0.0
            out[r, 6] = rank[c]
            out[r, 7] = nitems[c]
            out[r, 8] = repit[c]
            out[r, 9] = ent
            out[r, 10] = last_et
            out[r, 11] = nt1_10
            out[r, 12] = minh1[c]
            out[r, 13] = maxh[c]
            out[r, 14] = runs[c] / totruns
            out[r, 15] = ddeep[c] / sdeep if sdeep > 0 else np.nan


def run(split, groups=("E", "C")):
    f = feeds(split)
    M = f["M"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
    co = f["cand_off"].to_numpy(); ho = f["hist_off"].to_numpy()
    N = int(M.sum())
    hitem = np.asarray(arr(split, "hist_item_id")); het = np.asarray(arr(split, "hist_event_type")).astype(np.int64)
    if "E" in groups:
        table = item_table(split)
        out = np.empty((N, len(E_NAMES)), np.float32)
        log(split, "emb kernel start")
        _emb_kernel(co, M, ho, H, np.asarray(arr(split, "cand_emb_idx")), np.asarray(arr(split, "hist_emb_idx")),
                    hitem, het, np.asarray(arr(split, "hist_event_source")).astype(np.int64), DEEP, table, out)
        log(split, "emb kernel done")
        for i, nm in enumerate(E_NAMES):
            save(split, nm, out[:, i])
        del out
    if "C" in groups:
        out = np.empty((N, len(C_NAMES)), np.float32)
        _cat_kernel(co, M, ho, H, np.asarray(arr(split, "cand_category_id")).astype(np.int64),
                    np.asarray(arr(split, "hist_category_id")).astype(np.int64), het, hitem, DEEP, out)
        log(split, "cat kernel done")
        for i, nm in enumerate(C_NAMES):
            save(split, nm, out[:, i])


if __name__ == "__main__":
    nb.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 24)
    groups = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("E", "C")
    for s in sys.argv[1].split(","):
        run(s, groups)
