"""Numba kernels: per-candidate history-affinity features (category-level) computed from a feed's own history."""
import numpy as np
import numba as nb

NB_CAT = 54
FEATS = ['H', 'cat_cnt', 'cat_frac', 'cat_rec95', 'cat_rec80', 'cat_minpos', 'cat_nitems',
         'cat_cnt_t1', 'cat_cnt_t2', 'cat_cnt_t3', 'cat_cnt_t4', 'cat_cnt_t5', 'cat_cnt_t6', 'cat_cnt_t7',
         'cat_cnt_t8', 'cat_cnt_t9', 'cat_cnt_t10', 'cat_cnt_feedclick', 'cat_rec_feedclick', 'cat_frac20',
         'cat_cnt_src0', 'cat_cnt_t1_srcother', 'n_feedclick', 'cat_frac_feedclick', 'cat_rank']


@nb.njit(parallel=True, cache=True)
def aff_kernel(sel, coff, M, hoff, H, ccat, hcat, htype, hsrc, hpos, hitem, out_off, out):
    nf = len(sel)
    for q in nb.prange(nf):
        fi = sel[q]
        cnt = np.zeros(NB_CAT); rec95 = np.zeros(NB_CAT); rec80 = np.zeros(NB_CAT)
        minpos = np.full(NB_CAT, 256.0); ct = np.zeros((11, NB_CAT)); fc = np.zeros(NB_CAT); fcr = np.zeros(NB_CAT)
        c20 = np.zeros(NB_CAT); src0 = np.zeros(NB_CAT); t1o = np.zeros(NB_CAT)
        ho = hoff[fi]; h = H[fi]
        nfc = 0.0
        for j in range(h):
            c = hcat[ho + j]; t = htype[ho + j]; s = hsrc[ho + j]; p = hpos[ho + j]
            cnt[c] += 1; rec95[c] += 0.95 ** p; rec80[c] += 0.8 ** p
            if p < minpos[c]:
                minpos[c] = p
            ct[t, c] += 1
            if t == 1 and s == 1:
                fc[c] += 1; fcr[c] += 0.95 ** p; nfc += 1
            elif t == 1:
                t1o[c] += 1
            if s == 0:
                src0[c] += 1
            if p < 20:
                c20[c] += 1
        # distinct items per category (O(h^2) small)
        nit = np.zeros(NB_CAT)
        for j in range(h):
            dup = False
            for k in range(j):
                if hitem[ho + k] == hitem[ho + j]:
                    dup = True
                    break
            if not dup:
                nit[hcat[ho + j]] += 1
        # category rank by count (1 = most frequent)
        rk = np.zeros(NB_CAT)
        for c in range(NB_CAT):
            r = 1.0
            for c2 in range(NB_CAT):
                if cnt[c2] > cnt[c]:
                    r += 1
            rk[c] = r if cnt[c] > 0 else 99.0
        co = coff[fi]; m = M[fi]; oo = out_off[q]
        hh = max(h, 1)
        for i in range(m):
            c = ccat[co + i]
            r = oo + i
            out[r, 0] = h; out[r, 1] = cnt[c]; out[r, 2] = cnt[c] / hh; out[r, 3] = rec95[c]; out[r, 4] = rec80[c]
            out[r, 5] = minpos[c]; out[r, 6] = nit[c]
            for t in range(1, 11):
                out[r, 6 + t] = ct[t, c]
            out[r, 17] = fc[c]; out[r, 18] = fcr[c]; out[r, 19] = c20[c] / min(hh, 20)
            out[r, 20] = src0[c]; out[r, 21] = t1o[c]; out[r, 22] = nfc; out[r, 23] = fc[c] / max(nfc, 1.0)
            out[r, 24] = rk[c]


def compute_aff(split, sel=None):
    """Returns (feature matrix float32 [n_cands_of_sel, len(FEATS)], sel fidx, cand index array)."""
    from common import feeds, arr
    f = feeds(split).sort('fidx')
    coff = f['cand_off'].to_numpy(); M = f['M'].to_numpy(); hoff = f['hist_off'].to_numpy(); H = f['H'].to_numpy()
    if sel is None:
        sel = np.arange(len(f))
    sel = np.asarray(sel, np.int64)
    m = M[sel]
    out_off = np.concatenate([[0], np.cumsum(m)[:-1]]).astype(np.int64)
    out = np.zeros((m.sum(), len(FEATS)), np.float32)
    ccat = np.asarray(arr(split, 'cand_category_id')); hcat = np.asarray(arr(split, 'hist_category_id'))
    htype = np.asarray(arr(split, 'hist_event_type')); hsrc = np.asarray(arr(split, 'hist_event_source'))
    hpos = np.asarray(arr(split, 'hist_hpos')); hitem = np.asarray(arr(split, 'hist_item_id'))
    aff_kernel(sel, coff, M, hoff, H, ccat, hcat, htype, hsrc, hpos, hitem, out_off, out)
    cidx = np.concatenate([np.arange(coff[i], coff[i] + M[i]) for i in sel]) if len(sel) < len(f) else np.arange(M.sum())
    return out, sel, cidx
