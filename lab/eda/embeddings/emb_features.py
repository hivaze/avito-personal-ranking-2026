"""Embedding-based candidate features (per feed), numba-parallel.
Usage: F, names = compute_features(split, fidx=None)  -> F float32 [N_cand, K] in flat candidate order of the chosen feeds."""
import numpy as np
import numba as nb
from common import feeds, arr, item_table, gather_emb

NAMES = (['hmean_cos'] + [f'last{k}_cos' for k in (1, 5, 10, 20, 50)] +
         ['max_sim', 'top5_sim', 'smax_t0.05', 'smax_t0.1', 'smax_t0.2',
          'rec_hl5', 'rec_hl20', 'rec_hl50'] +
         [f'type{t}_mean' for t in range(1, 11)] + [f'type{t}_max' for t in range(1, 11)] +
         ['centrality_loo', 'n_sim_gt0.9', 'mean_sim', 'cand_mean_cos_raw', 'cand_cat_centrality'])
K = len(NAMES)


@nb.njit(parallel=True, fastmath=True, cache=True)
def _kernel(C, coff, M, Hm, hvalid, htype, hoff, H, ccat, out):
    nf = len(M)
    d = C.shape[1]
    for f in nb.prange(nf):
        m = M[f]; co = coff[f]
        c = C[co:co + m]
        # ---------- within-feed centrality (leave one out) ----------
        s = np.zeros(d, np.float32)
        for i in range(m):
            s += c[i]
        for i in range(m):
            loo = s - c[i]
            nl = np.sqrt((loo * loo).sum()) + 1e-9
            out[co + i, 34] = (c[i] * loo).sum() / nl
            out[co + i, 37] = (c[i] * s).sum() / (np.sqrt((s * s).sum()) + 1e-9)
        # same-category centrality (LOO within category in feed)
        for i in range(m):
            acc = np.zeros(d, np.float32); cnt = 0
            for j in range(m):
                if j != i and ccat[co + j] == ccat[co + i]:
                    acc += c[j]; cnt += 1
            if cnt > 0:
                out[co + i, 38] = (c[i] * acc).sum() / (np.sqrt((acc * acc).sum()) + 1e-9)
            else:
                out[co + i, 38] = np.nan
        # ---------- history ----------
        ho = hoff[f]; hn = H[f]
        nv = 0
        for j in range(hn):
            if hvalid[ho + j]:
                nv += 1
        if nv == 0:
            for i in range(m):
                for k in range(34):
                    out[co + i, k] = np.nan
                out[co + i, 35] = np.nan
                out[co + i, 36] = np.nan
            continue
        hv = np.empty((nv, d), np.float32)
        ht = np.empty(nv, np.int64)
        hp = np.empty(nv, np.float32)
        k = 0
        for j in range(hn):
            if hvalid[ho + j]:
                hv[k] = Hm[ho + j]; ht[k] = htype[ho + j]; hp[k] = j; k += 1
        S = c @ hv.T  # m x nv cosine (inputs normalized)
        # pooled vectors
        ks = (1, 5, 10, 20, 50)
        pooled = np.zeros((6, d), np.float32)
        for j in range(nv):
            pooled[0] += hv[j]
            for q in range(5):
                if j < ks[q]:
                    pooled[q + 1] += hv[j]
        for q in range(6):
            nq = np.sqrt((pooled[q] * pooled[q]).sum()) + 1e-9
            pooled[q] /= nq
        P = c @ pooled.T  # m x 6
        wr = np.empty((3, nv), np.float32)
        hls = (5.0, 20.0, 50.0)
        for q in range(3):
            for j in range(nv):
                wr[q, j] = np.exp(-hp[j] / hls[q])
        taus = (0.05, 0.1, 0.2)
        for i in range(m):
            o = co + i
            for q in range(6):
                out[o, q] = P[i, q]
            row = S[i]
            mx = -2.0
            for j in range(nv):
                if row[j] > mx:
                    mx = row[j]
            out[o, 6] = mx
            srt = np.sort(row)
            kk = min(5, nv)
            out[o, 7] = srt[nv - kk:].mean()
            for q in range(3):
                num = 0.0; den = 0.0
                for j in range(nv):
                    e = np.exp((row[j] - mx) / taus[q])
                    num += e * row[j]; den += e
                out[o, 8 + q] = num / den
            for q in range(3):
                num = 0.0; den = 0.0
                for j in range(nv):
                    num += wr[q, j] * row[j]; den += wr[q, j]
                out[o, 11 + q] = num / den
            tsum = np.zeros(11, np.float32); tcnt = np.zeros(11, np.int64); tmax = np.full(11, -2.0, np.float32)
            cgt = 0; ssum = 0.0
            for j in range(nv):
                t = ht[j]
                if t >= 1 and t <= 10:
                    tsum[t] += row[j]; tcnt[t] += 1
                    if row[j] > tmax[t]:
                        tmax[t] = row[j]
                if row[j] > 0.9:
                    cgt += 1
                ssum += row[j]
            for t in range(1, 11):
                if tcnt[t] > 0:
                    out[o, 13 + t] = tsum[t] / tcnt[t]
                    out[o, 23 + t] = tmax[t]
                else:
                    out[o, 13 + t] = np.nan
                    out[o, 23 + t] = np.nan
            out[o, 35] = cgt
            out[o, 36] = ssum / nv


def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(n, 1e-9)).astype(np.float32), n[:, 0]


def load_split(split, fidx=None):
    """Return dict with candidate/history arrays restricted to feeds fidx (sorted)."""
    f = feeds(split)
    if fidx is not None:
        f = f[np.sort(np.asarray(fidx))]
    M = f['M'].to_numpy().astype(np.int64); H = f['H'].to_numpy().astype(np.int64)
    co = f['cand_off'].to_numpy(); ho = f['hist_off'].to_numpy()
    if fidx is None:
        cidx = slice(None); hidx = slice(None)
    else:
        cidx = np.repeat(co - np.cumsum(np.r_[0, M[:-1]]), M) + np.arange(M.sum())
        hidx = np.repeat(ho - np.cumsum(np.r_[0, H[:-1]]), H) + np.arange(H.sum())
    D = {'feeds': f, 'M': M, 'H': H,
         'coff': np.r_[0, np.cumsum(M)[:-1]].astype(np.int64),
         'hoff': np.r_[0, np.cumsum(H)[:-1]].astype(np.int64)}
    for nm in ['item_id', 'category_id', 'f1', 'f2', 'f3', 'f4', 'f5', 'emb_idx']:
        D['c_' + nm] = np.asarray(arr(split, 'cand_' + nm)[cidx])
    for nm in ['target', 'pos']:
        try:
            D['c_' + nm] = np.asarray(arr(split, 'cand_' + nm)[cidx])
        except FileNotFoundError:
            pass
    for nm in ['item_id', 'category_id', 'event_type', 'event_source', 'emb_idx', 'hpos']:
        D['h_' + nm] = np.asarray(arr(split, 'hist_' + nm)[hidx])
    D['user'] = np.asarray(arr(split, 'user_emb')[f['fidx'].to_numpy()]).astype(np.float32)
    T = item_table(split)
    D['C_raw'] = gather_emb(T, D['c_emb_idx'])
    D['H_raw'] = gather_emb(T, D['h_emb_idx'])
    return D


def compute_features(D):
    Cn, cnorm = l2n(D['C_raw'])
    Hn, _ = l2n(D['H_raw'])
    out = np.zeros((len(Cn), K), np.float32)
    _kernel(Cn, D['coff'], D['M'], Hn, D['h_emb_idx'] >= 0, D['h_event_type'].astype(np.int64),
            D['hoff'], D['H'], D['c_category_id'].astype(np.int64), out)
    # user-item
    U = np.repeat(D['user'], D['M'], axis=0)
    dot = (U * D['C_raw']).sum(1)
    un = np.linalg.norm(U, axis=1)
    cos = dot / (un * cnorm + 1e-9)
    l2 = np.linalg.norm(U - D['C_raw'], axis=1)
    extra = {'user_dot': dot, 'user_cos': cos, 'item_norm': cnorm, 'neg_user_l2': -l2}
    names = list(NAMES) + list(extra)
    F = np.concatenate([out, np.stack([extra[k] for k in extra], 1).astype(np.float32)], 1)
    return F, names
