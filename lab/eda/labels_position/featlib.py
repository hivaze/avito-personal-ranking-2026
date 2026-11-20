"""Candidate feature builder for a set of feeds (used by detection + LightGBM experiments)."""
from common import *
from common import _ranges
import json

EMB = f'{ROOT}/data/competition'

def item_emb(split):
    meta = json.load(open(f'{EMB}/{split}/item_embeddings/data.meta'))
    return np.memmap(f'{EMB}/{split}/item_embeddings/data.memmap', dtype=np.int8, mode='r', shape=tuple(meta['shape']))

@nb.njit(parallel=True, cache=True)
def dot_feats(U, fid_of_cand, E, eidx):
    n = len(eidx)
    dot = np.zeros(n, np.float32); cos = np.zeros(n, np.float32); inorm = np.zeros(n, np.float32)
    for i in nb.prange(n):
        e = eidx[i]
        if e < 0:
            dot[i] = np.nan; cos[i] = np.nan; inorm[i] = np.nan
            continue
        u = U[fid_of_cand[i]]
        s = 0.0; nu = 0.0; ne = 0.0
        for j in range(128):
            a = float(u[j]); b = float(E[e, j])
            s += a * b; nu += a * a; ne += b * b
        dot[i] = s
        inorm[i] = np.sqrt(ne)
        cos[i] = s / np.sqrt(nu * ne) if nu > 0 and ne > 0 else 0.0
    return dot, cos, inorm

@nb.njit(parallel=True, cache=True)
def hist_cat_feats(hoff, H, hcat, hev, coff, M, ccat):
    """per candidate: # history events with same category (all), share, # with same cat in last 20 events, # same cat with event_type!=1"""
    nf = len(M); n = coff[-1] + M[-1] if nf > 0 else 0
    cnt = np.zeros(n, np.float32); share = np.zeros(n, np.float32); recent = np.zeros(n, np.float32); cnt_ne1 = np.zeros(n, np.float32)
    for f in nb.prange(nf):
        c = np.zeros(64, np.int32); r = np.zeros(64, np.int32); c2 = np.zeros(64, np.int32)
        ho = hoff[f]
        for j in range(H[f]):
            k = hcat[ho + j]
            c[k] += 1
            if j < 20:
                r[k] += 1
            if hev[ho + j] != 1:
                c2[k] += 1
        for j in range(M[f]):
            k = ccat[coff[f] + j]
            cnt[coff[f] + j] = c[k]
            share[coff[f] + j] = c[k] / max(H[f], 1)
            recent[coff[f] + j] = r[k]
            cnt_ne1[coff[f] + j] = c2[k]
    return cnt, share, recent, cnt_ne1

def build(split, sel=None, with_emb=True, with_hist=True):
    """Returns polars DataFrame with candidate features for feeds sel (None = all), plus offsets/M."""
    F = feeds(split)
    if sel is None:
        sel = np.arange(len(F))
    names = ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id', 'emb_idx', 'item_id'] + (['target'] if split != 'test' else []) + (['pos'] if split == 'train' else [])
    d, off, M = gather(split, sel, F, names)
    fid = np.repeat(np.arange(len(sel)), M)
    out = {k: d[k] for k in names if k not in ('emb_idx',)}
    out['fid'] = fid.astype(np.int32)
    Hs = F['H'].to_numpy()[sel]
    out['M'] = np.repeat(M, M).astype(np.int32); out['H'] = np.repeat(Hs, M).astype(np.int32)
    if with_emb:
        U = np.load(f'{FLAT}/{split}/user_emb.npy', mmap_mode='r')
        Us = np.asarray(U[sel])
        E = item_emb(split)
        dot, cos, inorm = dot_feats(Us, fid, E, d['emb_idx'].astype(np.int64))
        out['dot'] = dot; out['cos'] = cos; out['inorm'] = inorm
    if with_hist:
        hoff_all = F['hist_off'].to_numpy()[sel]
        hcat_all = np.load(f'{FLAT}/{split}/hist_category_id.npy', mmap_mode='r')
        hev_all = np.load(f'{FLAT}/{split}/hist_event_type.npy', mmap_mode='r')
        idx = _ranges(hoff_all, Hs)
        hcat = np.asarray(hcat_all[idx]).astype(np.int64); hev = np.asarray(hev_all[idx]).astype(np.int64)
        hoff = np.concatenate([[0], np.cumsum(Hs)[:-1]]).astype(np.int64)
        cnt, share, recent, cnt2 = hist_cat_feats(hoff, Hs.astype(np.int64), hcat, hev, off.astype(np.int64), M.astype(np.int64), d['category_id'].astype(np.int64))
        out['hcat_cnt'] = cnt; out['hcat_share'] = share; out['hcat_recent'] = recent; out['hcat_cnt_ne1'] = cnt2
    df = pl.DataFrame(out)
    # within-feed rank-normalised features (percentile of f within its feed)
    df = df.with_columns([(pl.col(k).rank('average').over('fid') / pl.col('M')).alias(f'{k}_r') for k in ['f1', 'f3', 'f4', 'f5', 'dot']])
    return df, off, M, sel


@nb.njit(parallel=True, cache=True)
def hist_sim_feats(coff, M, ceidx, hoff, H, heidx, hev, E, K):
    """per candidate: max and mean cosine to the K most recent history items with embeddings; max cos over 'contact-like' events (event_type!=1)."""
    n = coff[-1] + M[-1]
    mx = np.full(n, np.nan, np.float32); mean = np.full(n, np.nan, np.float32); mx2 = np.full(n, np.nan, np.float32)
    for f in nb.prange(len(M)):
        # collect history vectors (normalised)
        hv = np.zeros((K, 128), np.float32); ht = np.zeros(K, np.int64); nh = 0
        for j in range(H[f]):
            e = heidx[hoff[f] + j]
            if e < 0:
                continue
            nn = 0.0
            for d in range(128):
                v = float(E[e, d]); hv[nh, d] = v; nn += v * v
            if nn <= 0:
                continue
            nn = np.sqrt(nn)
            for d in range(128):
                hv[nh, d] /= nn
            ht[nh] = hev[hoff[f] + j]
            nh += 1
            if nh >= K:
                break
        if nh == 0:
            continue
        for i in range(M[f]):
            e = ceidx[coff[f] + i]
            if e < 0:
                continue
            cv = np.empty(128, np.float32); nn = 0.0
            for d in range(128):
                v = float(E[e, d]); cv[d] = v; nn += v * v
            if nn <= 0:
                continue
            nn = np.sqrt(nn)
            best = -2.0; s = 0.0; best2 = -2.0
            for h in range(nh):
                c = 0.0
                for d in range(128):
                    c += hv[h, d] * cv[d]
                c /= nn
                s += c
                if c > best:
                    best = c
                if ht[h] != 1 and c > best2:
                    best2 = c
            mx[coff[f] + i] = best; mean[coff[f] + i] = s / nh
            if best2 > -2.0:
                mx2[coff[f] + i] = best2
    return mx, mean, mx2


def add_hist_sim(df, split, sel, off, M, K=32):
    F = feeds(split)
    Hs = F['H'].to_numpy()[sel]; hoff_all = F['hist_off'].to_numpy()[sel]
    idx = _ranges(hoff_all, Hs)
    he = np.asarray(np.load(f'{FLAT}/{split}/hist_emb_idx.npy', mmap_mode='r')[idx]).astype(np.int64)
    hev = np.asarray(np.load(f'{FLAT}/{split}/hist_event_type.npy', mmap_mode='r')[idx]).astype(np.int64)
    hoff = np.concatenate([[0], np.cumsum(Hs)[:-1]]).astype(np.int64)
    ce = np.asarray(cand(split, 'emb_idx'))[_ranges(F['cand_off'].to_numpy()[sel], M)].astype(np.int64)
    E = item_emb(split)
    mx, mean, mx2 = hist_sim_feats(off.astype(np.int64), M.astype(np.int64), ce, hoff, Hs.astype(np.int64), he, hev, E, K)
    return df.with_columns(pl.Series('hsim_max', mx), pl.Series('hsim_mean', mean), pl.Series('hsim_max_ne1', mx2))
