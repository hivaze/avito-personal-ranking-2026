"""e05b: per-history-item 'pull' by event type, controlled for recency.
For every candidate c, find its nearest history item i* (max cosine over the feed's distinct history items with an
embedding).  If cos >= 0.7, attribute c to i*.  Per group of i* (type set) x recency (newest hpos of i*):
O/E = observed positives / expected positives (sum of the feed's base rate over attributed candidates).
Also prints the full Mantel-Haenszel table of e05 (mc_t>=0.8 / 0.9, same-category) which e05 truncated.
Splits: val; train exploration sample (same 40K feeds as e05)."""
import sys, os
import numpy as np, polars as pl, numba as nb
sys.path.insert(0, os.path.dirname(__file__))
from e05_intent import build, mh_or

nb.set_num_threads(12)
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(-1); pl.Config.set_float_precision(4)
ROOT = '<repo>/data/flat'; COMP = '<repo>/data/competition'
OUT = '<repo>/nn_x/codes/events/out'
T = list(range(1, 11))


@nb.njit(parallel=True, cache=True)
def nearest_kernel(coff, M, hoff, H, c_emb, h_item, h_emb, h_type, h_hpos, E, out_mask, out_hpos, out_cos):
    for f in nb.prange(len(coff)):
        c0 = coff[f]; m = M[f]; h0 = hoff[f]; h = H[f]
        items = np.empty(h, np.int64); masks = np.zeros(h, np.int64); embs = np.empty(h, np.int64); newest = np.empty(h, np.int64); nu = 0
        for j in range(h):
            it = h_item[h0 + j]; found = -1
            for u in range(nu):
                if items[u] == it:
                    found = u; break
            if found < 0:
                items[nu] = it; embs[nu] = h_emb[h0 + j]; masks[nu] = 0; newest[nu] = h_hpos[h0 + j]; found = nu; nu += 1
            masks[found] |= (1 << h_type[h0 + j])
            if h_hpos[h0 + j] < newest[found]:
                newest[found] = h_hpos[h0 + j]
        for i in range(m):
            r = c0 + i; out_cos[r] = -2.0; out_mask[r] = 0; out_hpos[r] = -1
            ce = c_emb[r]
            if ce < 0:
                continue
            for u in range(nu):
                he = embs[u]
                if he < 0:
                    continue
                s = 0.0
                for d in range(E.shape[1]):
                    s += E[ce, d] * E[he, d]
                if s > out_cos[r]:
                    out_cos[r] = s; out_mask[r] = masks[u]; out_hpos[r] = newest[u]


def prep(split, fidx=None):
    F = pl.read_parquet(f'{ROOT}/{split}/feeds.parquet').sort('fidx')
    if fidx is not None:
        F = F.filter(pl.col('fidx').is_in(fidx)).sort('fidx')
    coff_o = F['cand_off'].to_numpy(); M = F['M'].to_numpy().astype(np.int64)
    hoff_o = F['hist_off'].to_numpy(); H = F['H'].to_numpy().astype(np.int64)
    crow = np.concatenate([np.arange(o, o + m) for o, m in zip(coff_o, M)])
    hrow = np.concatenate([np.arange(o, o + h) for o, h in zip(hoff_o, H)])
    ld = lambda n, rows: np.asarray(np.load(f'{ROOT}/{split}/{n}.npy', mmap_mode='r')[rows])
    c_emb = ld('cand_emb_idx', crow).astype(np.int64); tgt = ld('cand_target', crow).astype(np.int8)
    h_item = ld('hist_item_id', hrow); h_emb = ld('hist_emb_idx', hrow).astype(np.int64)
    h_type = ld('hist_event_type', hrow).astype(np.int64); h_hpos = ld('hist_hpos', hrow).astype(np.int64)
    mm = np.memmap(f'{COMP}/{split}/item_embeddings/data.memmap', dtype=np.int8, mode='r').reshape(-1, 128)
    need = np.unique(np.concatenate([c_emb[c_emb >= 0], h_emb[h_emb >= 0]]))
    E = np.asarray(mm[need]).astype(np.float32); E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)
    remap = lambda x: np.where(x >= 0, np.searchsorted(need, np.maximum(x, 0)), -1)
    coff = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64); hoff = np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64)
    N = len(crow)
    om = np.zeros(N, np.int64); oh = np.zeros(N, np.int64); oc = np.zeros(N, np.float32)
    nearest_kernel(coff, M, hoff, H, remap(c_emb), h_item, remap(h_emb), h_type, h_hpos, E, om, oh, oc)
    fid = np.repeat(np.arange(len(M)), M)
    base = (np.bincount(fid, weights=tgt, minlength=len(M)) / M)[fid]
    return dict(tgt=tgt, mask=om, hpos=oh, cos=oc, base=base)


def group_of(mask):
    has = lambda t: ((mask >> t) & 1).astype(bool)
    g = np.full(len(mask), 'other', dtype=object)
    only = lambda t: mask == (1 << t)
    g[only(1)] = '1 only'
    g[has(1) & has(4) & ~has(2) & ~has(3) & ~has(5) & ~has(6) & ~has(7) & ~has(8) & ~has(9)] = '1+4'
    g[only(4)] = '4 only'
    g[(has(2) | has(3)) & ~has(5) & ~has(6) & ~has(8) & ~has(7) & ~has(9) & has(1)] = '2/3 (+1)'
    g[(has(2) | has(3)) & ~has(5) & ~has(6) & ~has(8) & ~has(7) & ~has(9) & ~has(1)] = '2/3 (no 1)'
    g[(has(5) | has(6) | has(8))] = 'has 5/6/8'
    g[(has(7) | has(9)) & ~(has(5) | has(6) | has(8))] = 'has 7/9'
    g[has(10)] = 'has 10'
    return g


def attrib(name, d):
    sel = d['cos'] >= 0.7
    g = group_of(d['mask'][sel])
    rb = np.digitize(d['hpos'][sel], [16, 64])
    df = pl.DataFrame({'g': g.astype(str), 'rec': rb, 'y': d['tgt'][sel], 'e': d['base'][sel], 'cos': d['cos'][sel]})
    print(f'\n[N1] {name}: candidates with nearest history item at cos>=0.7: {sel.sum():,} of {len(sel):,}; O/E of positives by nearest item group x recency (0: newest hpos<16, 1: 16-63, 2: >=64)')
    t = df.group_by(['g', 'rec']).agg(pl.len().alias('n'), pl.col('y').sum().alias('obs'), pl.col('e').sum().alias('exp'), pl.col('cos').mean().alias('cos_mean'))
    t = t.with_columns((pl.col('obs') / pl.col('exp')).alias('O/E'), (pl.col('obs').sqrt() / pl.col('exp')).alias('se'))
    p = t.pivot(on='rec', index='g', values=['O/E', 'n']).sort('g')
    print(p)
    t.write_csv(f'{OUT}/e05b_attrib_{name}.csv')
    print('   all groups pooled by recency:')
    print(df.group_by('rec').agg(pl.len(), (pl.col('y').sum() / pl.col('e').sum()).alias('O/E')).sort('rec'))
    # cosine-matched: within cos band 0.8-0.9 only
    df2 = df.filter((pl.col('cos') >= 0.8) & (pl.col('cos') < 0.9))
    print('   cos band [0.8, 0.9) only, pooled recency:')
    print(df2.group_by('g').agg(pl.len(), (pl.col('y').sum() / pl.col('e').sum()).alias('O/E')).sort('g'))


def mh_table(name, d):
    tgt = d['tgt']; coff = d['coff']; M = d['M']; mc = d['mc']; sc = d['sc']; hasT = d['hasT']
    fid = np.repeat(np.arange(len(M)), M)
    rows = []
    for t in T:
        hf = hasT[fid, t - 1]
        o1 = mh_or((mc[:, t - 1] >= 0.8) & hf, tgt, coff, M)
        o2 = mh_or((mc[:, t - 1] >= 0.9) & hf, tgt, coff, M)
        o3 = mh_or((sc[:, t - 1] > 0) & hf, tgt, coff, M)
        # 'exclusive' flag: near (>=0.8) a type-t item but NOT near (>=0.8) any type-1 item
        ex = (mc[:, t - 1] >= 0.8) & (mc[:, 0] < 0.8) & hf
        o4 = mh_or(ex, tgt, coff, M) if t != 1 else (np.nan, 0, np.nan, np.nan)
        rows.append({'type': t, 'n_cos>=.8': o1[1], 'MH_OR_cos>=.8': o1[0], 'n_cos>=.9': o2[1], 'MH_OR_cos>=.9': o2[0],
                     'n_samecat': o3[1], 'MH_OR_samecat': o3[0], 'n_near_t_not_near_1': o4[1], 'MH_OR_near_t_not_1': o4[0]})
    print(f'\n[M1] {name}: within-feed Mantel-Haenszel odds ratios (feeds having type t)')
    out = pl.DataFrame(rows); print(out); out.write_csv(f'{OUT}/e05b_mh_{name}.csv')


def main():
    z = np.load('<repo>/eda/labels_position/out/train_prandom.npz')
    samp = np.load(f'{OUT}/e05_train_explore40k_fidx.npy')
    for name, split, fx in [('val', 'val', None), ('train_explore40k', 'train', samp)]:
        d = build(split, fx)
        mh_table(name, d)
        a = prep(split, fx)
        attrib(name, a)


if __name__ == '__main__':
    main()
