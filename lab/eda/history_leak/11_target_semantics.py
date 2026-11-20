"""11) Which history events correspond to the target action?
Consecutive train feed pairs (A -> B) with perfect/near alignment (match>=0.9, so the 'new part' of B = first k events):
  - for A positives / negatives: event types/sources present in B's new part on that item
  - for B's new events: fraction that are A candidates / A positives, by (type, source)."""
import numpy as np
import polars as pl
import numba as nb
from common import feeds, arr

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
nb.set_num_threads(40)


@nb.njit(parallel=True, cache=True)
def kern(a_idx, b_idx, ks, hoff, coff, M, hi, ht, hs, ci, ct, ev_start, ev_type, ev_src, ev_isc, ev_ispos,
         c_start, c_t, c_has1src1, c_has1, c_hasany, c_ntypes):
    for p in nb.prange(len(a_idx)):
        a = a_idx[p]; b = b_idx[p]; k = ks[p]
        hb = hoff[b]; ca = coff[a]; ma = M[a]
        es = ev_start[p]
        for j in range(k):
            it = hi[hb + j]
            ev_type[es + j] = ht[hb + j]; ev_src[es + j] = hs[hb + j]
            ev_isc[es + j] = 0; ev_ispos[es + j] = 0
            for i in range(ma):
                if ci[ca + i] == it:
                    ev_isc[es + j] = 1
                    if ct[ca + i] == 1:
                        ev_ispos[es + j] = 1
                    break
        cs = c_start[p]
        for i in range(ma):
            it = ci[ca + i]
            c_t[cs + i] = ct[ca + i]
            h11 = 0; h1 = 0; ha = 0; mask = 0
            for j in range(k):
                if hi[hb + j] == it:
                    ha = 1
                    mask |= 1 << ht[hb + j]
                    if ht[hb + j] == 1:
                        h1 = 1
                        if hs[hb + j] == 1:
                            h11 = 1
            c_has1src1[cs + i] = h11; c_has1[cs + i] = h1; c_hasany[cs + i] = ha; c_ntypes[cs + i] = mask


def main():
    z = np.load('<repo>/eda/history_leak/out_pairs_train.npz')
    ok = (z['match'] >= 0.9) & (z['k'] > 0) & (z['gap'] <= 86400)
    a_idx = z['a'][ok]; b_idx = z['b'][ok]; ks = z['k'][ok].astype(np.int64); gap = z['gap'][ok]
    print('pairs used (match>=0.9, k>0, gap<=1d):', len(a_idx))
    F = feeds('train').sort('fidx')
    hoff = F['hist_off'].to_numpy(); coff = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
    hi = np.asarray(arr('train', 'hist_item_id')); ht = np.asarray(arr('train', 'hist_event_type'))
    hs = np.asarray(arr('train', 'hist_event_source'))
    ci = np.asarray(arr('train', 'cand_item_id')); ct = np.asarray(arr('train', 'cand_target'))
    ev_start = np.concatenate([[0], np.cumsum(ks)[:-1]]).astype(np.int64); NE = ks.sum()
    ma = M[a_idx]; c_start = np.concatenate([[0], np.cumsum(ma)[:-1]]).astype(np.int64); NC = ma.sum()
    ev_type = np.zeros(NE, np.int8); ev_src = np.zeros(NE, np.int8); ev_isc = np.zeros(NE, np.int8); ev_ispos = np.zeros(NE, np.int8)
    c_t = np.zeros(NC, np.int8); c11 = np.zeros(NC, np.int8); c1 = np.zeros(NC, np.int8); ca = np.zeros(NC, np.int8)
    cm = np.zeros(NC, np.int32)
    kern(a_idx, b_idx, ks, hoff, coff, M, hi, ht, hs, ci, ct, ev_start, ev_type, ev_src, ev_isc, ev_ispos,
         c_start, c_t, c11, c1, ca, cm)
    E = pl.DataFrame({'type': ev_type, 'src': ev_src, 'is_cand': ev_isc, 'is_pos': ev_ispos})
    print('\nB new-part events: share that are A candidates / A positives, by (type, src) [top 15 by count]')
    print(E.with_columns(pl.when(pl.col('src') <= 2).then(pl.col('src')).otherwise(99).alias('src')).group_by(['type', 'src']).agg(
        pl.len(), pl.col('is_cand').mean().alias('A_cand_share'), pl.col('is_pos').mean().alias('A_pos_share')).sort('len', descending=True).head(15))
    C = pl.DataFrame({'t': c_t, 'has_t1s1': c11, 'has_t1': c1, 'has_any': ca, 'mask': cm})
    print('\nA candidates: presence in B new part')
    print(C.group_by('t').agg(pl.len(), pl.col('has_any').mean(), pl.col('has_t1').mean(), pl.col('has_t1s1').mean()).sort('t'))
    print('P(target | has type1 src1 in B new part) =', C.filter(pl.col('has_t1s1') == 1)['t'].mean(),
          '; P(target | in new part but no t1s1) =', C.filter((pl.col('has_any') == 1) & (pl.col('has_t1s1') == 0))['t'].mean())
    # event types present on A positives (bitmask) found in B
    pos = C.filter((pl.col('t') == 1) & (pl.col('has_any') == 1))
    for ty in range(1, 11):
        print(f'  A positives found in B new part: share having type {ty}: {((pos["mask"].to_numpy() >> ty) & 1).mean():.3f}')
    m = pos['mask'].to_numpy()
    only = {ty: ((m == (1 << ty))).mean() for ty in range(1, 11)}
    print('  A positives found: share whose ONLY event type in B new part is ty:', {k: round(v, 3) for k, v in only.items()})
    no1 = m[(m & 2) == 0]
    print('  A positives found without any type-1 event: n=%d; type shares among them:' % len(no1),
          {ty: round(((no1 >> ty) & 1).mean(), 3) for ty in range(2, 11)})
    neg = C.filter((pl.col('t') == 0) & (pl.col('has_any') == 1))
    print(f'  A negatives found in B new part: n={len(neg)}, share having type1: {((neg["mask"].to_numpy() >> 1) & 1).mean():.3f}')


if __name__ == '__main__':
    main()
