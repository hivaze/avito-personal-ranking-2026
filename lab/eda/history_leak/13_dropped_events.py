"""13) Which of A's history events disappear from the later feed B's history (beyond simple truncation at 256)?
Supports the hypothesis of per-event-type retention windows."""
import time
import numpy as np
import polars as pl
import numba as nb
from common import feeds, arr
nb.set_num_threads(16)


@nb.njit(parallel=True, cache=True)
def kern(a_idx, b_idx, ks, hoff, H, hi, ht, out_start, o_type, o_in, o_pos):
    for p in nb.prange(len(a_idx)):
        a = a_idx[p]; b = b_idx[p]; k = ks[p]
        la = H[a]; lb = H[b]
        keep_n = max(0, min(la, 256 - k))
        s = out_start[p]
        for j in range(keep_n):
            it = hi[hoff[a] + j]; ty = ht[hoff[a] + j]
            f = 0
            for q in range(lb):
                if hi[hoff[b] + q] == it and ht[hoff[b] + q] == ty:
                    f = 1
                    break
            o_type[s + j] = ty; o_in[s + j] = f; o_pos[s + j] = j


def main():
    z = np.load('<repo>/eda/history_leak/out_pairs_train.npz')
    ok = np.where(z['k'] >= 0)[0]
    ok = np.sort(np.random.default_rng(0).choice(ok, 100000, replace=False))
    F = feeds('train').sort('fidx'); hoff = F['hist_off'].to_numpy(); H = F['H'].to_numpy().astype(np.int64)
    hi = np.asarray(arr('train', 'hist_item_id')); ht = np.asarray(arr('train', 'hist_event_type'))
    a_idx = z['a'][ok].astype(np.int64); b_idx = z['b'][ok].astype(np.int64); ks = z['k'][ok].astype(np.int64); gap = z['gap'][ok]
    keep = np.maximum(0, np.minimum(H[a_idx], 256 - ks))
    out_start = np.concatenate([[0], np.cumsum(keep)[:-1]]).astype(np.int64)
    N = int(keep.sum())
    o_type = np.zeros(N, np.int8); o_in = np.zeros(N, np.int8); o_pos = np.zeros(N, np.int16)
    t0 = time.time()
    kern(a_idx, b_idx, ks, hoff, H, hi, ht, out_start, o_type, o_in, o_pos)
    print('kernel', N, 'events', round(time.time() - t0, 1), 's', flush=True)
    D = pl.DataFrame({'type': o_type, 'inB': o_in, 'hposA': o_pos, 'gap': np.repeat(gap, keep),
                      'Bfull': np.repeat(H[b_idx] == 256, keep)})
    print('A events expected to survive in B (fit in window):', len(D), ' dropped share:', 1 - D['inB'].mean())
    print(D.group_by('type').agg(pl.len(), (1 - pl.col('inB').mean()).alias('dropped_share'),
                                 (1 - pl.col('inB').filter(~pl.col('Bfull')).mean()).alias('dropped_share_B_not_full')).sort('type'))
    D = D.with_columns(pl.col('gap').cut([86400, 3 * 86400, 7 * 86400]).alias('gb'))
    print(D.group_by(['gb']).agg(pl.len(), (1 - pl.col('inB').filter(pl.col('type') == 1).mean()).alias('type1_dropped'),
                                 (1 - pl.col('inB').filter(pl.col('type') == 2).mean()).alias('type2_dropped'),
                                 (1 - pl.col('inB').filter(pl.col('type') == 3).mean()).alias('type3_dropped'),
                                 (1 - pl.col('inB').filter(pl.col('type') == 5).mean()).alias('type5_dropped')).sort('gb'))
    D = D.with_columns((pl.col('hposA') // 32).alias('hb'))
    print('type-1 drop share by hpos in A (32-wide buckets), B full vs not:')
    print(D.filter(pl.col('type') == 1).group_by(['hb']).agg(pl.len(), (1 - pl.col('inB').mean()).alias('drop')).sort('hb'))


if __name__ == '__main__':
    main()
