"""e04: flow vs stock and per-type retention, from the 571,923 consecutive same-user train pairs (A -> B).
(a) flow = events in B's new part B[0:k] (pairs with alignment match >= 0.9, k > 0) vs stock = all of B's history;
    stock/flow ratio per type ~ relative time an event of that type stays in the 256-event window.
(b) survival: each A event (item, type) still present anywhere in B (same item & type), B not truncated (H_B < 256),
    by type x gap bucket and type x A-hpos bucket.
(c) fresh precedence: first-occurrence order of types on one item inside B's new part only (no retention bias)."""
import os
import numpy as np, polars as pl, numba as nb

nb.set_num_threads(12)
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_cols(40); pl.Config.set_float_precision(4)
ROOT = '<repo>/data/flat'


@nb.njit(parallel=True, cache=True)
def surv_kernel(ha_off, la, hb_off, lb, hi, ht, out_found):
    for p in nb.prange(len(ha_off)):
        a0 = ha_off[p]; b0 = hb_off[p]
        for j in range(la[p]):
            it = hi[a0 + j]; ty = ht[a0 + j]; f = 0
            for i in range(lb[p]):
                if hi[b0 + i] == it and ht[b0 + i] == ty:
                    f = 1
                    break
            out_found[a0 + j] = f


def main():
    z = np.load('<repo>/eda/history_leak/out_pairs_train.npz')
    a_idx = z['a']; b_idx = z['b']; gap = z['gap'].astype(np.int64); kk = z['k'].astype(np.int64); match = z['match']
    F = pl.read_parquet(f'{ROOT}/train/feeds.parquet').sort('fidx')
    hoff = F['hist_off'].to_numpy(); H = F['H'].to_numpy().astype(np.int64)
    hi = np.load(f'{ROOT}/train/hist_item_id.npy', mmap_mode='r')
    ht = np.load(f'{ROOT}/train/hist_event_type.npy', mmap_mode='r')
    hc = np.load(f'{ROOT}/train/hist_category_id.npy', mmap_mode='r')
    # ---------- (a) flow vs stock
    ok = (match >= 0.9) & (kk > 0) & (gap <= 7 * 86400)
    bs = b_idx[ok]; ks = kk[ok]
    rows_new = np.concatenate([np.arange(hoff[b], hoff[b] + k) for b, k in zip(bs, ks)])
    rows_all = np.concatenate([np.arange(hoff[b], hoff[b] + H[b]) for b in bs])
    tn = np.asarray(ht[rows_new]); ta = np.asarray(ht[rows_all])
    fn = np.bincount(tn, minlength=11)[1:] / len(tn); fa = np.bincount(ta, minlength=11)[1:] / len(ta)
    print(f'[F1] flow (B new part, {ok.sum():,} aligned pairs, gap<=7d, {len(tn):,} new events) vs stock (same B, {len(ta):,} events)')
    print(pl.DataFrame({'type': np.arange(1, 11), 'flow_share': fn, 'stock_share': fa, 'stock/flow': fa / fn}))
    # flow by category group for new events
    cn = np.asarray(hc[rows_new])
    df = pl.DataFrame({'t': tn, 'cat': cn})
    print('[F2] flow composition by category (new events), top 15 categories')
    x = df.group_by(['cat', 't']).agg(pl.len()).pivot(on='t', index='cat', values='len').fill_null(0)
    cols = [c for c in x.columns if c != 'cat']
    x = x.with_columns(pl.sum_horizontal(cols).alias('tot')).sort('tot', descending=True).head(15)
    print(x.select('cat', 'tot', *[(pl.col(c) / pl.col('tot')).round(3).alias('e' + c) for c in sorted(cols, key=int)]))
    # ---------- (c) fresh precedence within new part
    pair_of = np.repeat(np.arange(len(bs)), ks)
    item_new = np.asarray(hi[rows_new]); pos_new = np.concatenate([np.arange(k) for k in ks])
    dn = pl.DataFrame({'p': pair_of, 'item': item_new, 't': tn, 'pos': pos_new})
    fo = dn.group_by(['p', 'item', 't']).agg(pl.col('pos').max().alias('first_pos'), pl.len().alias('n'))
    w = fo.pivot(on='t', index=['p', 'item'], values='first_pos')
    T = list(range(1, 11))
    print('[F3] FRESH events only (B new part): P(first a OLDER than first b | item has both); ties 0.5; N in brackets')
    print('      ' + ' '.join(f'{b:>12d}' for b in T))
    for a in T:
        row = []
        for b in T:
            if a == b or str(a) not in w.columns or str(b) not in w.columns:
                row.append(f'{".":>12s}'); continue
            q = w.select(pl.col(str(a)).alias('a'), pl.col(str(b)).alias('b')).drop_nulls()
            if q.height < 30:
                row.append(f'{".":>12s}'); continue
            pr = ((q['a'] > q['b']).sum() + 0.5 * (q['a'] == q['b']).sum()) / q.height
            row.append(f'{pr:5.2f}({q.height:>6d})')
        print(f'{a:>5d} ' + ' '.join(row))
    print('[F4] fresh: events per (pair,item,type) mean, and share of items with type having repeats')
    print(fo.group_by('t').agg(pl.col('n').mean().alias('ev_per_item'), (pl.col('n') > 1).mean().alias('repeat')).sort('t'))
    # ---------- (b) survival
    nt = (H[b_idx] < 256)
    sel = np.where(nt)[0]
    rng = np.random.default_rng(1)
    sel = np.sort(rng.choice(sel, min(250_000, len(sel)), replace=False))
    A = a_idx[sel]; B = b_idx[sel]; G = gap[sel]
    rows_a = np.concatenate([np.arange(hoff[a], hoff[a] + H[a]) for a in A])
    rows_b = np.concatenate([np.arange(hoff[b], hoff[b] + H[b]) for b in B])
    hi_c = np.concatenate([np.asarray(hi[rows_a]), np.asarray(hi[rows_b])])
    ht_c = np.concatenate([np.asarray(ht[rows_a]), np.asarray(ht[rows_b])])
    la = H[A]; lb = H[B]
    ha_off = np.concatenate([[0], np.cumsum(la)[:-1]]).astype(np.int64)
    hb_off = (len(rows_a) + np.concatenate([[0], np.cumsum(lb)[:-1]])).astype(np.int64)
    found = np.zeros(len(rows_a), np.int8)
    surv_kernel(ha_off, la, hb_off, lb, hi_c, ht_c, found)
    gp = np.repeat(G, la); hp = np.concatenate([np.arange(l) for l in la]); tt = ht_c[:len(rows_a)]
    gb_edges = np.array([3600, 86400, 3 * 86400, 7 * 86400, 10 ** 9]); gnames = ['<=1h', '<=1d', '<=3d', '<=7d', '>7d']
    S = pl.DataFrame({'t': tt, 'found': found, 'gb': np.searchsorted(gb_edges, gp), 'hb': np.minimum(hp // 64, 3)})
    print(f'[S1] survival of A events into B (B not truncated, {len(sel):,} pairs, {len(rows_a):,} A events): P(same item+type in B)')
    s1 = S.group_by(['t', 'gb']).agg(pl.col('found').mean()).pivot(on='gb', index='t', values='found').sort('t')
    s1 = s1.rename({str(i): gnames[i] for i in range(5) if str(i) in s1.columns})
    print(s1)
    print('[S2] survival by A-hpos bucket (64-wide), all gaps')
    s2 = S.group_by(['t', 'hb']).agg(pl.col('found').mean()).pivot(on='hb', index='t', values='found').sort('t')
    print(s2)
    print('[S3] survival by A-hpos bucket, gap <= 1 day')
    s3 = S.filter(pl.col('gb') <= 1).group_by(['t', 'hb']).agg(pl.col('found').mean(), pl.len()).pivot(on='hb', index='t', values='found').sort('t')
    print(s3)


if __name__ == '__main__':
    main()
