"""e02: the leak mechanism -> which event types does a feed-A candidate produce in the SAME user's NEXT feed B?
Pairs: the 571,923 consecutive same-user (user_id != 0) train feed pairs of eda/history_leak/out_pairs_train.npz
(A earlier, B = next feed of that user; gap = tB - tA in seconds; k/match = alignment of B[k:] onto A).
Candidates are never in their own feed's history, so every occurrence of an A candidate in B's history is an event
that happened after A's snapshot.  For every A candidate found in B we keep the full per-item event sequence
(oldest -> newest, type and source).  Output: out/e02_found.parquet + tables in the log."""
import os
import numpy as np, polars as pl, numba as nb

nb.set_num_threads(12)
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30)
ROOT = '<repo>/data/flat'
OUT = '<repo>/nn_x/codes/events/out'; os.makedirs(OUT, exist_ok=True)
MAXE = 16


@nb.njit(parallel=True, cache=True)
def count_kernel(a_idx, b_idx, hoff, H, coff, M, hi, ci, cstart, nfound):
    for p in nb.prange(len(a_idx)):
        a = a_idx[p]; b = b_idx[p]
        hb = hoff[b]; lb = H[b]; ca = coff[a]
        for i in range(M[a]):
            it = ci[ca + i]; n = 0
            for j in range(lb):
                if hi[hb + j] == it:
                    n += 1
            nfound[cstart[p] + i] = n


@nb.njit(parallel=True, cache=True)
def fill_kernel(a_idx, b_idx, hoff, H, coff, M, hi, ht, hs, ci, cstart, nfound, fstart, seq_t, seq_s, newest, oldest):
    for p in nb.prange(len(a_idx)):
        a = a_idx[p]; b = b_idx[p]
        hb = hoff[b]; lb = H[b]; ca = coff[a]
        for i in range(M[a]):
            g = cstart[p] + i
            if nfound[g] == 0:
                continue
            r = fstart[g]; it = ci[ca + i]; m = 0
            for j in range(lb - 1, -1, -1):  # oldest -> newest
                if hi[hb + j] == it:
                    if m < MAXE:
                        seq_t[r, m] = ht[hb + j]; seq_s[r, m] = hs[hb + j]
                    if m == 0:
                        oldest[r] = j
                    newest[r] = j
                    m += 1


def main():
    z = np.load('<repo>/eda/history_leak/out_pairs_train.npz')
    a_idx = z['a']; b_idx = z['b']; gap = z['gap'].astype(np.int64); kk = z['k']; match = z['match']
    F = pl.read_parquet(f'{ROOT}/train/feeds.parquet').sort('fidx')
    hoff = F['hist_off'].to_numpy(); H = F['H'].to_numpy().astype(np.int64)
    coff = F['cand_off'].to_numpy(); M = F['M'].to_numpy().astype(np.int64)
    print('loading arrays', flush=True)
    hi = np.load(f'{ROOT}/train/hist_item_id.npy', mmap_mode='r')
    ht = np.load(f'{ROOT}/train/hist_event_type.npy', mmap_mode='r')
    hs = np.load(f'{ROOT}/train/hist_event_source.npy', mmap_mode='r')
    ci = np.load(f'{ROOT}/train/cand_item_id.npy', mmap_mode='r')
    ct = np.load(f'{ROOT}/train/cand_target.npy', mmap_mode='r')
    ccat = np.load(f'{ROOT}/train/cand_category_id.npy', mmap_mode='r')
    # compact copies: only the A candidates and B histories
    ca_rows = np.concatenate([np.arange(coff[a], coff[a] + M[a]) for a in a_idx])
    hb_rows = np.concatenate([np.arange(hoff[b], hoff[b] + H[b]) for b in b_idx])
    print('A cand rows', len(ca_rows), 'B hist rows', len(hb_rows), flush=True)
    ci_c = np.asarray(ci[ca_rows]); ct_c = np.asarray(ct[ca_rows]); cc_c = np.asarray(ccat[ca_rows])
    hi_c = np.asarray(hi[hb_rows]); ht_c = np.asarray(ht[hb_rows]); hs_c = np.asarray(hs[hb_rows])
    n = len(a_idx)
    coff2 = np.concatenate([[0], np.cumsum(M[a_idx])[:-1]]).astype(np.int64)
    hoff2 = np.concatenate([[0], np.cumsum(H[b_idx])[:-1]]).astype(np.int64)
    ar = np.arange(n)
    M2 = np.zeros(n, np.int64); M2[:] = M[a_idx]; H2 = np.zeros(n, np.int64); H2[:] = H[b_idx]
    # kernels index by "a"/"b" -> use identity indices into compact offsets
    nfound = np.zeros(len(ci_c), np.int16)
    count_kernel(ar, ar, hoff2, H2, coff2, M2, hi_c, ci_c, coff2, nfound)
    found = nfound > 0
    fstart = np.full(len(ci_c), -1, np.int64); fstart[found] = np.arange(found.sum())
    NF = int(found.sum())
    seq_t = np.zeros((NF, MAXE), np.int8); seq_s = np.full((NF, MAXE), -1, np.int8)
    newest = np.zeros(NF, np.int16); oldest = np.zeros(NF, np.int16)
    fill_kernel(ar, ar, hoff2, H2, coff2, M2, hi_c, ht_c, hs_c, ci_c, coff2, nfound, fstart, seq_t, seq_s, newest, oldest)
    pair_of_row = np.repeat(ar, M2)
    gap_r = gap[pair_of_row]
    gb_edges = [128, 600, 3600, 6 * 3600, 86400, 3 * 86400, 7 * 86400, 10 ** 9]
    gb_names = ['<=128s', '<=10m', '<=1h', '<=6h', '<=1d', '<=3d', '<=7d', '>7d']
    gbi = np.searchsorted(np.array(gb_edges), gap_r, side='left')
    # ---- table 1: recall / precision by gap
    T = pl.DataFrame({'t': ct_c.astype(np.int8), 'found': found, 'gb': gbi.astype(np.int8)})
    print('\n[T1] pairs:', n, ' A candidates:', len(ct_c), ' positives:', int(ct_c.sum()), ' found in B history:', NF)
    t1 = T.group_by('gb').agg(pl.len().alias('cands'), pl.col('t').sum().alias('pos'),
                              (pl.col('found') & (pl.col('t') == 1)).sum().alias('pos_found'),
                              (pl.col('found') & (pl.col('t') == 0)).sum().alias('neg_found')).sort('gb')
    t1 = t1.with_columns((pl.col('pos_found') / pl.col('pos')).alias('recall_pos'),
                         (pl.col('neg_found') / (pl.col('cands') - pl.col('pos'))).alias('found_rate_neg'),
                         (pl.col('pos_found') / (pl.col('pos_found') + pl.col('neg_found'))).alias('P(t|found)'),
                         pl.col('gb').map_elements(lambda i: gb_names[i], return_dtype=pl.Utf8).alias('gap'))
    print(t1)
    # ---- per found item
    fr = np.where(found)[0]
    t_f = ct_c[fr]; g_f = gbi[fr]; cat_f = cc_c[fr]; n_f = nfound[fr]
    mask = np.zeros(NF, np.int32)
    for m in range(MAXE):
        v = seq_t[:, m].astype(np.int32)
        mask |= np.where(v > 0, 1 << v, 0)
    has = {ty: ((mask >> ty) & 1).astype(bool) for ty in range(1, 11)}
    # sequence strings (oldest->newest), sources shown for types 1/4/7
    def tok(t, s):
        return f'{t}' if t not in (1, 4, 7) else f'{t}s{s if s <= 3 else "x"}'
    seqs = []
    for r in range(NF):
        L = min(int(n_f[r]), MAXE)
        toks = [tok(int(seq_t[r, m]), int(seq_s[r, m])) for m in range(L)]
        # collapse consecutive repeats: 3,3,3 -> 3+
        out = []
        for x in toks:
            if out and out[-1].rstrip('+') == x:
                out[-1] = x + '+'
            else:
                out.append(x)
        seqs.append('>'.join(out) + ('...' if n_f[r] > MAXE else ''))
    D = pl.DataFrame({'t': t_f.astype(np.int8), 'gb': g_f.astype(np.int8), 'cat': cat_f, 'nev': n_f, 'mask': mask,
                      'first_t': seq_t[:, 0], 'first_s': seq_s[:, 0], 'newest_hpos': newest, 'oldest_hpos': oldest,
                      'seq': seqs, 'gap': gap_r[fr], 'k': kk[pair_of_row[fr]], 'match': match[pair_of_row[fr]],
                      **{f'h{ty}': has[ty] for ty in range(1, 11)}})
    D.write_parquet(f'{OUT}/e02_found.parquet')
    print('\n[T2] among FOUND positives vs FOUND negatives: share having each event type on the item (any time in B)')
    rows = []
    for lab in (1, 0):
        d = D.filter(pl.col('t') == lab)
        rows.append({'label': lab, 'n': d.height, **{f'e{ty}': round(d[f'h{ty}'].mean(), 4) for ty in range(1, 11)},
                     'e1_src1': round(((d['first_t'] == 1) & (d['first_s'] == 1)).mean(), 4)})
    print(pl.DataFrame(rows))
    print('\n[T3] target rate P(t=1 | found, item has type e in B) and P(t=1 | found, ONLY type e) ')
    rows = []
    for ty in range(1, 11):
        d = D.filter(pl.col(f'h{ty}'))
        only = D.filter(pl.col('mask') == (1 << ty))
        rows.append({'type': ty, 'n_found_with': d.height, 'P(t|has)': round(d['t'].mean(), 4) if d.height else None,
                     'n_only': only.height, 'P(t|only)': round(only['t'].mean(), 4) if only.height else None})
    print(pl.DataFrame(rows))
    print('\n[T4] first (oldest) event on the found item: (type, source) -> n, P(t=1)')
    print(D.with_columns(pl.when(pl.col('first_s') <= 4).then(pl.col('first_s')).otherwise(99).alias('fs'))
          .group_by(['first_t', 'fs']).agg(pl.len(), pl.col('t').mean().alias('P(t)'), pl.col('t').sum().alias('pos'))
          .sort('len', descending=True).head(30))
    print('\n[T5] FOUND positives: share with each type by gap (funnel progression after the click)')
    rows = []
    for g in range(len(gb_names)):
        d = D.filter((pl.col('t') == 1) & (pl.col('gb') == g))
        if d.height == 0:
            continue
        rows.append({'gap': gb_names[g], 'n': d.height, **{f'e{ty}': round(d[f'h{ty}'].mean(), 4) for ty in range(1, 11)},
                     'mean_nev': round(d['nev'].mean(), 2)})
    print(pl.DataFrame(rows))
    print('\n[T5b] FOUND negatives: share with each type by gap')
    rows = []
    for g in range(len(gb_names)):
        d = D.filter((pl.col('t') == 0) & (pl.col('gb') == g))
        if d.height == 0:
            continue
        rows.append({'gap': gb_names[g], 'n': d.height, **{f'e{ty}': round(d[f'h{ty}'].mean(), 4) for ty in range(1, 11)},
                     'first_e1s1': round(((d['first_t'] == 1) & (d['first_s'] == 1)).mean(), 4)})
    print(pl.DataFrame(rows))
    for lab in (1, 0):
        print(f'\n[T6] top per-item sequences (oldest > newest; "+" = repeated; sources for 1/4/7) label={lab}')
        d = D.filter(pl.col('t') == lab)
        print(d.group_by('seq').agg(pl.len()).with_columns((pl.col('len') / d.height).alias('share')).sort('len', descending=True).head(40))
    print('\n[T7] short-gap pairs (gap <= 1h): sequences of FOUND positives')
    d = D.filter((pl.col('t') == 1) & (pl.col('gap') <= 3600))
    print(d.group_by('seq').agg(pl.len()).with_columns((pl.col('len') / d.height).alias('share')).sort('len', descending=True).head(20))


if __name__ == '__main__':
    main()
