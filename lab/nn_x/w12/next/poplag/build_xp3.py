"""poplag v3: xp_imp_c50 with the exposure days preserved under the band (DECLARATION.txt, third block). Views <pub|priv>_v3 = v2 g=3 columns for the
eight W8 inputs (copied) + the new xp_imp_c50. Label-free; light (one sequential pass over the train candidate item ids + one polars join)."""
import sys, time, shutil
from pl_common import *
import numba as nb
G = 3
T0 = time.time(); lg = lambda *a: print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
DAY0 = int(day_of(np.array([DATA0]))[0]); DAYC = int(day_of(np.array([CUT_VT]))[0]); ND = DAYC - DAY0 + 1


@nb.njit(cache=False)
def imp_by_day(citem, cfidx, fday, nidx, n_items, day0, nd):
    cnt = np.zeros((n_items, nd), np.int32)
    for r in range(len(citem)):
        k = nidx[citem[r]]
        if k >= 0:
            d = fday[cfidx[r]] - day0
            if 0 <= d < nd:
                cnt[k, d] += 1
    return cnt


mini = CACHE / 'mini'; L_ = lambda n: np.load(mini / f'{n}.npy')
hold, crow, Mh = L_('hold'), L_('crow'), L_('M'); n_rows = len(crow); feed_of_row = np.repeat(np.arange(len(hold)), Mh)
c_item = L_('c_item').astype(np.int64)
ftr = feeds('train'); fts = ftr['timestamp'].to_numpy().astype(np.int64); fuser = ftr['user_id'].to_numpy().astype(np.int64); fday = day_of(fts).astype(np.int32)
hu = fuser[hold]; ht = fts[hold]; hday = fday[hold].astype(np.int64)
t_row = ht[feed_of_row]; u_row = hu[feed_of_row]; dt_row = hday[feed_of_row]
NI = int(max(np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r').max(), c_item.max())) + 1
uit = np.unique(c_item); nidx = np.full(NI, -1, np.int32); nidx[uit] = np.arange(len(uit), dtype=np.int32); ck = nidx[c_item].astype(np.int64)
cnt = imp_by_day(np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r'), np.load(FLAT / 'train/cand_fidx.npy', mmap_mode='r'), fday, nidx, len(uit), DAY0, ND)
cum = np.concatenate([np.zeros((len(uit), 1), np.int64), np.cumsum(cnt, 1, dtype=np.int64)], 1); del cnt
hu_nz = np.unique(hu[hu != 0]); uf = np.flatnonzero(np.isin(fuser, hu_nz)); fco = ftr['cand_off'].to_numpy().astype(np.int64); fM = ftr['M'].to_numpy().astype(np.int64)
urow = rows_of(fco[uf], fM[uf]); uff = np.repeat(uf, fM[uf])
later = pl.DataFrame({'user': fuser[uff], 'item': np.asarray(np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r')[urow]).astype(np.int64), 'ts2': fts[uff], 'day2': fday[uff].astype(np.int64)})
lg('impressions by day ready')
for nm in ('priv', 'pub'):
    D = np.load(CACHE / f'view_{nm}_D.npy').astype(np.int64)
    lo_d = dt_row - G + 1; hi_d = dt_row + G                      # band = days lo_d .. hi_d - 1
    nb_days = np.maximum(hi_d - np.maximum(lo_d, DAY0), 0)        # band days on or after Sep 6
    D2 = np.where(D <= lo_d, D, np.minimum(D + nb_days, DAYC))    # window ends before the band: unchanged; else extended by the skipped band days
    for mode, Dx in (('v2', D), ('v3', D2)):
        Dd = np.clip(Dx - DAY0, 0, ND); lo = np.clip(np.minimum(Dx, lo_d) - DAY0, 0, ND); hi = np.clip(np.minimum(Dx, hi_d) - DAY0, 0, ND)
        n_imp = cum[ck, lo] + (cum[ck, Dd] - cum[ck, hi])
        rows_df = pl.DataFrame({'row': np.arange(n_rows), 'user': u_row, 'item': c_item, 't': t_row, 'D': Dx, 'dt': dt_row})
        j = (rows_df.lazy().filter(pl.col('user') != 0).join(later.lazy(), on=['user', 'item'], how='inner')
             .filter((pl.col('day2') < pl.col('D')) & (pl.col('ts2') >= pl.col('t')) & ((pl.col('day2') - pl.col('dt')).abs() >= G)).group_by('row').agg(pl.len().alias('n')).collect())
        sub = np.zeros(n_rows, np.int64); sub[j['row'].to_numpy()] = j['n'].to_numpy()
        n_adj = n_imp - sub; assert n_adj.min() >= 0
        xp = np.log1p(np.minimum(n_adj, 50)).astype(np.float32)
        if mode == 'v2':
            ref = np.load(CACHE / f'lag_{nm}_g{G}_xp_imp_c50.npy'); lg(f'[{nm}] code check: this script at v2 semantics == build_lag2 g={G} file: {bool(np.array_equal(xp, ref))}')
            assert np.array_equal(xp, ref)
        else:
            for n in SEQ + LUO:
                shutil.copyfile(CACHE / f'lag_{nm}_g{G}_{n}.npy', CACHE / f'lag_{nm}_v3_{n}.npy')
            np.save(CACHE / f'lag_{nm}_v3_xp_imp_c50.npy', xp)
            lg(f'[{nm}_v3] saved: xp_imp_c50 zero share {float((xp == 0).mean()):.4f} (v2 g={G}: {float((ref == 0).mean()):.4f}); rows changed vs v2 {float((xp != ref).mean()):.4f}; rows with extended window {float((D2 != D).mean()):.4f}')
lg('done')
