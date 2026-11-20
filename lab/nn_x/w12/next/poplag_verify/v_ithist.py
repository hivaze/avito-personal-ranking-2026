"""poplag_verify step C: my own rebuild of it_hist_{all,et1..4}_luo for a random SAMPLE of holdout feeds, (a) with the stored lag-0 rule and (b) with the
declared v2/v3 rule at band g = 3 (private-like and public-like lags), by an event-level join (no numba kernel, no code of the study). LABEL-FREE.
Also measures how much of the "lagged" count comes from snapshots dated AFTER the holdout row.
  cd <repo>/nn_x/w12/next/poplag_verify && CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_ithist.py [n_feeds=600]"""
from v_common import *
import time

T0 = time.time()


def lg(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


NS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
G = 3
PAT = [2, 4, 6, 7, 8, 9]
EDGES = np.array([0, 1, 3, 7, 10, 24, 76, np.inf]); BN = ['<1d', '1-3d', '3-7d', '7-10d', '10-24d', '24-76d', '>76d+cap']
KN = np.load(ROOT / 'data/feats/life_f4_age_knots.npy')
ftr, hold, _ = my_holdout()
fts = ftr['timestamp'].to_numpy().astype(np.int64); fday = (fts + TZ) // DAY; fuser = ftr['user_id'].to_numpy().astype(np.int64)
fM = ftr['M'].to_numpy().astype(np.int64); fco = ftr['cand_off'].to_numpy().astype(np.int64); ffid = ftr['feed_id'].to_numpy()
fsp = (((ffid.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)) >> np.uint64(40)) % np.uint64(37)) == 0      # shared 1/37 sample (feat_new/hz_common.py sparse_flag, re-typed)
lg(f'train feeds {len(ftr)}; sparse-sample share {fsp.mean():.4f} (1/37 = {1 / 37:.4f})')
pick = np.sort(np.random.default_rng(20261115).choice(len(hold), NS, replace=False))            # positions in the holdout list
off_h = np.concatenate([[0], np.cumsum(fM[hold])[:-1]])
hpos = np.concatenate([np.arange(off_h[i], off_h[i] + fM[hold[i]]) for i in pick])              # row positions inside the 4,630,966 holdout rows
fr = np.repeat(pick, fM[hold[pick]])                                                             # holdout-list index of each sampled row
rows = np.concatenate([np.arange(fco[hold[i]], fco[hold[i]] + fM[hold[i]]) for i in pick])       # train candidate rows
item = np.asarray(np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r')[rows]).astype(np.int64)
f4 = np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')[rows]).astype(np.float64)
u_ = f4 * 64.0; k_ = np.clip(np.floor(u_).astype(np.int64), 1, 62)
ah = np.where(f4 < 63 / 64, KN[k_] + (u_ - k_) * (KN[k_ + 1] - KN[k_]), np.nan)
bk = np.clip(np.searchsorted(EDGES, np.where(np.isnan(ah), 1e9, ah / 24.0), side='right') - 1, 0, 6)
nR = len(rows); lg(f'sample: {NS} holdout feeds, {nR} rows, {len(np.unique(item))} items')

# ---- all history events of the sampled items: train snapshots (chunked scan), val / test snapshots
need = np.unique(item); NI = int(need.max()) + 1; look = np.zeros(NI, np.bool_); look[need] = True
hi = np.load(FLAT / 'train/hist_item_id.npy', mmap_mode='r'); hf = np.load(FLAT / 'train/hist_fidx.npy', mmap_mode='r'); he = np.load(FLAT / 'train/hist_event_type.npy', mmap_mode='r')
ev_i, ev_f, ev_e = [], [], []
STEP = 32_000_000
for s in range(0, len(hi), STEP):
    x = np.asarray(hi[s:s + STEP]); m = np.flatnonzero((x < NI) & look[np.minimum(x, NI - 1)])
    ev_i.append(x[m].astype(np.int64)); ev_f.append(np.asarray(hf[s:s + STEP])[m].astype(np.int64)); ev_e.append(np.asarray(he[s:s + STEP])[m].astype(np.int64))
ev_i, ev_f, ev_e = np.concatenate(ev_i), np.concatenate(ev_f), np.concatenate(ev_e)
lg(f'train history events of the sampled items: {len(ev_i)} of {len(hi)}')
E_tr = pl.DataFrame({'item': ev_i, 's': ev_f, 'et': ev_e, 's_user': fuser[ev_f], 's_ts': fts[ev_f], 's_day': fday[ev_f], 's_sp': fsp[ev_f], 'src': np.zeros(len(ev_i), np.int8)})
parts = [E_tr]
for sp in ('val', 'test'):
    f = pl.read_parquet(FLAT / sp / 'feeds.parquet'); x = np.load(FLAT / sp / 'hist_item_id.npy'); xf = np.load(FLAT / sp / 'hist_fidx.npy').astype(np.int64)
    m = np.flatnonzero((x < NI) & look[np.minimum(x, NI - 1)]); ts_ = f['timestamp'].to_numpy().astype(np.int64)[xf[m]]
    parts.append(pl.DataFrame({'item': x[m].astype(np.int64), 's': np.full(len(m), -1, np.int64), 'et': np.load(FLAT / sp / 'hist_event_type.npy')[m].astype(np.int64),
                               's_user': f['user_id'].to_numpy().astype(np.int64)[xf[m]], 's_ts': ts_, 's_day': (ts_ + TZ) // DAY, 's_sp': np.ones(len(m), np.bool_), 'src': np.ones(len(m), np.int8)}))
    lg(f'{sp} history events of the sampled items: {len(m)}')
EV = pl.concat(parts)
ht = fts[hold]; hd = fday[hold]; hu = fuser[hold]
stored = {n: np.asarray(np.load(FC / 'train' / f'{n}.npy', mmap_mode='r')[rows]) for n in ['it_hist_all_luo'] + [f'it_hist_et{k}_luo' for k in (1, 2, 3, 4)]}
te = pl.read_parquet(FLAT / 'test/feeds.parquet'); tday = (te['timestamp'].to_numpy().astype(np.int64) + TZ) // DAY; tpub = te['timestamp'].to_numpy() < PUB_TS


def counts(J, keep):
    J = J.filter(keep)
    out = {}
    a = J.group_by('row').agg(pl.len().alias('n'))
    x = np.zeros(nR, np.int64); x[a['row'].to_numpy()] = a['n'].to_numpy(); out['it_hist_all_luo'] = x
    for k in (1, 2, 3, 4):
        a = J.filter(pl.col('et') == k).group_by('row').agg(pl.len().alias('n'))
        x = np.zeros(nR, np.int64); x[a['row'].to_numpy()] = a['n'].to_numpy(); out[f'it_hist_et{k}_luo'] = x
    return out, J


for nm in ('lag0', 'priv', 'pub'):
    if nm == 'lag0':
        L = np.zeros(len(hold), np.int64)
    else:
        sel = tpub if nm == 'pub' else ~tpub
        a, b = np.unique(tday[sel] - SEP20, return_counts=True); cdf = np.cumsum(b) / b.sum()
        un = ((ffid[hold].astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93)) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
        L = a[np.minimum(np.searchsorted(cdf, un, side='right'), len(a) - 1)].astype(np.int64)
    cd = hd - L; c_sec = cd * DAY - TZ
    E = c_sec[fr] - (ht[fr] - 3600.0 * ah)
    w = np.minimum(SEP20 * DAY - TZ, np.maximum(c_sec[fr], (SEP6 * DAY - TZ) + np.clip(np.where(np.isnan(E), 1e12, E), 0, 14 * DAY)))
    D = np.floor((w + TZ) / DAY + 0.5).astype(np.int64)
    Q = pl.DataFrame({'row': np.arange(nR), 'item': item, 'u': hu[fr], 't': ht[fr], 'dt': hd[fr], 'cd': cd[fr], 'D': D, 'own': hold[fr].astype(np.int64)})
    J = Q.join(EV, on='item', how='inner')
    not_later_same_user = ~((pl.col('u') != 0) & (pl.col('s_user') == pl.col('u')) & (pl.col('s_ts') > pl.col('t')))
    if nm == 'lag0':
        c, _ = counts(J, not_later_same_user)                                    # stored rule: every snapshot of train + val + test, minus later ones of the same user
        for n in stored:
            lg(f'CHECK lag-0 rule vs STORED {n}: exact on {np.mean(c[n] == stored[n]):.6f} of {nR} sampled rows (max |diff| {np.abs(c[n] - stored[n]).max():.0f})')
        continue
    dense = (pl.col('src') == 0) & (pl.col('s_day') < pl.col('D')) & ((pl.col('s_day') - pl.col('dt')).abs() >= G)
    own = (pl.col('src') == 0) & (pl.col('s') == pl.col('own'))
    sparse = pl.col('s_sp') & (pl.col('s_day') - pl.col('cd')).is_in(PAT)
    c, Jk = counts(J.with_columns(dense.alias('is_dense'), own.alias('is_own')), not_later_same_user & (dense | own | sparse))
    for n in stored:
        st = np.load(STUDY / f'lag_{nm}_v3_{n}.npy')[hpos]
        lg(f'[{nm}] MY REBUILD vs the study\'s lag_{nm}_v3_{n}: exact on {np.mean(c[n] == st):.6f} of {nR} sampled rows (max |diff| {np.abs(c[n] - st).max():.0f}); zero share mine {np.mean(c[n] == 0):.4f} study {np.mean(st == 0):.4f} stored lag-0 {np.mean(stored[n] == 0):.4f}')
    # where the lagged count comes from (label-free): dense snapshots before / after the row's day, sparse snapshots, own snapshot
    Jk = Jk.with_columns(pl.when(pl.col('is_own')).then(pl.lit('own snapshot')).when(pl.col('is_dense') & (pl.col('s_day') > pl.col('dt'))).then(pl.lit('dense, day AFTER the row'))
                         .when(pl.col('is_dense')).then(pl.lit('dense, day before the row')).when(pl.col('s_day') > pl.col('dt')).then(pl.lit('sparse, day after the row'))
                         .otherwise(pl.lit('sparse, day on/before the row')).alias('kind'))
    tot = len(Jk)
    for r in Jk.group_by('kind').agg(pl.len().alias('n')).sort('n', descending=True).iter_rows():
        lg(f'[{nm}] share of the lagged it_hist_all_luo count mass from "{r[0]}": {r[1] / tot:.3f}')
    a = Jk.group_by('row').agg(((pl.col('is_dense') & (pl.col('s_day') > pl.col('dt'))) | (~pl.col('is_dense') & ~pl.col('is_own') & (pl.col('s_day') > pl.col('dt')))).sum().alias('fut'), pl.len().alias('n'))
    fut = np.zeros(nR); tot_r = np.zeros(nR); fut[a['row'].to_numpy()] = a['fut'].to_numpy(); tot_r[a['row'].to_numpy()] = a['n'].to_numpy()
    nz = tot_r > 0
    lg(f'[{nm}] rows with a non-zero lagged count: {nz.mean():.3f}; of those, rows whose count comes ONLY from snapshots dated after the row\'s day: {np.mean(fut[nz] == tot_r[nz]):.3f}; '
       f'rows with at least half of the count from after the row: {np.mean(fut[nz] >= 0.5 * tot_r[nz]):.3f}; mean per-row future share {np.mean(fut[nz] / tot_r[nz]):.3f}')
    for b in range(7):
        m = nz & (bk == b)
        if m.sum() > 50:
            print(f'      {BN[b]:9s} rows {np.mean(bk == b):.3f} non-zero {np.mean(nz[bk == b]):.3f} | mean per-row share of the count from snapshots after the row {np.mean(fut[m] / tot_r[m]):.3f}', flush=True)
lg('done')
