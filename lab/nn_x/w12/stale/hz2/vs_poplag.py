"""hz2 vs poplag v3 on the same holdout rows (label-free, information): private-like feeds of hz2 have the lag of poplag's private-like view."""
from hz2_common import *
PL = ROOT / 'nn_x/w12/next/poplag/cache'
hold = np.load(PL / 'mini/hold.npy'); crow = np.load(PL / 'mini/crow.npy'); Mh = np.load(PL / 'mini/M.npy'); ts = np.load(PL / 'mini/ts.npy')
L = np.load(OUT / 'train/hz2_lag.npy')[hold].astype(np.int64); pub = np.load(OUT / 'train/hz2_part_pub.npy')[hold]
Lp = np.load(PL / 'view_priv_L.npy'); assert np.array_equal(L[~pub], Lp[~pub]), 'private-like lags differ from poplag'
d = day_of(ts) - DAY0; dg = np.where(d <= 2, 0, np.where(d <= 6, 1, np.where(d <= 9, 2, 3)))
frow = np.repeat(np.arange(len(hold)), Mh); sel = ~pub[frow]
NM = ['Sep 6-8', 'Sep 9-12', 'Sep 13-14', 'Sep 17-19']
out = [f'holdout feeds {len(hold)}; private-like {int((~pub).sum())} (lags identical to poplag private-like view: True)']
for col in ('it_hist_all_luo', 'seq_cv_pop', 'xp_imp_c50'):
    v3 = xt_transform(col, np.load(PL / f'lag_priv_v3_{col}.npy'))
    st = xt_transform(col, np.asarray(np.load(FC / 'train' / f'{col}.npy', mmap_mode='r')[crow]))
    for vn, vd in (('hz2', 'feats'), ('hz2L', 'feats_L')):
        x = xt_transform(col, np.asarray(np.load(HERE / vd / 'train' / f'{HZ2_NAME[col]}.npy', mmap_mode='r')[crow]))
        for g in range(4):
            m = sel & (dg[frow] == g)
            out.append(f'{col:16s} {vn:5s} {NM[g]:10s} rows {int(m.sum()):7d}: zero share stored {float((st[m] == 0).mean()):.3f} | poplag v3 {float((v3[m] == 0).mean()):.3f} | {vn} {float((x[m] == 0).mean()):.3f}; '
                       f'same zero pattern as v3 {float(((x[m] == 0) == (v3[m] == 0)).mean()):.3f}; corr(model scale) with v3 {float(np.corrcoef(x[m], v3[m])[0, 1]):.3f}, with stored {float(np.corrcoef(x[m], st[m])[0, 1]):.3f}; '
                       f'mean {float(st[m].mean()):.3f} / {float(v3[m].mean()):.3f} / {float(x[m].mean()):.3f}')
print('\n'.join(out)); open(LOGS / 'vs_poplag.txt', 'w').write('\n'.join(out) + '\n')
