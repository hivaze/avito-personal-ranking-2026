"""J. (a) cold-minus-warm FT offset by day (val Tue/Thu; test Sat..Tue); (b) test-side correlations of candidate files with the W8 file."""
from lib import *
luo = np.load(FEATS / 'val/it_hist_all_luo.npy'); cold = (np.nan_to_num(np.asarray(luo, np.float64), nan=0.0) == 0).astype(np.float64)
luot = np.load(FEATS / 'test/it_hist_all_luo.npy'); coldt = (np.nan_to_num(np.asarray(luot, np.float64), nan=0.0) == 0).astype(np.float64)
def gap(s, c, M, off):
    nc = np.add.reduceat(c, off); nw = M - nc
    return np.add.reduceat(s * c, off) / np.maximum(nc, 1) - np.add.reduceat(s * (1 - c), off) / np.maximum(nw, 1), (nc > 0) & (nw > 0), nc / M
sv = zv(pool(W8, 'ft', 'val')) - zv(pool(W8, 'to', 'val')); st_ = zt(pool(W8, 'ft', 'test')) - zt(pool(W8, 'to', 'test'))
gv, okv, csv_ = gap(sv, cold, Mv, ov); gt, okt, cst = gap(st_, coldt, Mt, ot)
dayv = (tv.astype(np.int64) + 3 * 3600) // 86400; dayt = (tt.astype(np.int64) + 3 * 3600) // 86400; d0 = np.unique(dayv)[0]
print('(a) W8 pooled FT shift, cold minus warm candidates (per-feed mean, z units) and feed cold share, by MSK day (day 0 = Tue Sep 22)')
for d in np.unique(dayv):
    m = (dayv == d) & okv
    if m.sum() > 100: print(f'  val  day {d - d0}: offset {gv[m].mean():+.4f}+-{gv[m].std() / np.sqrt(m.sum()):.4f} | cold share {csv_[dayv == d].mean():.3f} | feeds {m.sum()}')
for d in np.unique(dayt):
    m = (dayt == d) & okt
    print(f'  test day {d - d0}: offset {gt[m].mean():+.4f}+-{gt[m].std() / np.sqrt(m.sum()):.4f} | cold share {cst[dayt == d].mean():.3f} | feeds {m.sum()}')
# offset at matched feed cold share (is the larger test offset a composition effect?)
bins = [0, .15, .3, .45, .6, 1.01]
print('  offset by feed cold-share bin: ' + ' | '.join(f'[{a:.2f},{b:.2f}) val {gv[okv & (csv_ >= a) & (csv_ < b)].mean():+.4f} test {gt[okt & (cst >= a) & (cst < b)].mean():+.4f}' for a, b in zip(bins[:-1], bins[1:])))
print('\n(b) test-side per-feed-z correlation with the W8 file (0.5 z(FT) + 0.5 z(TO))')
ft, to = zt(pool(W8, 'ft', 'test')), zt(pool(W8, 'to', 'test')); base = zt(0.5 * ft + 0.5 * to)
pub = np.repeat(np.arange(len(Mt)) < 8562, Mt)
for nm, x in (('TO only (w_ft 0)', to), ('w_ft 0.25', 0.25 * ft + 0.75 * to), ('w_ft 0.75', 0.75 * ft + 0.25 * to), ('FT only (w_ft 1)', ft)):
    xz = zt(x); print(f'  {nm:18s} corr {np.corrcoef(xz, base)[0, 1]:.5f} (public rows {np.corrcoef(xz[pub], base[pub])[0, 1]:.5f}); mean per-feed Spearman {np.nanmean(feed_corr(rkt(x), rkt(base), Mt, ot)[Mt > 2]):.5f}')
g_ft, g_to = np.load(P / 'ens_G3_ft_test.npy'), np.load(P / 'ens_G3_to_test.npy'); g = zt(0.5 * zt(g_ft) + 0.5 * zt(g_to))
print(f'  reference: G3 TO-only vs G3 file corr {np.corrcoef(zt(g_to), g)[0, 1]:.5f} (that pair read public -0.00047 at a paired SE 0.00050)')
