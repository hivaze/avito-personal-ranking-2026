"""LABEL-FREE exploration of candidate gate features on val / public-like / private-like test feeds (no labels loaded)."""
from lib import *
BOUND = 1790497920
d = pl.read_parquet(ROOT / 'eda/errors/segments/feed_table.parquet'); t = pl.read_parquet(ROOT / 'eda/errors/segments/test_segments.parquet')
out = {}
for sp, tab in (('val', d), ('test', t)):
    f, M, off = meta(sp); n = len(M); fid = np.repeat(np.arange(n), M)
    assert (tab['fidx'].to_numpy() == np.arange(n)).all() and (tab['M'].to_numpy() == M).all()
    luo = np.load(ROOT / f'nn/feat_cache/{sp}/it_hist_all_luo.npy', mmap_mode='r'); luo = np.asarray(luo, np.float32)
    assert len(luo) == M.sum()
    f4 = np.load(FLAT / f'{sp}/cand_f4.npy').astype(np.float64)
    c = {k: tab[k].to_numpy().astype(np.float64) for k in ('M', 'H', 'uid0', 'n_train_feeds', 'cold_share', 'coldcand_share', 'hour', 'n_cat', 'max_cat_share', 'hist_n_cat', 'hist_top_cat_share', 'cand_cat_in_hist_share')}
    c['cold_luo'] = np.bincount(fid, luo == 0, n) / M
    c['mlog_luo'] = np.bincount(fid, np.log1p(np.maximum(luo, 0)), n) / M
    c['f4_mean'] = np.bincount(fid, f4, n) / M
    c['f4_med'] = np.array([np.median(f4[o:o + m]) for o, m in zip(off, M)])
    c['ts'] = f['timestamp'].to_numpy().astype(np.float64)
    out[sp] = c
    print(sp, 'f4 quantiles', np.quantile(f4, [0, .01, .1, .5, .9, .99, 1]), 'luo q', np.quantile(luo, [0, .1, .5, .9, .99, 1]), 'nan', np.isnan(f4).sum(), np.isnan(luo).sum())
pub = out['test']['ts'] < BOUND
print('n val', len(out['val']['M']), 'public', pub.sum(), 'private', (~pub).sum())
def psi(a, b, k=10):
    e = np.unique(np.quantile(a, np.linspace(0, 1, k + 1))); e[0] = -np.inf; e[-1] = np.inf
    pa = np.histogram(a, e)[0] / len(a) + 1e-4; pb = np.histogram(b, e)[0] / len(b) + 1e-4
    return ((pb - pa) * np.log(pb / pa)).sum()
print(f"{'feature':24s} {'val mean':>9s} {'pub':>9s} {'priv':>9s} | val q01 q50 q99 | pub q01 q50 q99 | priv q01 q50 q99 | PSI pub priv | val min max | test min max")
for k in out['val']:
    v = out['val'][k]; a = out['test'][k][pub]; b = out['test'][k][~pub]
    q = lambda x: ' '.join(f'{z:8.3f}' for z in np.quantile(x, [.01, .5, .99]))
    print(f'{k:24s} {v.mean():9.3f} {a.mean():9.3f} {b.mean():9.3f} | {q(v)} | {q(a)} | {q(b)} | {psi(v, a):.3f} {psi(v, b):.3f} | {v.min():.3f} {v.max():.3f} | {out["test"][k].min():.3f} {out["test"][k].max():.3f}')
ks = ['M', 'H', 'uid0', 'n_train_feeds', 'cold_share', 'coldcand_share', 'cold_luo', 'mlog_luo', 'n_cat', 'max_cat_share', 'hist_n_cat', 'hist_top_cat_share', 'cand_cat_in_hist_share', 'f4_mean', 'f4_med']
X = np.stack([np.log(out['val'][k]) if k == 'M' else np.log1p(out['val'][k]) if k in ('H', 'n_train_feeds') else out['val'][k] for k in ks], 1)
C = np.corrcoef(X.T); np.set_printoptions(linewidth=250, precision=2, suppress=True); print(ks); print(C)
np.savez(CACHE / 'feat_raw.npz', **{f'{sp}__{k}': out[sp][k] for sp in out for k in out[sp]})
