"""Detect exploration (random-order) feeds in train: learn a proxy of the production ranker (predict within-feed pos rank
from candidate features), then look at per-feed Spearman(pred, pos) distribution. Random feeds => z ~ N(0,1)."""
from featlib import *
import lightgbm as lgb, time
nb.set_num_threads(40)
from importlib import import_module
p2 = import_module('02_position')
t0 = time.time()
F = feeds('train')
sel = subsample_feeds(F, 400_000, seed=7)
df, off, M, sel = build('train', sel)
print('built', df.shape, time.time() - t0, flush=True)
# within feed rank of pos normalised
df = df.with_columns(((pl.col('pos').rank('ordinal').over('fid') - 1) / (pl.col('M') - 1)).alias('prank'))
FEAT = ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id', 'dot', 'cos', 'inorm', 'hcat_cnt', 'hcat_share', 'hcat_recent', 'hcat_cnt_ne1',
        'f1_r', 'f3_r', 'f4_r', 'f5_r', 'dot_r', 'M', 'H']
nfe = len(sel); half = nfe // 2
tr = df.filter(pl.col('fid') < half); te = df.filter(pl.col('fid') >= half)
X = tr.select(FEAT).to_numpy().astype(np.float32); y = tr['prank'].to_numpy()
ds = lgb.Dataset(X, y, feature_name=FEAT, categorical_feature=['category_id'], free_raw_data=False)
params = dict(objective='regression', learning_rate=0.1, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.5, bagging_freq=1, num_threads=40, verbose=-1)
bst = lgb.train(params, ds, num_boost_round=300)
pred = bst.predict(te.select(FEAT).to_numpy().astype(np.float32), num_threads=40)
imp = dict(zip(FEAT, bst.feature_importance('gain').round(0).tolist()))
print('trained', time.time() - t0, imp, flush=True)
# per feed spearman pred vs pos on held-out feeds
Mte = M[half:]; offte = np.concatenate([[0], np.cumsum(Mte)[:-1]]).astype(np.int64)
posv = te['pos'].to_numpy().astype(np.float64)
feats = np.stack([pred, te['f1'].to_numpy(), te['dot'].to_numpy(), te['f4'].to_numpy()]).astype(np.float64)
feats = np.nan_to_num(feats, nan=-1e9)
rho, rho_t, *_ = p2.per_feed_stats(posv, feats, te['target'].to_numpy(), offte, Mte.astype(np.int64))
z = rho * np.sqrt(Mte[:, None] - 1)
np.savez(f'{OUT}/04_detect.npz', rho=rho, z=z, M=Mte, sel=sel[half:])
print('mean rho (pred,f1,dot,f4):', np.nanmean(rho, 0), 'median', np.nanmedian(rho, 0))
bins = np.linspace(-1, 1, 41)
for lo, hi in [(5, 20), (20, 50), (50, 100), (100, 200), (200, 301)]:
    m = (Mte >= lo) & (Mte < hi)
    h, _ = np.histogram(rho[m, 0], bins)
    print(f'M[{lo},{hi}) n={m.sum()} rho_pred hist(x1000):', (h / h.sum() * 1000).astype(int).tolist())
    zz = z[m, 0]
    print('   frac |z|<1:', np.mean(np.abs(zz) < 1).round(4), ' frac z<1 (i.e. not clearly ranked):', np.mean(zz < 1).round(4),
          ' frac z<0:', np.mean(zz < 0).round(4), ' z quantiles', np.quantile(zz[~np.isnan(zz)], [.01, .05, .1, .25, .5]).round(2))
