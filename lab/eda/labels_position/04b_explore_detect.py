"""Stronger production-ranker proxy (adds item-level position target encoding + history similarity) and a
2-component mixture fit of per-feed z = rho(pred,pos)*sqrt(M-1) with the random component fixed to N(0,1)."""
from featlib import *
import lightgbm as lgb, time, json
from scipy.stats import norm
nb.set_num_threads(40)
from importlib import import_module
p2 = import_module('02_position')
t0 = time.time()
F = feeds('train')
sel = subsample_feeds(F, 600_000, seed=11)
df, off, M, sel = build('train', sel)
df = add_hist_sim(df, 'train', sel, off, M)
print('built', df.shape, time.time() - t0, flush=True)
df = df.with_columns(((pl.col('pos').rank('ordinal').over('fid') - 1) / (pl.col('M') - 1)).alias('prank'))
n = len(sel); A = n // 3; B = 2 * n // 3
# item-level TE of prank from part A (feeds 0..A)
te_item = (df.filter(pl.col('fid') < A).group_by('item_id').agg(pl.col('prank').mean().alias('item_prank'), pl.len().alias('item_n')))
df = df.join(te_item, on='item_id', how='left').sort(['fid'], maintain_order=True)
print('item TE coverage in C:', df.filter(pl.col('fid') >= B)['item_prank'].is_not_null().mean())
FEAT = ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id', 'dot', 'cos', 'inorm', 'hcat_cnt', 'hcat_share', 'hcat_recent', 'hcat_cnt_ne1',
        'f1_r', 'f3_r', 'f4_r', 'f5_r', 'dot_r', 'M', 'H', 'hsim_max', 'hsim_mean', 'hsim_max_ne1', 'item_prank', 'item_n']
trd = df.filter((pl.col('fid') >= A) & (pl.col('fid') < B)); ted = df.filter(pl.col('fid') >= B)
ds = lgb.Dataset(trd.select(FEAT).to_numpy().astype(np.float32), trd['prank'].to_numpy(), feature_name=FEAT, categorical_feature=['category_id'])
params = dict(objective='regression', learning_rate=0.1, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.5, bagging_freq=1, num_threads=40, verbose=-1)
bst = lgb.train(params, ds, num_boost_round=400)
pred = bst.predict(ted.select(FEAT).to_numpy().astype(np.float32), num_threads=40)
imp = dict(zip(FEAT, (bst.feature_importance('gain') / bst.feature_importance('gain').sum()).round(4).tolist()))
print('trained', time.time() - t0, imp, flush=True)
Mte = M[B:]; offte = np.concatenate([[0], np.cumsum(Mte)[:-1]]).astype(np.int64)
posv = ted['pos'].to_numpy().astype(np.float64)
feats = np.nan_to_num(np.stack([pred, ted['hsim_max'].to_numpy(), ted['item_prank'].to_numpy()]).astype(np.float64), nan=0.5)
rho, rho_t, *_ = p2.per_feed_stats(posv, feats, ted['target'].to_numpy(), offte, Mte.astype(np.int64))
z = rho[:, 0] * np.sqrt(Mte - 1)
print('mean rho (pred, hsim_max, item_prank):', np.nanmean(rho, 0))
res = {'imp': imp, 'mean_rho': np.nanmean(rho, 0).tolist(), 'bins': []}

def em(zz, iters=300):
    # mixture: pi*N(0,1) + (1-pi)*N(mu,s^2)
    pi, mu, s = 0.2, np.median(zz), np.std(zz)
    for _ in range(iters):
        a = pi * norm.pdf(zz); b = (1 - pi) * norm.pdf(zz, mu, s)
        r = a / (a + b)
        pi = r.mean(); w = 1 - r
        mu = (w * zz).sum() / w.sum(); s = np.sqrt((w * (zz - mu) ** 2).sum() / w.sum())
    ll = np.log(pi * norm.pdf(zz) + (1 - pi) * norm.pdf(zz, mu, s)).sum()
    ll1 = np.log(norm.pdf(zz, zz.mean(), zz.std())).sum()
    return pi, mu, s, ll - ll1, r

post = np.full(len(z), np.nan)
for lo, hi in [(10, 20), (20, 50), (50, 100), (100, 200), (200, 301)]:
    m = (Mte >= lo) & (Mte < hi) & ~np.isnan(z)
    zz = z[m]
    pi, mu, s, dll, r = em(zz)
    post[np.where(m)[0]] = r
    h, _ = np.histogram(zz, np.linspace(-4, 10, 29))
    b = dict(M=f'[{lo},{hi})', n=int(m.sum()), pi_random=float(pi), mu_prod=float(mu), sd_prod=float(s), dLL_vs_1gauss=float(dll),
             P_z_lt0=float(np.mean(zz < 0)), z_q=np.quantile(zz, [.01, .05, .1, .25, .5, .75]).round(2).tolist(), hist=(h / h.sum() * 1000).astype(int).tolist())
    res['bins'].append(b); print(b, flush=True)
np.savez(f'{OUT}/04b_detect.npz', z=z, rho=rho, M=Mte, sel=sel[B:], post=post)
json.dump(res, open(f'{OUT}/04b_detect.json', 'w'), indent=1)
bst.save_model(f'{OUT}/04b_posproxy.txt')
