"""Are exploration feeds user-bucketed? Compute proxy z for train feeds of users that appear in val/test vs a random control.
Also characterise likely-random vs likely-production feeds from the 04b mixture posterior."""
from featlib import *
import lightgbm as lgb, time, json
nb.set_num_threads(40)
from importlib import import_module
p2 = import_module('02_position')
t0 = time.time()
F = feeds('train')
uv = set(feeds('val')['user_id'].to_list()); ut = set(feeds('test')['user_id'].to_list())
users = F['user_id'].to_numpy()
inv = np.isin(users, np.array(sorted(uv))); intest = np.isin(users, np.array(sorted(ut)))
print('train feeds of val users', inv.sum(), 'of test users', intest.sum())
z04 = np.load(f'{OUT}/04b_detect.npz'); ctrl = z04['sel']
sel = np.where(inv | intest)[0]
sel = np.setdiff1d(sel, ctrl)
df, off, M, sel = build('train', sel)
df = add_hist_sim(df, 'train', sel, off, M)
# item TE must come from part A of 04b: rebuild it
selA = subsample_feeds(F, 600_000, seed=11)[:200_000]
dA, offA, MA = gather('train', selA, F, ['item_id', 'pos'])
tA = pl.DataFrame({'item_id': dA['item_id'], 'pos': dA['pos'], 'fid': np.repeat(np.arange(len(selA)), MA), 'M': np.repeat(MA, MA)})
tA = tA.with_columns(((pl.col('pos').rank('ordinal').over('fid') - 1) / (pl.col('M') - 1)).alias('prank'))
te_item = tA.group_by('item_id').agg(pl.col('prank').mean().alias('item_prank'), pl.len().alias('item_n'))
df = df.join(te_item, on='item_id', how='left').sort('fid', maintain_order=True)
bst = lgb.Booster(model_file=f'{OUT}/04b_posproxy.txt')
FEAT = bst.feature_name()
pred = bst.predict(df.select(FEAT).to_numpy().astype(np.float32), num_threads=40)
rho, *_ = p2.per_feed_stats(df['pos'].to_numpy().astype(np.float64), pred[None, :].astype(np.float64), df['target'].to_numpy(), off.astype(np.int64), M.astype(np.int64))
z = rho[:, 0] * np.sqrt(M - 1)
zc = z04['z']; Mc = z04['M']
res = {}
for lo, hi in [(20, 50), (50, 100), (100, 301)]:
    a = (M >= lo) & (M < hi); b = (Mc >= lo) & (Mc < hi)
    for nm, zz, mm in [('val/test users', z, a), ('control', zc, b)]:
        q = np.nanquantile(zz[mm], [.05, .1, .25, .5]).round(2).tolist()
        print(f'M[{lo},{hi}) {nm}: n={mm.sum()} P(z<0)={np.nanmean(zz[mm] < 0):.4f} quantiles {q}')
        res[f'{nm}_M{lo}_{hi}'] = dict(n=int(mm.sum()), P_z_lt0=float(np.nanmean(zz[mm] < 0)), q=q)
np.savez(f'{OUT}/04c_uservt.npz', z=z, M=M, sel=sel)
json.dump(res, open(f'{OUT}/04c_user_bucket.json', 'w'), indent=1)
