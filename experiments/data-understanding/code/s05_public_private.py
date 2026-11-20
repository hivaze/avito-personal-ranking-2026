# Finds the public/private boundary of the test split (the earliest 30% of feeds by timestamp) and
# compares the two parts: days covered, feed size, user overlap with train, share of candidate rows
# whose item was a train candidate. It also re-weights val NDCG by feed size to the public and the
# private mix of feed sizes.
# How it was run: from lab/eda/shift_overlap (imports common.py). The command line is not on record;
# the output is lab/eda/shift_overlap/out/s05.log.
# What it produced: boundary timestamp 1790497920 (2026-09-27 11:32 Moscow time), 8,562 public feeds,
# train-candidate coverage 56.41% public against 52.68% private (results/shift_s05_public_private.log).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/eda/shift_overlap/s05_public_private.py
"""Test public (first 30% feeds by timestamp) vs private (last 70%): boundary, composition, expected metric level."""
from common import *
import datetime as dt
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
NID = 111_400_000
F = {s: feeds(s) for s in SPLITS}
ft = F['test']
ts = ft['timestamp'].to_numpy()
o = np.argsort(ts, kind='stable')
n = len(ts)
for frac in [0.25, 0.3, 0.35]:
    k = int(round(frac * n)); b = ts[o[k]]
    print(f'{frac:.2f}: boundary ts {b} = {dt.datetime.fromtimestamp(int(b), dt.UTC) + dt.timedelta(hours=3):%Y-%m-%d %H:%M} MSK; feeds with ts<b {np.sum(ts<b)} ({np.mean(ts<b)*100:.2f}%), ts==b {np.sum(ts==b)}')
k30 = int(round(0.3 * n)); b30 = ts[o[k30]]
pub = ts < b30  # strict; ties at boundary are ambiguous (<=18 feeds)
print(f'using public := ts < {b30}  -> {pub.sum()} feeds ({pub.mean()*100:.2f}%)')
ft = ft.with_columns(pl.Series('public', pub))
# composition
g = ft.group_by('public').agg(pl.len().alias('feeds'), pl.col('M').mean().alias('M_mean'), pl.col('M').median().alias('M_med'),
                              pl.col('H').mean().alias('H_mean'), (pl.col('H') == 256).mean().alias('H256'), (pl.col('user_id') == 0).mean().alias('user0'),
                              pl.col('user_id').n_unique().alias('users'), pl.col('date').n_unique().alias('ndays')).sort('public')
print(g)
print('public covers', ft.filter(pl.col('public'))['date'].unique().sort().to_list(), ' private covers', ft.filter(~pl.col('public'))['date'].unique().sort().to_list())
# weekday/weekend share
dow = ft['date'].dt.weekday().to_numpy()  # 1=Mon..7=Sun
print('weekend (Sat/Sun) share: public %.1f%%, private %.1f%%, val %.1f%%' % (np.mean(dow[pub] >= 6) * 100, np.mean(dow[~pub] >= 6) * 100, np.mean(F['val']['date'].dt.weekday().to_numpy() >= 6) * 100))
# user overlap
tru = F['train']['user_id'].unique().to_numpy(); vu = F['val']['user_id'].unique().to_numpy()
u = ft['user_id'].to_numpy()
for nm, m in [('public', pub), ('private', ~pub)]:
    nz = u[m] != 0
    print(f'{nm}: feeds of users in train (excl user0) {np.isin(u[m][nz], tru).mean()*100:.2f}%  in val {np.isin(u[m][nz], vu).mean()*100:.2f}%')
pu, qu = set(u[pub]) - {0}, set(u[~pub]) - {0}
print(f'users in both public and private (excl user0): {len(pu & qu)} ({len(pu&qu)/len(pu|qu)*100:.2f}% of test users)')
# hours
hr = ft['hour'].to_numpy()
hp = np.bincount(hr[pub], minlength=24) / pub.sum(); hq = np.bincount(hr[~pub], minlength=24) / (~pub).sum()
print('JS(hour) pub vs priv %.4f' % js_div(hp, hq))
# M distribution
for nm, m in [('public', pub), ('private', ~pub)]:
    print(nm, 'M quantiles', np.quantile(ft['M'].to_numpy()[m], [.1, .25, .5, .75, .9]).tolist())
bins = np.arange(0, 310, 10)
print('JS(M) pub-priv %.4f, val-pub %.4f, val-priv %.4f' % (js_div(np.histogram(ft['M'].to_numpy()[pub], bins)[0], np.histogram(ft['M'].to_numpy()[~pub], bins)[0]),
      js_div(np.histogram(F['val']['M'].to_numpy(), bins)[0], np.histogram(ft['M'].to_numpy()[pub], bins)[0]),
      js_div(np.histogram(F['val']['M'].to_numpy(), bins)[0], np.histogram(ft['M'].to_numpy()[~pub], bins)[0])))
# categories (candidate level)
cat = np.asarray(cand('test', 'category_id')).astype(int); cp = np.repeat(pub, ft['M'].to_numpy())
cv = np.bincount(np.asarray(cand('val', 'category_id')).astype(int), minlength=64)
c1 = np.bincount(cat[cp], minlength=64); c0 = np.bincount(cat[~cp], minlength=64)
print('JS(cat) pub-priv %.5f  val-pub %.5f  val-priv %.5f' % (js_div(c1, c0), js_div(cv, c1), js_div(cv, c0)))
sh1, sh0, shv = c1 / c1.sum() * 100, c0 / c0.sum() * 100, cv / cv.sum() * 100
mv = sorted(range(64), key=lambda k: -abs(sh1[k] - sh0[k]))[:10]
print('largest abs share diffs (cat: public%, private%, val%):', [(k, round(sh1[k], 2), round(sh0[k], 2), round(shv[k], 2)) for k in mv])
# per-day composition in test incl. f-feature means and train coverage
trc = np.bincount(np.asarray(cand('train', 'item_id')), minlength=NID)
it = np.asarray(cand('test', 'item_id'))
print('train-candidate coverage: public %.2f%%  private %.2f%%' % (np.mean(trc[it[cp]] > 0) * 100, np.mean(trc[it[~cp]] > 0) * 100))
# expected NDCG level by M: per-feed f1 NDCG on val vs M bucket, re-weighted to public/private M distributions
fv = F['val']; Mv = fv['M'].to_numpy(); offv = fv['cand_off'].to_numpy(); tv = np.asarray(cand('val', 'target'))
f1v = np.asarray(cand('val', 'f1'), dtype=np.float64)
rng = np.random.default_rng(0)
nd_f1 = ndcg_per_feed(f1v + rng.random(len(f1v)) * 1e-7, tv, offv, Mv)
nd_rand = np.mean([ndcg_per_feed(rng.random(len(f1v)), tv, offv, Mv) for _ in range(10)], axis=0)
edges = np.array([5, 10, 20, 40, 70, 100, 150, 200, 250, 301])
bv = np.digitize(Mv, edges) - 1
Mt = ft['M'].to_numpy(); bt = np.digitize(Mt, edges) - 1
print('M bucket | val share | val rand NDCG | val f1 NDCG | pub share | priv share')
for k in range(len(edges) - 1):
    m = bv == k
    print(f'[{edges[k]},{edges[k+1]}) | {m.mean()*100:5.1f}% | {nd_rand[m].mean():.4f} | {nd_f1[m].mean():.4f} | {np.mean(bt[pub]==k)*100:5.1f}% | {np.mean(bt[~pub]==k)*100:5.1f}%')
mr = np.array([nd_rand[bv == k].mean() for k in range(len(edges) - 1)]); mf = np.array([nd_f1[bv == k].mean() for k in range(len(edges) - 1)])
for nm, m in [('val', None), ('public', pub), ('private', ~pub), ('test-all', np.ones(len(pub), bool))]:
    w = np.bincount(bv if m is None else bt[m], minlength=len(edges) - 1)[:len(edges) - 1]; w = w / w.sum()
    print(f'  M-reweighted expected NDCG on {nm:8s}: random {np.dot(w, mr):.4f}  f1 {np.dot(w, mf):.4f}')
# per-day val f1 NDCG (Tue vs Thu)
dv = fv['date'].cast(pl.Int32).to_numpy()
for d in np.unique(dv):
    m = dv == d
    if m.sum() > 100:
        print(f'val {np.datetime64(int(d),"D")}: feeds {m.sum()}, f1 NDCG {nd_f1[m].mean():.4f} +- {nd_f1[m].std()/np.sqrt(m.sum()):.4f}, random {nd_rand[m].mean():.4f}, pos/feed {np.add.reduceat(tv.astype(int), offv)[m].mean():.3f}')
# public-like / private-like split of val by time: first 30% vs last 70% of val
tvs = fv['timestamp'].to_numpy(); ov = np.argsort(tvs, kind='stable'); kv = int(0.3 * len(tvs)); pv = np.zeros(len(tvs), bool); pv[ov[:kv]] = True
d = nd_f1
print(f'val first30% f1 NDCG {d[pv].mean():.4f}  last70% {d[~pv].mean():.4f} (diff {d[pv].mean()-d[~pv].mean():+.4f}, SE ~{np.sqrt(d[pv].var()/pv.sum()+d[~pv].var()/(~pv).sum()):.4f})')
