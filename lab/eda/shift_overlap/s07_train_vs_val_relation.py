"""Train (production, position-biased) vs val (random exploration): label-feature relationship shift, and whether
within-feed normalisation reduces covariate shift (adversarial AUC raw vs within-feed rank / z-score)."""
from common import *
from featlib import subset
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(7)
ftr = feeds('train')
dd = ftr['date'].cast(pl.Int32).to_numpy()
sel = rng.choice(np.where(dd >= dd.max() - 3)[0], 60000, replace=False)
T = subset('train', sel); V = subset('val'); Te = subset('test')
pos = np.asarray(cand('train', 'pos'))
# map pos for subset
f = T['f']; M = T['M']; co = f['cand_off'].to_numpy()
cidx = np.repeat(co, M) + (np.arange(M.sum()) - np.repeat(np.cumsum(M) - M, M))
T['C']['pos'] = pos[cidx].astype(np.float64)
def offs(Mx): return np.r_[0, np.cumsum(Mx)[:-1]]
import os
SKIP_DESC = os.environ.get('SKIP_DESC') == '1'  # descriptive part already logged in out/s07_desc.log
if not SKIP_DESC:
  print('==== single-feature NDCG: train sample (60k feeds, Sep 17-19) vs val; random tie-break averaged over 3 seeds')
  def nd(d, x):
      t = d['C']['target'].astype(float); Mx = d['M']; o = offs(Mx)
      return np.mean([ndcg(x + rng.random(len(x)) * 1e-7, t, o, Mx) for _ in range(3)])
  for c in ['f1', 'f2', 'f3', 'f4', 'f5']:
      a = nd(T, T['C'][c].astype(float)); b = nd(V, V['C'][c].astype(float))
      a2 = nd(T, -T['C'][c].astype(float)); b2 = nd(V, -V['C'][c].astype(float))
      print(f'  {c}: train {a:.4f} (neg {a2:.4f}) | val {b:.4f} (neg {b2:.4f})')
  print(f'  random: train {nd(T, rng.random(len(T["cf"]))):.4f} | val {nd(V, rng.random(len(V["cf"]))):.4f}')
  print(f'  train -pos (production order): {nd(T, -T["C"]["pos"]):.4f}')
  print(f'  train mean M {T["M"].mean():.1f} pos/feed {np.bincount(T["cf"], weights=T["C"]["target"]).mean():.3f} rate {T["C"]["target"].mean()*100:.2f}% | val M {V["M"].mean():.1f} pos/feed {np.bincount(V["cf"], weights=V["C"]["target"]).mean():.3f} rate {V["C"]["target"].mean()*100:.2f}%')
  # NDCG of f1 restricted to similar M
  # positive rate by feature decile (global deciles from val)
  print('\n==== positive rate by global decile (val-defined edges): train | val  (ratio to split mean)')
  for c in ['f1', 'f3', 'f4']:
      e = np.quantile(V['C'][c], np.linspace(0, 1, 11)[1:-1])
      bt = np.digitize(T['C'][c], e); bv = np.digitize(V['C'][c], e)
      rt = np.bincount(bt, weights=T['C']['target'], minlength=10) / np.bincount(bt, minlength=10) / T['C']['target'].mean()
      rv = np.bincount(bv, weights=V['C']['target'], minlength=10) / np.bincount(bv, minlength=10) / V['C']['target'].mean()
      print(f'  {c} train: {np.round(rt,2).tolist()}\n  {c} val  : {np.round(rv,2).tolist()}')
  # position bias in train: positive rate by pos bucket
  pb = np.minimum(T['C']['pos'] // 10, 30).astype(int)
  r = np.bincount(pb, weights=T['C']['target'], minlength=31) / np.maximum(np.bincount(pb, minlength=31), 1)
  print('\ntrain positive rate by pos bucket (0-9,10-19,...):', np.round(r[:16] * 100, 2).tolist())
  # within-feed Spearman(pos, f1) distribution in train -> how many feeds look unranked?
  df = pl.DataFrame({'f': T['cf'], 'pos': T['C']['pos'], 'f1': T['C']['f1']}).with_columns(pl.col('pos').rank().over('f').alias('rp'), pl.col('f1').rank().over('f').alias('r1'))
  sp = df.group_by('f').agg(pl.corr('rp', 'r1').alias('rho'), pl.len().alias('m')).filter(pl.col('m') >= 20)['rho'].drop_nans().to_numpy()
  print(f'train within-feed Spearman(pos, f1) over feeds with M>=20: mean {sp.mean():.3f}, q10/50/90 {np.quantile(sp,[.1,.5,.9]).round(3).tolist()}, share |rho|<0.1 {np.mean(np.abs(sp)<0.1)*100:.1f}%')
  # pos gaps: fraction of max(pos)+1 vs M
  mx = np.maximum.reduceat(T['C']['pos'], offs(T['M']))
  print(f'train: M/(max pos+1) mean {np.mean(T["M"]/(mx+1)):.3f}; share feeds with no gaps {np.mean(T["M"]==mx+1)*100:.1f}%')

# ---- normalisation vs covariate shift: adversarial train vs val/test at candidate level
def norm_feats(d, mode):
    C = d['C']; cf = d['cf']; Mx = d['M']
    X = {}
    for c in ['f1', 'f2', 'f3', 'f4', 'f5']:
        v = C[c].astype(np.float64)
        if mode == 'raw':
            X[c] = v
        elif mode == 'rank':
            r = pl.DataFrame({'f': cf, 'v': v}).with_columns(pl.col('v').rank('average').over('f').alias('r'))['r'].to_numpy()
            X[c] = (r - 1) / np.maximum(Mx[cf] - 1, 1)
        elif mode == 'z':
            m1 = np.bincount(cf, weights=v) / Mx; m2 = np.bincount(cf, weights=v * v) / Mx
            sd = np.sqrt(np.maximum(m2 - m1 ** 2, 1e-12))
            X[c] = (v - m1[cf]) / sd[cf]
        elif mode == 'raw+rank':
            X[c] = v
            r = pl.DataFrame({'f': cf, 'v': v}).with_columns(pl.col('v').rank('average').over('f').alias('r'))['r'].to_numpy()
            X[c + '_r'] = (r - 1) / np.maximum(Mx[cf] - 1, 1)
    return pl.DataFrame(X)
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
def adv(Xa, Xb, ga, gb, n=200000):
    ia = rng.choice(len(Xa), min(n, len(Xa)), replace=False); ib = rng.choice(len(Xb), min(n, len(Xb)), replace=False)
    X = pl.concat([Xa[ia], Xb[ib]]).to_pandas(); y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]; g = np.r_[ga[ia], gb[ib] + 10**8]
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = lgb.train(PAR, lgb.Dataset(X.iloc[tr], y[tr]), 150); oof[te] = m.predict(X.iloc[te])
    return roc_auc_score(y, oof)
print('\n==== candidate-level adversarial AUC (f1..f5 only) by normalisation')
for mode in ['raw', 'rank', 'z']:
    Xt, Xv, Xs = norm_feats(T, mode), norm_feats(V, mode), norm_feats(Te, mode)
    print(f'  {mode:5s}: train-vs-val {adv(Xt, Xv, T["cf"], V["cf"]):.4f}   val-vs-test {adv(Xv, Xs, V["cf"], Te["cf"]):.4f}', flush=True)
# value of within-feed normalisation for ranking on val itself (single feature)
print('\n==== val NDCG: raw vs within-feed rank vs z (identical orderings per feed => identical NDCG; shown as sanity) and pooled LGB on val 5-fold')
def lgb_val(X, d, rounds=200):
    t = d['C']['target']; g = d['cf']; Xp = X.to_pandas(); oof = np.zeros(len(t))
    for tr, te in GroupKFold(5).split(Xp, t, g):
        m = lgb.train(dict(PAR, min_data_in_leaf=100), lgb.Dataset(Xp.iloc[tr], t[tr]), rounds); oof[te] = m.predict(Xp.iloc[te])
    return ndcg_per_feed(oof + rng.random(len(oof)) * 1e-9, t.astype(float), offs(d['M']), d['M'])
res = {}
for mode in ['raw', 'rank', 'z', 'raw+rank']:
    res[mode] = lgb_val(norm_feats(V, mode), V)
    print(f'  pointwise LGB on val (5-fold by feed), f1..f5 {mode:8s}: NDCG {res[mode].mean():.4f}', flush=True)
for mode in ['rank', 'z', 'raw+rank']:
    d = res[mode] - res['raw']
    print(f'  paired {mode} - raw: {d.mean():+.4f} +- {d.std()/np.sqrt(len(d)):.4f}')
# train-fitted model evaluated on val: raw vs rank (does normalisation help transfer from position-biased train?)
print('\n==== LGB trained on train sample (60k feeds), evaluated on val: raw vs rank vs raw+rank (f1..f5 + category)')
for mode in ['raw', 'rank', 'raw+rank']:
    Xt = norm_feats(T, mode).with_columns(pl.Series('cat', T['C']['category_id'].astype(np.int32)))
    Xv = norm_feats(V, mode).with_columns(pl.Series('cat', V['C']['category_id'].astype(np.int32)))
    m = lgb.train(dict(PAR, min_data_in_leaf=200), lgb.Dataset(Xt.to_pandas(), T['C']['target'], categorical_feature=['cat']), 300)
    p = m.predict(Xv.to_pandas())
    s = ndcg_per_feed(p + rng.random(len(p)) * 1e-9, V['C']['target'].astype(float), offs(V['M']), V['M'])
    res['tr_' + mode] = s
    print(f'  {mode:8s}: val NDCG {s.mean():.4f}', flush=True)
for mode in ['rank', 'raw+rank']:
    d = res['tr_' + mode] - res['tr_raw']; print(f'  paired {mode} - raw: {d.mean():+.4f} +- {d.std()/np.sqrt(len(d)):.4f}')
