"""NDCG-optimal key = log P(y_i=1) + log E[1/IDCG(npos) | y_i=1, x_i]. Learn the second term from positives
(LightGBM regression of log(1/IDCG) on feature columns, cross-fitted), add m * (prediction) to the score with m on
a grid chosen inside the cross-fit. m=1 is the theoretical weight if score were exactly log P."""
import numpy as np, polars as pl, lightgbm as lgb
import prep
from prep import kit
df = prep.derive(kit.load())
y = df['label'].to_numpy().astype(float); s = df['score'].to_numpy().astype(float)
off, sz = kit.feed_layout(df); fold = kit.folds(df); ff = fold[off]
npos = np.repeat(np.add.reduceat(y, off), sz)
cs = np.r_[0, np.cumsum(1 / np.log2(np.arange(2, 400))) ]
liw = -np.log(cs[np.maximum(npos.astype(int), 1)])
names = [c for c in prep.feature_names(df) if c not in ('s_ft', 'br_diff')]
X = prep.matrix(df, names)
pred = np.zeros(len(y))
for k in range(5):
    tr = (fold != k) & (y == 1)
    bst = lgb.train(dict(objective='regression', num_leaves=7, learning_rate=0.03, min_data_in_leaf=200, lambda_l2=10,
                         feature_fraction=0.8, num_threads=12, verbose=-1), lgb.Dataset(X[tr], liw[tr]), 300)
    te = fold == k
    pred[te] = bst.predict(X[te])
    # quality on held-out positives
    tp = te & (y == 1)
    print(k, 'R2 on held-out positives', 1 - ((liw[tp] - pred[tp]) ** 2).mean() / liw[tp].var(), flush=True)
imp = bst.feature_importance('gain'); o = np.argsort(-imp)[:10]
print('top features', [(names[i], round(imp[i] / imp.sum(), 3)) for i in o])
F = prep.FastNDCG(y.astype(int), off, sz); b = F(s)
# within-feed centred prediction (feed-constant parts do not reorder)
Wm = [-0.5, -0.25, 0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
nd = np.stack([F(s + m * pred + 1e-7 * s) - b for m in Wm])
oof = np.zeros(len(sz)); picks = []
for k in range(5):
    t = ff != k; j = int(np.argmax(nd[:, t].mean(1))); picks.append(Wm[j]); oof[~t] = nd[j, ~t]
print('per-m in-sample', dict(zip(Wm, np.round(nd.mean(1), 5))))
print(f"iw_key oof={oof.mean():+.5f} se={oof.std(ddof=1) / np.sqrt(len(oof)):.5f} picks={picks}")
j1 = Wm.index(1.0); d1 = nd[j1]
print(f"theory m=1 (no fit): delta={d1.mean():+.5f} se={d1.std(ddof=1) / np.sqrt(len(d1)):.5f}")
