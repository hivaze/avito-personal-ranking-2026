"""Power check: start from a deliberately weaker base (s_to alone, -0.0011) and see whether the residual
learner recovers the gain when s_ft is among its features. Tells whether the residual method can find real
effects of this size at all."""
import sys
import lightgbm as lgb, numpy as np
import prep
from prep import kit
obj = sys.argv[1]
df = prep.derive(kit.load())
names = [c for c in prep.feature_names(df) if c not in ("score",)]
X = prep.matrix(df, names); y = df['label'].to_numpy().astype(float)
s = df['s_to'].to_numpy().astype(float); s0 = df['score'].to_numpy().astype(float)
off, sz = kit.feed_layout(df); fold = kit.folds(df); ffold = fold[off]
CHECK = (10, 25, 50, 100, 200)
oof = {t: np.zeros(len(y)) for t in CHECK}
for k in range(5):
    tr = fold != k; trf = ffold != k; toff = np.r_[0, np.cumsum(sz[trf])[:-1]]
    params = dict(num_leaves=7, learning_rate=0.05, min_data_in_leaf=2000, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=10, num_threads=12, verbose=-1, seed=k)
    if obj == 'softmax':
        fid_tr = np.repeat(np.arange(trf.sum()), sz[trf]); npos = np.add.reduceat(y[tr], toff); yy = y[tr]
        def fobj(pred, data):
            m = np.maximum.reduceat(pred, toff)[fid_tr]; e = np.exp(pred - m); pz = e / np.add.reduceat(e, toff)[fid_tr]
            w = 1.0 / np.maximum(npos, 1)[fid_tr]
            return (pz * npos[fid_tr] - yy) * w, np.maximum(pz * (1 - pz) * npos[fid_tr] * w, 1e-6)
        params.update(objective=fobj, min_sum_hessian_in_leaf=5.0); init = s
    elif obj == 'binary':
        from sklearn.linear_model import LogisticRegression
        lrm = LogisticRegression(C=1e6).fit(s[tr, None], y[tr]); init = lrm.intercept_[0] + lrm.coef_[0, 0] * s
        params.update(objective='binary')
    else:
        params.update(objective='lambdarank', lambdarank_truncation_level=300, label_gain=[0, 1]); init = s
    bst = lgb.train(params, lgb.Dataset(X[tr], y[tr], init_score=init[tr], group=sz[trf] if obj != 'binary' else None), num_boost_round=max(CHECK))
    te = fold == k
    for t in CHECK:
        oof[t][te] = init[te] + bst.predict(X[te], num_iteration=t, raw_score=True)
print('s_to vs score', kit.evaluate(df, s)['delta'])
for t in CHECK:
    r = kit.evaluate(df, oof[t]); print(obj, t, f"delta vs score={r['delta']:+.5f} se={r['se']:.5f}")
