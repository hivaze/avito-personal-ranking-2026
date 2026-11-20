"""POST-HOC diagnostic (not a candidate): why does the residual GBDT lose 0.003-0.005? Fold 0 of the FT folds, seed 1.
In-sample (training feeds) vs held-out gain, per objective / init form, B0 and B2 inputs (full form)."""
from lib import *
import time
import lightgbm as lgb
from feats import build

T0 = time.time()
V = S('val')
log = open(OUT / 'e4_diag.txt', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
ref = V.ev(np.load(CACHE / 'val_g3.npy'))
k = 0; tr = V.fold != k; te = V.fold == k; rtr = np.repeat(tr, V.M); rte = np.repeat(te, V.M)
for arm in ('B0', 'B2'):
    X, names, pool = build(V, arm, 'full')
    for nm, obj, init_c, lr, extra in [('lambdarank c2 lr.03 (declared)', 'lambdarank', 2.0, 0.03, {}),
                                       ('lambdarank c2 lr.03 norm off', 'lambdarank', 2.0, 0.03, {'lambdarank_norm': False}),
                                       ('lambdarank c5 lr.03', 'lambdarank', 5.0, 0.03, {}),
                                       ('rank_xendcg c2 lr.03', 'rank_xendcg', 2.0, 0.03, {}),
                                       ('lambdarank no-init lr.03', 'lambdarank', 0.0, 0.03, {}),
                                       ('lambdarank c2 lr.01', 'lambdarank', 2.0, 0.01, {})]:
        p = dict(objective=obj, lambdarank_truncation_level=300, learning_rate=lr, num_leaves=15, min_data_in_leaf=1000, lambda_l2=100.0,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=1, num_threads=12, verbose=-1, metric='none', deterministic=True,
                 force_row_wise=True, **extra)
        ds = lgb.Dataset(X[rtr], V.y[rtr], group=V.M[tr], init_score=init_c * pool[rtr] if init_c else None, feature_name=names)
        b = lgb.train(p, ds, num_boost_round=300)
        out = []
        for r in (10, 30, 100, 300):
            sc = init_c * pool + b.predict(X, num_iteration=r)
            n = V.ev(sc)
            out.append(f'r{r}: in {(n - ref)[tr].mean():+.5f} out {(n - ref)[te].mean():+.5f}')
        pr(f'{arm} {nm:34s} ' + ' | '.join(out) + f'  [{time.time()-T0:.0f}s]')
