"""Cross-fitted residual GBM, pointwise logistic with clogit offset, small steps."""
import hi, numpy as np, polars as pl, lightgbm as lgb, sys
D = hi.Data(); df = D.df
lens = ['h1', 'h5', 'h10', 'hmean5', 'c_contact', 'c_order', 'c_fav', 'c_lead', 'c_vfeed', 'c_vsearch',
        'cat_newest_idx', 'same_cat_newest', 'newest_type', 'newest_src', 'hmax', 'hmax_idx', 'hmax_type', 'hmax_src']
ctx = ['rank', 'M', 'score', 'gap_top']
allc = [c for c in df.columns if c not in ('label', 'fidx', 'user_id', 'item_id', 'row0', 'row', 'gap_next', 's_ft', 's_to')]
sets = {'lens': lens + ctx, 'all': allc + ['s_ft', 's_to']}
E = hi.clogit_expect(np.c_[D.s], np.array([0.9825]), D.y, D.off, D.sz)
E = np.clip(E, 1e-5, 0.9)
offs = np.log(E / (1 - E))
for name in sys.argv[1:]:
    cols = sets[name]
    X = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    for nl, ne in [(4, 100), (8, 200), (16, 300)]:
        oofs = {}
        for k in range(5):
            tr = D.fold_row != k; te = ~tr
            m = lgb.LGBMClassifier(objective='binary', n_jobs=12, verbose=-1, num_leaves=nl, learning_rate=0.02,
                                   n_estimators=ne, min_child_samples=1000, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, reg_lambda=20)
            m.fit(X[tr], D.y[tr], init_score=offs[tr])
            for nt in [ne // 4, ne // 2, ne]:
                oofs.setdefault(nt, np.zeros(len(D.s)))[te] = D.s[te] + m.predict(X[te], raw_score=True, num_iteration=nt)
        for nt, o in oofs.items():
            d, se = D.delta(o)
            print(f'{name:6s} leaves {nl} trees {nt}: oof delta {d:+.5f} se {se:.5f} z {d/se:.2f}', flush=True)
