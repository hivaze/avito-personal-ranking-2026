"""Cross-fitted residual GBM: does any combination of lens features improve the ranking beyond score?"""
import hi, numpy as np, polars as pl, lightgbm as lgb, sys
D = hi.Data(); df = D.df
lens = ['h1', 'h5', 'h10', 'hmean5', 'c_contact', 'c_order', 'c_fav', 'c_lead', 'c_vfeed', 'c_vsearch',
        'cat_newest_idx', 'same_cat_newest', 'newest_type', 'newest_src', 'hmax', 'hmax_idx', 'hmax_type', 'hmax_src']
ctx = ['rank', 'M', 'score', 'gap_top']
allc = [c for c in df.columns if c not in ('label', 'fidx', 'user_id', 'item_id', 'row0', 'row', 'gap_next', 's_ft', 's_to')]
sets = {'lens': lens + ctx, 'lens+sft': lens + ctx + ['s_ft', 's_to'], 'all': allc + ['s_ft', 's_to']}
which = sys.argv[1:] or list(sets)
for name in which:
    cols = sets[name]
    X = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    for params in [dict(num_leaves=7, learning_rate=0.03, n_estimators=150, min_child_samples=400),
                   dict(num_leaves=15, learning_rate=0.03, n_estimators=300, min_child_samples=400)]:
        oof = np.zeros(len(D.s))
        for k in range(5):
            tr = D.fold_row != k
            te = ~tr
            off_tr = np.r_[0, np.cumsum(D.sz[D.fold != k])]
            grp_tr = D.sz[D.fold != k]
            grp_te = D.sz[D.fold == k]
            m = lgb.LGBMRanker(objective='lambdarank', n_jobs=12, verbose=-1, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=10, lambdarank_truncation_level=300, **params)
            m.fit(X[tr], D.y[tr], group=grp_tr, init_score=D.s[tr])
            oof[te] = D.s[te] + m.predict(X[te], raw_score=True)
        d, se = D.delta(oof)
        print(f'{name:10s} {params} oof delta {d:+.5f} se {se:.5f} z {d/se:.2f}', flush=True)
