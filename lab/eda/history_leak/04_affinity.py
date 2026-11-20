"""4) Non-leak history features: candidate category vs history category affinity. Single-feature val NDCG,
combination with f1, and a small LightGBM (train sample -> val) to measure the incremental value of history features."""
import numpy as np
import polars as pl
import lightgbm as lgb
from common import load_split, eval_score, feeds, arr, ndcg_per_feed
from hist_feats import compute_aff, FEATS

pl.Config.set_tbl_rows(80)


def main():
    d = load_split('val', cand_cols=('item_id', 'category_id', 'f1', 'f2', 'f3', 'f4', 'f5', 'target'))
    t = d['c_target']
    X, _, _ = compute_aff('val')
    np.save('<repo>/eda/history_leak/out_aff_val.npy', X)
    f1 = d['c_f1'].astype(np.float64)
    rng = np.random.default_rng(0)
    base = eval_score(f1, d, 'f1', verbose=True)
    rnd = eval_score(rng.random(len(t)), d, 'random', verbose=True)
    print('\n--- single history features (higher = better, sign chosen by direction) ---')
    res = []
    for j, n in enumerate(FEATS):
        x = X[:, j].astype(np.float64)
        sgn = -1.0 if n in ('cat_minpos', 'cat_rank') else 1.0
        # tie-break by tiny random noise to avoid pessimistic-tie penalty dominating count features
        pf = ndcg_per_feed(sgn * x + 1e-9 * rng.random(len(x)), t, d['feeds']['cand_off'].to_numpy(), d['feeds']['M'].to_numpy())
        pfp = ndcg_per_feed(sgn * x, t, d['feeds']['cand_off'].to_numpy(), d['feeds']['M'].to_numpy())
        res.append((n, pf.mean(), pfp.mean(), (pf - rnd).mean() / ((pf - rnd).std() / np.sqrt(len(pf)))))
    print(pl.DataFrame(res, schema=['feature', 'ndcg_randtie', 'ndcg_pessimistic', 't_vs_random'], orient='row'))
    # target rate by cat_cnt buckets
    df = pl.DataFrame({'t': t, 'cat_cnt': X[:, 1], 'fc': X[:, 17], 'frac': X[:, 2]})
    print(df.with_columns(pl.col('cat_cnt').cut([0, 1, 3, 10, 30, 100], left_closed=False).alias('b')).group_by('b').agg(
        pl.len(), pl.col('t').mean().alias('target_rate')).sort('b'))
    print(df.with_columns(pl.col('fc').cut([0, 1, 3, 10, 30], left_closed=False).alias('b')).group_by('b').agg(
        pl.len(), pl.col('t').mean().alias('target_rate')).sort('b'))
    # combos with f1: within-feed rank blending
    fidx = d['c_fidx']
    def prank(x):
        return pl.DataFrame({'f': fidx, 'x': x}).select((pl.col('x').rank('average').over('f') / pl.len().over('f')))['x'].to_numpy()
    rf1 = prank(f1)
    print('\n--- f1 + w * rank(feature) ---')
    for n in ['cat_frac', 'cat_rec95', 'cat_cnt_feedclick', 'cat_rec_feedclick', 'cat_frac_feedclick', 'cat_cnt']:
        x = X[:, FEATS.index(n)].astype(np.float64)
        rx = prank(x)
        for w in [0.25, 0.5, 1.0]:
            eval_score(rf1 + w * rx, d, f'rank(f1)+{w}*rank({n})', base)
    # ---- LightGBM: train sample -> val
    print('\n--- LightGBM (train sample of 200k feeds, lambdarank) ---')
    ft = feeds('train')
    sel = np.sort(np.random.default_rng(1).choice(len(ft), 200_000, replace=False))
    Xtr, _, cidx = compute_aff('train', sel)
    fcols = ['f1', 'f2', 'f3', 'f4', 'f5']
    Ftr = np.stack([np.asarray(arr('train', 'cand_' + c)[cidx]) for c in fcols] +
                   [np.asarray(arr('train', 'cand_category_id')[cidx]).astype(np.float32)], 1)
    ytr = np.asarray(arr('train', 'cand_target')[cidx])
    gtr = ft.sort('fidx')['M'].to_numpy()[sel]
    Fva = np.stack([d['c_' + c] for c in fcols] + [d['c_category_id'].astype(np.float32)], 1)
    gva = d['feeds']['M'].to_numpy()
    params = dict(objective='lambdarank', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=300, num_threads=40, verbose=-1,
                  eval_at=[1000])
    out = {}
    for name, A, B in [('f1..f5+cat', Ftr, Fva), ('f1..f5+cat+hist', np.hstack([Ftr, Xtr]), np.hstack([Fva, X]))]:
        dtr = lgb.Dataset(A, ytr, group=gtr, categorical_feature=[5])
        m = lgb.train(params, dtr, num_boost_round=400)
        p = m.predict(B)
        out[name] = eval_score(p, d, f'LGB {name}', base)
        if 'hist' in name:
            imp = m.feature_importance('gain')
            names = fcols + ['cat'] + FEATS
            o = np.argsort(-imp)
            print('top gain:', [(names[i], round(imp[i] / imp.sum(), 3)) for i in o[:15]])
            np.save('<repo>/eda/history_leak/out_lgb_hist_val_pred.npy', p)
    diff = out['f1..f5+cat+hist'] - out['f1..f5+cat']
    print(f'hist features increment: {diff.mean():+.4f} (paired SE {diff.std()/np.sqrt(len(diff)):.4f})')


if __name__ == '__main__':
    main()
