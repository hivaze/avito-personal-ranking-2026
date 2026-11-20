import hi, numpy as np, polars as pl, sys
D = hi.Data(); df = D.df
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_hide_column_data_types(True); pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_float_precision(2)
cols = ['rank', 'label', 'score', 's_ft', 'cat', 'ucos', 'h1', 'h5', 'hmean5', 'hmax', 'hmax_idx', 'hmax_type', 'hmax_src',
        'c_contact', 'c_fav', 'cat_newest_idx', 'cat_views', 'f4', 'xp_imp_c50', 'it_hist_all_luo']
y = D.y
best = np.array([D.col('rank')[o:o+m][y[o:o+m] > 0].min() if y[o:o+m].sum() > 0 else 0 for o, m in zip(D.off, D.sz)])
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
mode = sys.argv[1]
if mode == 'bad':
    cand = np.nonzero((best >= 30) & (D.sz <= 150))[0]
else:
    cand = np.nonzero((best >= 1) & (best <= 3) & (D.sz <= 150))[0]
for f in rng.choice(cand, 6, replace=False):
    o, m = D.off[f], D.sz[f]
    sub = df[o:o+m]
    h = sub.row(0, named=True)
    print(f"\n### feed {h['fidx']} M={m} H={h['H']} guest={h['guest']} newest type={h['newest_type']} src={h['newest_src']} cat={h['newest_cat']} n_hist_cats32={h['n_hist_cats32']} n_feed_cats={h['n_feed_cats']} hour={h['hour']} npos={int(D.npos[f])} best={best[f]} ndcg={D.base[f]:.3f}")
    s = sub.sort('rank')
    show = pl.concat([s.head(8), s.filter((pl.col('label') == 1) & (pl.col('rank') > 8))])
    print(show.select(cols))
