"""Near-tie pairs in the top-30 (|score gap| < G, exactly one positive): does a feature difference say which one is positive?
Reports, per feature, the mean of sign(d feature) * (2*upper_pos-1) (= 2*AUC-1 approx) with an SE clustered by feed."""
import hi, numpy as np, polars as pl
D = hi.Data(); df = D.df
r = D.col('rank')
feats = ['h1', 'h5', 'h10', 'h32', 'hmean5', 'hmax', 'c_contact', 'c_order', 'c_fav', 'c_lead', 'c_vfeed', 'c_vsearch', 'ucos', 'cand_maxcos',
         'f1', 'f2', 'f3', 'f4', 'f5', 'xp_imp_c50', 'it_hist_all_luo', 'seq_cv_invd20', 'pre2_fe_z', 'pre2_new_z', 's_ft', 's_to', 'cat_hist_share', 'cat_views', 'n_hdup95', 'n_cand_dup95', 'cat_n_in_feed']
X = np.c_[[np.nan_to_num(D.col(c).astype(float), nan=-1.0) for c in feats]].T
cni = D.col('cat_newest_idx'); lcni = np.where(cni < 0, 9.0, np.log1p(np.maximum(cni, 0)))
hidx = np.log1p(np.maximum(D.col('hmax_idx'), 0))
X = np.c_[X, -lcni, -hidx, D.col('same_cat_newest').astype(float)]
names = feats + ['-log_cni', '-log_hidx', 'same_cat_newest']
for G, K in [(0.05, 30), (0.1, 30), (0.2, 30)]:
    I, J, F = [], [], []
    for f, (o, m) in enumerate(zip(D.off, D.sz)):
        idx = o + np.argsort(r[o:o+m])[:K]
        yy = D.y[idx]
        if yy.sum() == 0 or yy.sum() == len(idx): continue
        ss = D.s[idx]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if yy[a] != yy[b] and ss[a] - ss[b] < G:
                    I.append(idx[a]); J.append(idx[b]); F.append(f)
    I, J, F = map(np.array, (I, J, F))
    up = 2 * D.y[I] - 1
    print(f'\nG={G} K={K}: pairs {len(I)} feeds {len(np.unique(F))} upper-positive share {(up > 0).mean():.3f}')
    out = []
    for k, n in enumerate(names):
        d = np.sign(X[I, k] - X[J, k]) * up
        # cluster SE by feed
        sums = np.bincount(F, weights=d, minlength=D.nf); cnt = np.bincount(F, minlength=D.nf)
        keep = cnt > 0
        mean = d.mean()
        resid = sums[keep] - mean * cnt[keep]
        se = np.sqrt((resid ** 2).sum()) / len(d)
        out.append((n, mean, se, mean / se, (X[I, k] != X[J, k]).mean()))
    t = pl.DataFrame(out, schema=['feat', 'mean_sign', 'se', 'z', 'share_diff'], orient='row').sort('z', descending=True)
    pl.Config.set_tbl_rows(40)
    print(t)
