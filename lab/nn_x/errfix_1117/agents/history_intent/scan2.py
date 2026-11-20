"""Head-restricted clogit score test (top-K of each feed) for segment x feature terms."""
import hi, numpy as np, polars as pl, sys
K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
D = hi.Data(); df = D.df
rank = D.col('rank')
keep = rank <= K
sub = df.filter(pl.Series(keep))
off, sz = hi.kit.feed_layout(sub)
y = sub['label'].to_numpy().astype(float); s = sub['score'].to_numpy().astype(float)
feed = np.repeat(np.arange(len(sz)), sz)
npos = np.add.reduceat(y, off)
b, _ = hi.clogit_fit(np.c_[s], y, off, sz)
u = s * b[0]; mx = np.maximum.reduceat(u, off); e = np.exp(u - mx[feed]); p = e / np.add.reduceat(e, off)[feed]
E = p * npos[feed]
print('head K', K, 'slope', b, 'feeds with pos in head', (npos > 0).sum())
def ztest(g):
    g = np.nan_to_num(g.astype(np.float64))
    U = (g * (y - E)).sum()
    m1 = np.add.reduceat(p * g, off); m2 = np.add.reduceat(p * g * g, off)
    V = (npos * (m2 - m1 * m1)).sum()
    return U / np.sqrt(V + 1e-12)
col = lambda k: sub[k].to_numpy()
raw = lambda k: col(k).astype(np.float64)
cf = {k: np.nan_to_num(raw(k), nan=-1.0) for k in ['h1', 'h5', 'h10', 'h32', 'hmean5', 'hmax', 'c_contact', 'c_order', 'c_fav', 'c_lead', 'c_vfeed', 'c_vsearch', 'ucos', 'cand_maxcos']}
isn = {k: np.isnan(raw(k)) for k in cf}
nt = col('newest_type'); ns = col('newest_src'); M = col('M'); H = col('H'); nh = col('n_hist_cats32'); nfc = col('n_feed_cats')
cni = col('cat_newest_idx'); r = col('rank'); hidx = col('hmax_idx'); htype = col('hmax_type'); hsrc = col('hmax_src')
cvw = col('cat_views'); cac = col('cat_actions'); cur = col('cat_user_rank'); csh = col('cat_hist_share'); hour = col('hour')
seg = {'all': np.ones(len(y), bool),
       'v_feed': (nt == 1) & (ns == 1), 'v_search': (nt == 1) & (ns == 2), 'v_src0': (nt == 1) & (ns == 0),
       'v_other': (nt == 1) & ~np.isin(ns, [0, 1, 2]), 'contact': np.isin(nt, [2, 3]), 'fav': nt == 4,
       'order': np.isin(nt, [5, 6, 8]), 'lead': nt == 7, 'guest': nt == 10,
       'M<50': M < 50, 'M50-150': (M >= 50) & (M < 150), 'M150+': M >= 150,
       'H<50': H < 50, 'H50-200': (H >= 50) & (H < 200), 'H200+': H >= 200,
       'nh<=2': nh <= 2, 'nh3-5': (nh >= 3) & (nh <= 5), 'nh6+': nh >= 6,
       'night': (hour >= 0) & (hour < 7), 'eve': (hour >= 18)}
feats = {}
for k in cf:
    for t in [0.9, 0.95]:
        feats[f'{k}>={t}'] = (cf[k] >= t).astype(float)
    feats[k] = cf[k]
feats['h1-h10'] = cf['h1'] - cf['h10']
feats['h10-hmax'] = cf['h10'] - cf['hmax']
feats['hmean5-h5'] = cf['hmean5'] - cf['h5']
feats['ccontact-hmax'] = np.where(isn['c_contact'], 0, cf['c_contact'] - cf['hmax'])
feats['cfav-hmax'] = np.where(isn['c_fav'], 0, cf['c_fav'] - cf['hmax'])
feats['corder-hmax'] = np.where(isn['c_order'], 0, cf['c_order'] - cf['hmax'])
feats['vsearch-vfeed'] = np.where(isn['c_vsearch'] | isn['c_vfeed'], 0, cf['c_vsearch'] - cf['c_vfeed'])
feats['cni==0'] = (cni == 0).astype(float)
feats['cni==-1'] = (cni < 0).astype(float)
feats['log1p_cni'] = np.where(cni < 0, np.log1p(300), np.log1p(np.maximum(cni, 0)))
feats['hidx<5'] = (hidx < 5).astype(float)
feats['log1p_hidx'] = np.log1p(np.maximum(hidx, 0))
feats['htype_act'] = np.isin(htype, [2, 3, 4, 5, 6, 7, 8, 9]).astype(float)
feats['hsrc_search'] = (hsrc == 2).astype(float)
feats['hsrc_feed'] = (hsrc == 1).astype(float)
feats['log1p_catviews'] = np.log1p(cvw)
feats['log1p_catact'] = np.log1p(cac)
feats['cat_share'] = csh
feats['cat_rank1'] = (cur == 1).astype(float)
res = []
for sn, sm in seg.items():
    for fn, g in feats.items():
        gg = g * sm
        if (gg != 0).sum() < 300: continue
        res.append((sn, fn, ztest(gg)))
rr = pl.DataFrame(res, schema=['seg', 'feat', 'z'], orient='row').with_columns(pl.col('z').abs().alias('az')).sort('az', descending=True)
pl.Config.set_tbl_rows(50)
print('n tests', len(rr))
print(rr.head(45))
rr.write_csv(f'scan2_K{K}.csv')
