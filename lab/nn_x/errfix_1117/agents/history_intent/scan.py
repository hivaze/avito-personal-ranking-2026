"""Clogit score test for many candidate correction terms g: z = sum g (y-E) / sqrt(sum_f npos_f Var_pf(g))."""
import hi, numpy as np, polars as pl
D = hi.Data(); df = D.df
X = np.c_[D.s]; b = np.array([0.9825])
u = X @ b
feed = D.feed
mx = np.maximum.reduceat(u, D.off)
e = np.exp(u - mx[feed]); p = e / np.add.reduceat(e, D.off)[feed]
npos = D.npos
E = p * npos[feed]
def ztest(g):
    g = np.nan_to_num(g.astype(np.float64))
    U = (g * (D.y - E)).sum()
    m1 = np.add.reduceat(p * g, D.off); m2 = np.add.reduceat(p * g * g, D.off)
    V = (npos * (m2 - m1 * m1)).sum()
    return U / np.sqrt(V + 1e-12), (D.y * g).sum()
c = {k: D.col(k).astype(np.float64) for k in ['h1', 'h5', 'h10', 'h32', 'hmean5', 'hmax', 'c_contact', 'c_order', 'c_fav', 'c_lead', 'c_vfeed', 'c_vsearch', 'ucos', 'cand_maxcos']}
nanv = {k: np.isnan(v) for k, v in c.items()}
cf = {k: np.nan_to_num(v, nan=-1.0) for k, v in c.items()}
rank = D.col('rank'); M = D.col('M'); nt = D.col('newest_type'); ns = D.col('newest_src')
cni = D.col('cat_newest_idx'); scn = D.col('same_cat_newest'); hidx = D.col('hmax_idx'); htype = D.col('hmax_type'); hsrc = D.col('hmax_src')
seg = {
    'all': np.ones(len(rank), bool),
    'v_feed': (nt == 1) & (ns == 1), 'v_search': (nt == 1) & (ns == 2), 'v_src0': (nt == 1) & (ns == 0),
    'v_other': (nt == 1) & ~np.isin(ns, [0, 1, 2]), 'contact': np.isin(nt, [2, 3]), 'fav': nt == 4,
    'order': np.isin(nt, [5, 6, 8]), 'lead': nt == 7, 'done': nt == 9, 'guest': nt == 10,
    'r1-10': rank <= 10, 'r11-50': (rank > 10) & (rank <= 50), 'r51+': rank > 50,
    'M<50': M < 50, 'M50-150': (M >= 50) & (M < 150), 'M150+': M >= 150,
}
feats = {}
for k in cf:
    for t in [0.8, 0.9, 0.95, 0.99]:
        feats[f'{k}>={t}'] = (cf[k] >= t).astype(float)
    feats[f'{k}'] = cf[k]
feats['h1-h10'] = cf['h1'] - cf['h10']
feats['h10-hmax'] = cf['h10'] - cf['hmax']
feats['hmax-h32'] = cf['hmax'] - cf['h32']
feats['hmean5-h5'] = cf['hmean5'] - cf['h5']
feats['ccontact-hmax'] = np.where(nanv['c_contact'], 0, cf['c_contact'] - cf['hmax'])
feats['cfav-hmax'] = np.where(nanv['c_fav'], 0, cf['c_fav'] - cf['hmax'])
feats['corder-hmax'] = np.where(nanv['c_order'], 0, cf['c_order'] - cf['hmax'])
feats['vsearch-vfeed'] = np.where(nanv['c_vsearch'] | nanv['c_vfeed'], 0, cf['c_vsearch'] - cf['c_vfeed'])
for t in [0, 1, 2, 3, 5, 10, 32]:
    feats[f'cni=={t}' if t == 0 else f'cni<={t}'] = ((cni >= 0) & (cni <= t)).astype(float)
feats['cni==-1'] = (cni < 0).astype(float)
feats['cni>=32'] = (cni >= 32).astype(float)
feats['log1p_cni'] = np.where(cni < 0, np.log1p(300), np.log1p(np.maximum(cni, 0)))
feats['hidx<5'] = (hidx < 5).astype(float)
feats['hidx>=50'] = (hidx >= 50).astype(float)
feats['htype_act'] = np.isin(htype, [2, 3, 4, 5, 6, 7, 8, 9]).astype(float)
feats['htype_contact'] = np.isin(htype, [2, 3, 9]).astype(float)
feats['hsrc_search'] = (hsrc == 2).astype(float)
feats['hsrc_feed'] = (hsrc == 1).astype(float)
feats['near_newest_contact'] = ((cf['h1'] >= 0.9) & np.isin(nt, [2, 3])).astype(float)
res = []
for sn, sm in seg.items():
    for fn, g in feats.items():
        gg = g * sm
        if (gg != 0).sum() < 500: continue
        z, py = ztest(gg)
        res.append((sn, fn, z, py))
r = pl.DataFrame(res, schema=['seg', 'feat', 'z', 'pos_w'], orient='row').with_columns(pl.col('z').abs().alias('az')).sort('az', descending=True)
pl.Config.set_tbl_rows(70)
print('n tests', len(r))
print(r.head(60))
r.write_csv('scan_results.csv')
