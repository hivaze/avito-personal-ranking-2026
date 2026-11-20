import hi, numpy as np
D = hi.Data()
sft = D.col('s_ft').astype(float); sto = D.col('s_to').astype(float)
print('nan s_ft', np.isnan(sft).sum())
nt = D.col('newest_type')[D.off]; ns = D.col('newest_src')[D.off]; H = D.col('H')[D.off]; M = D.col('M')[D.off]; nh = D.col('n_hist_cats32')[D.off]
segs = {'v_feed': (nt == 1) & (ns == 1), 'v_search': (nt == 1) & (ns == 2), 'v_src0': (nt == 1) & (ns == 0),
        'v_other': (nt == 1) & ~np.isin(ns, [0, 1, 2]), 'contact': np.isin(nt, [2, 3]), 'fav': nt == 4,
        'order': np.isin(nt, [5, 6, 8]), 'lead/done': np.isin(nt, [7, 9]), 'guest/empty': np.isin(nt, [10, -1]),
        'H<50': H < 50, 'H50-200': (H >= 50) & (H < 200), 'H200+': H >= 200, 'M<50': M < 50, 'M50-150': (M >= 50) & (M < 150), 'M150+': M >= 150,
        'nh<=2': nh <= 2, 'nh3-5': (nh >= 3) & (nh <= 5), 'nh6+': nh >= 6}
alphas = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
nd = {a: D.ndcg(a * sft + (1 - a) * sto) - D.base for a in alphas}
print('overall', {a: round(nd[a].mean(), 5) for a in alphas})
for sn, m in segs.items():
    print(f'{sn:12s} n={m.sum():5d} ' + ' '.join(f'{a}:{nd[a][m].mean():+.5f}' for a in alphas))
