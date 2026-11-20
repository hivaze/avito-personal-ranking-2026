import hi, numpy as np
D = hi.Data()
sft = D.col('s_ft').astype(float); sto = D.col('s_to').astype(float)
nt = D.col('newest_type')[D.off]; ns = D.col('newest_src')[D.off]
grp = np.select([(nt == 1) & (ns == 1), (nt == 1) & (ns == 2), (nt == 1) & (ns == 0), (nt == 1), np.isin(nt, [2, 3]), nt == 4, np.isin(nt, [5, 6, 8]), np.isin(nt, [7, 9])],
                [0, 1, 2, 3, 4, 5, 6, 7], 8)
alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
nd = {a: D.ndcg(a * sft + (1 - a) * sto) - D.base for a in alphas}
oof = np.zeros(D.nf); chosen = []
for k in range(5):
    tr = D.fold != k; te = D.fold == k
    ch = {}
    for g in np.unique(grp):
        ch[g] = max(alphas, key=lambda a: nd[a][tr & (grp == g)].mean())
        m = te & (grp == g)
        oof[m] = nd[ch[g]][m]
    chosen.append(ch)
print('oof %+.5f se %.5f z %.2f' % (oof.mean(), oof.std(ddof=1) / np.sqrt(D.nf), oof.mean() / (oof.std(ddof=1) / np.sqrt(D.nf))))
for c in chosen: print(c)
