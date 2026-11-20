"""Pairs (i above j) within top-15 with exactly one positive: does a lens-feature difference predict which is positive beyond the score gap?"""
import hi, numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
D = hi.Data(); df = D.df
K = 15
r = D.col('rank')
feats = ['h1', 'h5', 'h10', 'h32', 'hmean5', 'hmax', 'c_contact', 'c_order', 'c_fav', 'c_lead', 'c_vfeed', 'c_vsearch', 'ucos', 'cand_maxcos']
X0 = np.c_[[np.nan_to_num(D.col(c).astype(float), nan=-1.0) for c in feats]].T
cni = D.col('cat_newest_idx'); X0 = np.c_[X0, (cni == 0), np.where(cni < 0, np.log1p(300), np.log1p(np.maximum(cni, 0))), np.log1p(np.maximum(D.col('hmax_idx'), 0))]
names = feats + ['cni==0', 'log_cni', 'log_hidx']
# rows by feed and rank
rows_i, rows_j = [], []
for o, m in zip(D.off, D.sz):
    rr = r[o:o+m]; idx = o + np.argsort(rr)[:K]
    yy = D.y[idx]
    if yy.sum() == 0: continue
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if yy[a] != yy[b]:
                rows_i.append(idx[a]); rows_j.append(idx[b])
rows_i = np.array(rows_i); rows_j = np.array(rows_j)
lab = D.y[rows_i]  # 1 if the upper is positive
gap = D.s[rows_i] - D.s[rows_j]
dX = X0[rows_i] - X0[rows_j]
fold = D.fold_row[rows_i]
print('pairs', len(lab), 'upper positive share', lab.mean())
from sklearn.metrics import log_loss, roc_auc_score
base_ll = []; full_ll = []
for k in range(5):
    tr, te = fold != k, fold == k
    m0 = LogisticRegression(C=1.0, fit_intercept=False, max_iter=500).fit(gap[tr, None], lab[tr])
    p0 = m0.predict_proba(gap[te, None])[:, 1]
    Z = np.c_[gap, dX]
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
    m1 = LogisticRegression(C=0.1, fit_intercept=False, max_iter=1000).fit((Z[tr]) / sd, lab[tr])
    p1 = m1.predict_proba(Z[te] / sd)[:, 1]
    base_ll.append(log_loss(lab[te], p0)); full_ll.append(log_loss(lab[te], p1))
print('oof logloss gap-only %.5f  gap+lens %.5f' % (np.mean(base_ll), np.mean(full_ll)))
Z = np.c_[gap, dX]; sd = Z.std(0)
m1 = LogisticRegression(C=0.1, fit_intercept=False, max_iter=1000).fit(Z / sd, lab)
for n, c in zip(['gap'] + names, m1.coef_[0]):
    print(f'{n:12s} {c:+.3f}')
