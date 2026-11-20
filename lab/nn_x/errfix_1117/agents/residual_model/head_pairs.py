"""Head pairs: feeds where exactly one of the model's rank-1/rank-2 items is positive. Which feature differences
(r1 - r2) predict that rank 1 is the positive, beyond the score gap? Univariate logistic z per feature."""
import numpy as np, polars as pl
from scipy.optimize import minimize
import prep
from prep import kit
df = prep.derive(kit.load())
names = [n for n in prep.feature_names(df) if not n.startswith('fm_') and n not in ('rank','fidx','M','H','guest','uvec_missing','hour','weekday','newest_type','newest_src','newest_cat','n_hist_cats32','n_feed_cats')]
h = df.filter(pl.col('rank') <= 2).sort(['fidx', 'rank'])
a = h.filter(pl.col('rank') == 1); b = h.filter(pl.col('rank') == 2)
assert (a['fidx'].to_numpy() == b['fidx'].to_numpy()).all()
ya, yb = a['label'].to_numpy(), b['label'].to_numpy()
m = ya != yb
print('feeds with exactly one of top-2 positive', m.sum(), 'rank1 wins', ya[m].mean())
gap = (a['score'].to_numpy() - b['score'].to_numpy())[m]
Y = ya[m]
rows = []
for n in names:
    xa = a[n].cast(pl.Float64).fill_nan(None).to_numpy()[m]; xb = b[n].cast(pl.Float64).fill_nan(None).to_numpy()[m]
    d = xa - xb
    ok = np.isfinite(d)
    if ok.sum() < 500 or np.nanstd(d) == 0: continue
    X = np.column_stack([np.ones(ok.sum()), gap[ok], (d[ok] - d[ok].mean()) / d[ok].std()])
    yy = Y[ok]
    def f(b):
        z = X @ b; p = 1 / (1 + np.exp(-z))
        return -(yy * z - np.logaddexp(0, z)).sum(), -X.T @ (yy - p)
    bb = minimize(f, np.zeros(3), jac=True, method="BFGS").x
    p = 1 / (1 + np.exp(-X @ bb)); Hm = (X * (p * (1 - p))[:, None]).T @ X
    se = np.sqrt(np.diag(np.linalg.inv(Hm)))
    rows.append((n, int(ok.sum()), bb[2], bb[2] / se[2]))
t = pl.DataFrame(rows, schema=['feat', 'n', 'coef', 'z'], orient='row').sort(pl.col('z').abs(), descending=True)
pl.Config.set_tbl_rows(30)
print(t.head(25)); print('features', t.height)
