"""Null calibration of the one-feature MLE scan: random N(0,1) columns (row-level and feed-smooth), same
cross-fitted procedure. Gives the OOF z distribution expected when there is no signal."""
import sys
import numpy as np
from scipy.optimize import minimize_scalar
import prep
from prep import kit
base = sys.argv[1] if len(sys.argv) > 1 else "score"
df = kit.load(['fidx', 'user_id', 'label', 'score', 's_to', 's_ft'])
y = df['label'].to_numpy().astype(float); s = df[base].to_numpy().astype(float)
off, sz = kit.feed_layout(df); fid = np.repeat(np.arange(len(sz)), sz); fold = kit.folds(df)
F = prep.FastNDCG(y, off, sz); b0 = F(s)
def nll(w, z0, x, yy, o, f, npos):
    z = z0 + w * x; mx = np.maximum.reduceat(z, o)
    return -((yy * z).sum() - (npos * (np.log(np.add.reduceat(np.exp(z - mx[f]), o)) + mx)).sum())
fi = []
for k in range(5):
    tr = fold != k; trf = np.unique(fid[tr]); o = np.r_[0, np.cumsum(sz[trf])[:-1]]
    fi.append((tr, o, np.repeat(np.arange(len(trf)), sz[trf]), np.add.reduceat(y[tr], o), prep.softmax_scale(s[tr], y[tr], o, sz[trf])))
rng = np.random.default_rng(1)
zs, ws = [], []
for r in range(40):
    x = rng.standard_normal(len(y))
    oof = np.zeros(len(y))
    for k, (tr, o, f, npos, c) in enumerate(fi):
        w = minimize_scalar(nll, bounds=(-1, 1), method='bounded', args=(c * s[tr], x[tr], y[tr], o, f, npos), options=dict(xatol=1e-4)).x
        oof[fold == k] = c * s[fold == k] + w * x[fold == k]; ws.append(w)
    d = F(oof) - b0
    zs.append(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
print(base, 'null OOF z: mean', np.mean(zs), 'sd', np.std(zs), 'mean|w|', np.mean(np.abs(ws)))
