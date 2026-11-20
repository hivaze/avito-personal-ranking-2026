"""Flexible within-feed calibration baseline + clustered score tests for candidate category features.

Baseline: conditional logit p_i ∝ exp(b . g(score_i, rank_i)), g = [score, max(score-gt,0) for knots, log rank].
Residual r_i = y_i - npos_f * p_i. For feature x, U = sum r_i (x_i - xbar_f) with feed-clustered SE.
Positive z: the model under-weights x (positives have more x than the model expects).
"""
import numpy as np
import polars as pl
from scipy.optimize import minimize
import cu

SP = "<repo>/lab/nn_x/errfix_1117/agents/category/work"
df = pl.read_parquet(f"{SP}/calib.parquet")
fid, off, sz = cu.feed_id(df)
nf = len(sz)
y = df["label"].to_numpy().astype(np.float64)
s = df["score"].to_numpy().astype(np.float64)
rk = df["rank"].to_numpy().astype(np.float64)
gt = df["gap_top"].to_numpy().astype(np.float64)
npos = np.bincount(fid, weights=y)

G = np.column_stack([s, np.maximum(s - 1, 0), np.maximum(s - 2, 0), np.maximum(s - 3, 0), np.log(rk),
                     np.minimum(gt, 0.5), (rk == 1).astype(float), (rk <= 3).astype(float)])


def sm(z):
    mx = np.maximum.reduceat(z, off)
    e = np.exp(z - mx[fid])
    den = np.bincount(fid, weights=e, minlength=nf)
    return e / den[fid]


def fit(X):
    def f(b):
        z = X @ b
        p = sm(z)
        ll = -(y * np.log(p + 1e-300)).sum()
        # gradient: -sum_i (y_i - npos p_i) x_i
        r = y - npos[fid] * p
        return ll, -(r @ X)
    b0 = np.zeros(X.shape[1]); b0[0] = 1.0
    res = minimize(f, b0, jac=True, method="L-BFGS-B")
    return res.x, -res.fun


b, ll = fit(G)
print("baseline coefs", np.round(b, 3), "ll", round(ll, 1))
b1, ll1 = fit(G[:, :1])
print("score-only ll", round(ll1, 1))
p = sm(G @ b)
E = npos[fid] * p
np.save(f"{SP}/E_flex.npy", E)
r = y - E


def test(x, name):
    x = np.asarray(x, dtype=np.float64)
    U_f = np.bincount(fid, weights=r * x, minlength=nf)
    U = U_f.sum()
    se = np.sqrt((U_f ** 2).sum())
    O = (y * x).sum(); Ex = (E * x).sum()
    return dict(name=name, n=int((x != 0).sum()), O=round(O, 1), E=round(Ex, 1), OE=round(O / max(Ex, 1e-9), 3),
                z=round(U / se if se > 0 else 0, 2))


rows = []
cat = df["cat"].to_numpy()
vert = df["vert"].to_numpy()
cst = df["cstate"].to_numpy()
rb = df["rb"].to_numpy()
for c in np.unique(cat):
    rows.append(test(cat == c, f"cat:{cu.CATNAME[c]}"))
for v in np.unique(vert):
    rows.append(test(vert == v, f"vert:{v}"))
for c in np.unique(cst):
    rows.append(test(cst == c, f"cstate:{c}"))
    for b_ in np.unique(rb):
        rows.append(test((cst == c) & (rb == b_), f"cstate:{c} x rb:{b_}"))
for b_ in np.unique(rb):
    rows.append(test(rb == b_, f"rb:{b_}"))
for v in np.unique(vert):
    for c in np.unique(cst):
        rows.append(test((vert == v) & (cst == c), f"vert:{v} x cstate:{c}"))
    for b_ in ("01-03", "04-10", "11-30"):
        rows.append(test((vert == v) & (rb == b_), f"vert:{v} x rb:{b_}"))
res = pl.DataFrame(rows)
pl.Config.set_tbl_rows(400)
pl.Config.set_tbl_width_chars(200)
print(res.filter(pl.col("E") >= 15).sort("z"))
res.write_csv(f"{SP}/screen1.csv")
