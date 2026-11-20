"""Pairwise discordant pairs within the head (ranks <= K): which feature differences predict that the
lower-ranked item is the positive, beyond the score gap?"""
import hk, numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression

c = hk.Ctx(); df = c.df
K = 10
r = c.col("rank")
head = df.with_columns(pl.Series("frow", np.arange(len(df)))).filter(pl.col("rank") <= K)
num = [x for x, t in df.schema.items() if t in (pl.Float32, pl.Int16) and x not in ("rank", "gap_next", "gap_top", "score")]
num += ["s_ft", "s_to"]
num = list(dict.fromkeys(num))
X = head.select([pl.col(x).cast(pl.Float64) for x in num]).to_numpy()
X = np.where(np.isnan(X), np.nan, X)
fid = head["fidx"].to_numpy(); y = head["label"].to_numpy(); sc = head["score"].to_numpy().astype(float)
rk = head["rank"].to_numpy()
# build pairs
starts = np.r_[0, np.nonzero(fid[1:] != fid[:-1])[0] + 1, len(fid)]
I, J = [], []
for a, b in zip(starts[:-1], starts[1:]):
    idx = np.arange(a, b)
    yy = y[idx]
    if yy.sum() == 0 or yy.sum() == len(yy):
        continue
    for i in idx:
        for j in idx:
            if rk[i] < rk[j] and y[i] != y[j]:
                I.append(i); J.append(j)
I = np.array(I); J = np.array(J)
t = y[I]  # 1 if the higher-ranked is positive
print("pairs", len(I), "P(higher is pos)", t.mean().round(4))
gap = sc[I] - sc[J]
# baseline logistic on gap
def fit(F):
    F = np.column_stack(F)
    ok = np.isfinite(F).all(1)
    m = LogisticRegression(C=1e4, max_iter=2000).fit(F[ok], t[ok])
    p = m.predict_proba(F[ok])[:, 1]
    ll = -np.mean(t[ok] * np.log(p) + (1 - t[ok]) * np.log(1 - p))
    return m, ok.sum(), ll
m0, n0, ll0 = fit([gap])
print("gap only coef", m0.coef_, m0.intercept_, "ll", ll0)
res = []
for k, name in enumerate(num):
    d = X[I, k] - X[J, k]
    ok = np.isfinite(d)
    if ok.sum() < 2000 or np.nanstd(d) == 0:
        continue
    dz = (d - np.nanmean(d)) / np.nanstd(d)
    F = np.column_stack([gap[ok], dz[ok]])
    m = LogisticRegression(C=1e4, max_iter=2000).fit(F, t[ok])
    # crude se via Fisher info
    p = m.predict_proba(F)[:, 1]
    W = p * (1 - p)
    Fa = np.column_stack([np.ones(ok.sum()), F])
    cov = np.linalg.inv(Fa.T @ (Fa * W[:, None]))
    se = np.sqrt(cov[2, 2])
    res.append((name, ok.sum(), m.coef_[0][1], m.coef_[0][1] / se))
res.sort(key=lambda x: -abs(x[3]))
print("feature diff (higher - lower) coef controlling gap; + means larger value for higher-ranked -> higher is pos")
for r_ in res:
    print(f"{r_[0]:20s} n={r_[1]:6d} coef={r_[2]:+.3f} z={r_[3]:+.2f}")
