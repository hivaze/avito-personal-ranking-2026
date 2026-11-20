"""Branch disagreement as an uncertainty signal: score - lam*|s_ft - s_to|; soft-min / soft-max of the branches;
also restricted to the head."""
import hk, numpy as np

c = hk.Ctx()
a = c.col("s_ft").astype(float); b = c.col("s_to").astype(float); s = c.score; r = c.col("rank")
d = np.abs(a - b)
print("|ft-to| quantiles all", np.quantile(d, [.5, .9, .99]).round(3), " head10", np.quantile(d[r <= 10], [.5, .9, .99]).round(3))
for lam in [-0.3, -0.15, 0.15, 0.3, 0.6, 1.0]:
    c.rep(s - lam * d, f"score - {lam}*|d| full")
for lam in [-0.3, 0.3, 0.6]:
    for K in [3, 10, 30]:
        inh = r <= K
        new = s.copy(); new[inh] = 100 + s[inh] - lam * d[inh]
        c.rep(new, f"head{K} score - {lam}*|d|")
# rank average
import polars as pl
df = c.df.with_columns(pl.col("s_ft").rank("average").over("fidx").alias("rft"), pl.col("s_to").rank("average").over("fidx").alias("rto"))
ra = df["rft"].to_numpy() + df["rto"].to_numpy()
c.rep(ra + 1e-6 * s, "rank-average")
