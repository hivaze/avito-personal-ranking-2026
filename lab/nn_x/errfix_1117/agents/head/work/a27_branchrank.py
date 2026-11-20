import hk, numpy as np, polars as pl
c = hk.Ctx(); df = c.df
df = df.with_columns(pl.col("s_ft").rank("ordinal", descending=True).over("fidx").alias("rft"),
                     pl.col("s_to").rank("ordinal", descending=True).over("fidx").alias("rto"))
r = df["rank"].to_numpy(); rft = df["rft"].to_numpy(); rto = df["rto"].to_numpy(); y = c.y
for k in [1, 2, 3]:
    m = r == k
    worst = np.maximum(rft, rto)[m]
    for lo, hi in [(1, 1), (2, 2), (3, 4), (5, 999)]:
        mm = (worst >= lo) & (worst <= hi)
        print(f"rank {k}: max(branch rank) in [{lo},{hi}] n={mm.sum():5d} pos={y[m][mm].mean():.4f}")
# items at rank 2-5 that are rank 1 in one branch
for k in [2, 3, 4, 5]:
    m = r == k
    b1 = (np.minimum(rft, rto) == 1)[m]
    print(f"rank {k}: top-1 in one branch n={b1.sum()} pos={y[m][b1].mean():.4f} | not: n={(~b1).sum()} pos={y[m][~b1].mean():.4f}")
