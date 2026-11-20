import hk, numpy as np, polars as pl
c = hk.Ctx(); df = c.df; s = c.score; r = c.col("rank")
a = c.col("s_ft").astype(float); b = c.col("s_to").astype(float)
for lam in [-0.1, -0.05, 0.05, 0.1]:
    c.rep(s + lam * a * b, f"agreement product lam={lam}")
for K in [5, 10, 20, 40]:
    d = df.with_columns(pl.Series("inh", r <= K)).with_columns(
        [((pl.col(x) - pl.col(x).filter(pl.col("inh")).mean().over("fidx")) / (pl.col(x).filter(pl.col("inh")).std().over("fidx") + 1e-6)).alias("z_" + x) for x in ["s_ft", "s_to"]])
    za = d["z_s_ft"].to_numpy(); zb = d["z_s_to"].to_numpy()
    new = s.copy(); inh = r <= K
    new[inh] = 100 + 0.5 * (za[inh] + zb[inh])
    c.rep(new + 1e-9 * s, f"head-local z combine K={K}")
