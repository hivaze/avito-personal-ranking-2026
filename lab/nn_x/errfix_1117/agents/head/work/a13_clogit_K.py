import hk, clogit, numpy as np, polars as pl
c = hk.Ctx(); r = c.col("rank"); fid = c.col("fidx"); sc = c.score
M = c.col("M").astype(float)
feats = {
 "log_cat_n": np.log1p(c.col("cat_n_in_feed").astype(float)),
 "cat_share": c.col("cat_n_in_feed").astype(float) / M,
 "f2": c.col("f2").astype(float),
 "f2_home": c.col("f2_home").astype(float),
 "pre2_fe_z": c.col("pre2_fe_z").astype(float),
 "d_ft_to": (c.col("s_ft") - c.col("s_to")).astype(float),
 "c_fav_nan0": np.nan_to_num(c.col("c_fav").astype(float), nan=0.0),
 "hmean5_nan0": np.nan_to_num(c.col("hmean5").astype(float), nan=0.0),
}
for K in [3, 5, 10, 20, 50, 300]:
    inh = r <= K
    P, S, Soff = clogit.build_sets(fid, c.y, inh)
    line = []
    for nm, x in feats.items():
        mu = x[inh].mean(); sd = x[inh].std()
        X = np.column_stack([sc, (x - mu) / sd])
        b, se, ll = clogit.fit(X, P, S, Soff)
        line.append(f"{nm}:{b[1]:+.3f}({b[1]/se[1]:+.1f})")
    print(f"K={K:3d} sets={len(P)} " + " ".join(line))
