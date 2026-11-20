import hk, numpy as np
c = hk.Ctx(); r = c.col("rank"); s = c.score
M = c.col("M").astype(float); n = c.col("cat_n_in_feed").astype(float)
share = n / M; lcn = np.log1p(n)
for nm, x in [("share", share), ("log_cat_n", lcn), ("log_share", np.log(n / M))]:
    for w in [0.02, 0.05, 0.1, 0.2, 0.3]:
        sx = (x - x.mean()) / x.std()
        c.rep(s + w * sx, f"{nm} w={w} full")
