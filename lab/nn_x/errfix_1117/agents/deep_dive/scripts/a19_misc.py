"""Close-out tests from feed reading: category depth / share, fresh-cold head demotion, re-post saturation in head."""
from util import *
df = load(); c = Ctx(df); s = c.base
def col(n): return df[n].to_numpy().astype(np.float64)
def zf(a):
    a = a.copy(); a[~np.isfinite(a)] = np.nanmean(a)
    m = np.bincount(c.fid, weights=a) / c.sz; v = np.bincount(c.fid, weights=a * a) / c.sz - m * m
    return (a - m[c.fid]) / np.sqrt(np.maximum(v[c.fid], 1e-6))
rk = col("rank")
G = {
 "z_log_cat_n_in_feed": zf(np.log(col("cat_n_in_feed"))),
 "cat_share_in_feed": col("cat_n_in_feed") / col("M"),
 "deep_in_big_cat(r>80)": (rk > 80) * zf(np.log(col("cat_n_in_feed"))),
 "head_fresh_cold(r<=3)": ((rk <= 3) & (col("xp_imp_c50") == 0) & (col("f4") < 0.25)).astype(float),
 "head_hd95>=10(r<=10)": ((rk <= 10) & (col("n_hdup95") >= 10)).astype(float),
 "head_hty4(r<=5)": ((rk <= 5) & (col("hmax_type") == 4)).astype(float),
 "head_scn(r<=5)": ((rk <= 5) & (col("same_cat_newest") == 1)).astype(float),
}
W = [-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3]
for k, g in G.items():
    r = c.crossfit(lambda w: s + w * g, W)
    print(f"{k:24s} cf={r['cf']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} chosen={r['chosen']} ins={r['ins_delta']:+.5f}")
print("variants", len(G) * len(W))
