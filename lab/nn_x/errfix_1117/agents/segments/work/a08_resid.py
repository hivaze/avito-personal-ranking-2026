"""Residuals at the head (rank<=K) under a clogit with an M-bucket temperature: obs vs exp by feature bins."""
import fx, kit, numpy as np, polars as pl, clogit, sys
df = kit.load()
c = fx.Ctx(df)
y = c.y; feed = c.feed; s = c.base
M = df["M"].to_numpy()
mb = np.digitize(M, [20, 50, 100, 200])
# per-M-bucket temperature: z = a_b * s
X = np.stack([(mb == b) * s for b in range(1, 5)], 1)
a, w, ll = clogit.fit(s, X, y, feed, c.nf, l2=0.0)
print("temps", a, a + w)
z = a * s + X @ w
zmax = np.full(c.nf, -np.inf); np.maximum.at(zmax, feed, z)
e = np.exp(z - zmax[feed]); se = np.bincount(feed, weights=e)
p = c.npos[feed] * e / se[feed]
np.save("p_clogit.npy", p)
rk = df["rank"].to_numpy()
K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
head = rk <= K
print("head rows", head.sum(), "pos", y[head].sum(), "exp", p[head].sum())

def nz(x, fill):
    x = np.asarray(x, float); return np.where(np.isfinite(x), x, fill)
feats = {}
for col in ["ucos", "h1", "h5", "hmax", "f1", "f2", "f3", "f4", "f5", "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z",
            "cand_maxcos", "cat_hist_share", "c_contact", "c_vfeed", "c_vsearch", "c_fav", "c_order", "c_lead"]:
    x = df[col].to_numpy().astype(float)
    feats[col] = x
for col in ["hmax_idx", "hmax_type", "hmax_src", "cat_user_rank", "cat_newest_idx", "cat_rank_in_feed", "cat_n_in_feed", "n_hdup95",
            "n_cand_dup95", "cat_views", "cat_actions"]:
    feats[col] = df[col].to_numpy().astype(float)
for col in ["it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]:
    feats[col] = np.log1p(df[col].to_numpy().astype(float))
feats["sft_minus_sto"] = df["s_ft"].to_numpy() - df["s_to"].to_numpy()
feats["same_cat_newest"] = df["same_cat_newest"].to_numpy().astype(float)
feats["f2_home"] = df["f2_home"].to_numpy().astype(float)
rows = []
for nm, x in feats.items():
    xh = x[head]
    fin = np.isfinite(xh)
    yh = y[head]; ph = p[head]
    if (~fin).sum() > 0:
        rows.append(dict(f=nm, bin="nan", n=int((~fin).sum()), obs=yh[~fin].sum(), exp=ph[~fin].sum()))
    xs = xh[fin]
    qs = np.unique(np.quantile(xs, [0.1, 0.25, 0.5, 0.75, 0.9]))
    b = np.digitize(xs, qs)
    for k in range(len(qs) + 1):
        mm = b == k
        if mm.sum() < 50: continue
        lo = qs[k - 1] if k > 0 else xs.min(); hi = qs[k] if k < len(qs) else xs.max()
        rows.append(dict(f=nm, bin=f"[{lo:.3g},{hi:.3g}]", n=int(mm.sum()), obs=yh[fin][mm].sum(), exp=ph[fin][mm].sum()))
R = pl.DataFrame(rows).with_columns((pl.col("obs") / pl.col("exp")).alias("ratio"),
                                    ((pl.col("obs") - pl.col("exp")) / pl.col("exp").sqrt()).alias("zres"))
pl.Config.set_tbl_rows(500); pl.Config.set_tbl_width_chars(200)
print(R.filter(pl.col("zres").abs() > 2.5).sort("zres"))
R.write_parquet(f"resid_head{K}.parquet")
