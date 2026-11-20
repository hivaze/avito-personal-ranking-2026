"""Step 2: global and per-segment blend scan: score + w * z_feed(x)."""
import fx, kit, numpy as np, polars as pl
df = kit.load()
c = fx.Ctx(df)
off, sz, feed = c.off, c.sz, c.feed
F = pl.read_parquet("feeds.parquet")

def nz(a, fill=0.0):
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.isfinite(a), a, fill)

sig = {}
for col in ["ucos", "h1", "h5", "h10", "h32", "hmax", "hmean5", "f1", "f2", "f3", "f4", "f5", "xp_imp_c50",
            "seq_cv_invd20", "pre2_fe_z", "pre2_new_z", "cand_maxcos", "cat_hist_share", "c_contact", "c_vfeed", "c_vsearch"]:
    x = df[col].to_numpy().astype(np.float64)
    fill = np.nanmin(x) if col.startswith("c_") else np.nanmean(x)
    sig[col] = fx.zf(nz(x, fill), off, sz, feed)
for col in ["it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]:
    sig["log_" + col] = fx.zf(np.log1p(df[col].to_numpy().astype(np.float64)), off, sz, feed)
sig["ft_minus_to"] = df["s_ft"].to_numpy().astype(np.float64) - df["s_to"].to_numpy().astype(np.float64)
sig["neg_cat_rank_in_feed_log"] = fx.zf(-np.log(df["cat_rank_in_feed"].to_numpy().astype(np.float64)), off, sz, feed)

grid = [-0.3, -0.15, -0.07, -0.03, 0.03, 0.07, 0.15, 0.3]
segs = {
    "all": np.ones(c.nf, bool),
    "M<20": F["M"].to_numpy() < 20,
    "M20-100": (F["M"].to_numpy() >= 20) & (F["M"].to_numpy() < 100),
    "M100+": F["M"].to_numpy() >= 100,
    "H<20": F["H"].to_numpy() < 20,
    "guest": F["guest"].to_numpy(),
    "g110_low": F["g110"].to_numpy() < np.quantile(F["g110"].to_numpy(), .3),
    "g110_high": F["g110"].to_numpy() > np.quantile(F["g110"].to_numpy(), .7),
    "corr_low": F["corr_ft_to"].to_numpy() < np.quantile(F["corr_ft_to"].to_numpy(), .3),
}
rows = []
for name, x in sig.items():
    D = np.stack([c.ndcg(c.base + w * x) - c.base_nd for w in grid])
    for sn, m in segs.items():
        dm = D[:, m].mean(1)
        se = D[:, m].std(1, ddof=1) / np.sqrt(m.sum())
        b = int(np.argmax(dm))
        rows.append(dict(sig=name, seg=sn, n=int(m.sum()), best_w=grid[b], best_d=dm[b], z=dm[b] / se[b],
                         dneg=dm[2], dpos=dm[5]))
R = pl.DataFrame(rows)
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_width_chars(200)
for sn in segs:
    print("=== seg", sn)
    print(R.filter(pl.col("seg") == sn).sort("best_d", descending=True).head(12))
R.write_parquet("blendscan.parquet")
