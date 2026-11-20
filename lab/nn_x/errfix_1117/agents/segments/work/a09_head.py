"""Head-only re-ordering: new = score + w * zx * [rank <= K], cross-fitted, for a few residual-driven features."""
import fx, kit, numpy as np, polars as pl
df = kit.load()
c = fx.Ctx(df)
s = c.base
rk = df["rank"].to_numpy()
F = pl.read_parquet("feeds.parquet"); Mf = F["M"].to_numpy()
feats = {
    "f3": df["f3"].to_numpy().astype(float),
    "f1": df["f1"].to_numpy().astype(float),
    "same_cat_newest": df["same_cat_newest"].to_numpy().astype(float),
    "neg_log_cur": -np.log1p(np.where(df["cat_user_rank"].to_numpy() == 0, 30, df["cat_user_rank"].to_numpy())).astype(float),
    "neg_log_cni": -np.log(2 + df["cat_newest_idx"].to_numpy().astype(float).clip(-1, None) + 30 * (df["cat_newest_idx"].to_numpy() < 0)),
    "cat_hist_share": df["cat_hist_share"].to_numpy().astype(float),
    "sto_minus_sft": (df["s_to"].to_numpy() - df["s_ft"].to_numpy()).astype(float),
    "h1": np.nan_to_num(df["h1"].to_numpy().astype(float), nan=0.4),
    "ucos": np.nan_to_num(df["ucos"].to_numpy().astype(float), nan=0.5),
}
grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
segs = {"all": np.ones(c.nf, bool), "M<30": Mf < 30, "M30-100": (Mf >= 30) & (Mf < 100), "M100+": Mf >= 100}
for K in (5, 10, 20):
    head = rk <= K
    for nm, x in feats.items():
        # standardise x on head rows across all feeds (global scale, not per feed)
        xs = (x - x[head].mean()) / x[head].std()
        D = np.stack([c.ndcg(s + w * xs * head) - c.base_nd for w in grid])
        mean, se, z, ch, _ = c.crossfit(lambda w: s + w * xs * head, grid)
        segtxt = " | ".join(f"{sn}:" + ",".join(f"{d*1e4:+.1f}" for d in D[:, m].mean(1)[1:]) for sn, m in segs.items() if sn != "all")
        print(f"K={K:2d} {nm:16s} in-sample " + " ".join(f"{d*1e4:+6.1f}" for d in D.mean(1)[1:]) +
              f"  CF {mean*1e4:+.2f} se {se*1e4:.2f} z {z:+.2f} ch {ch}")
        print("        ", segtxt)
