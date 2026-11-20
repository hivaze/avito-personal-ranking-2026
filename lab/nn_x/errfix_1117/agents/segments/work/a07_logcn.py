"""Cross-fit score + w*log(cat_n_in_feed) and variants, by segment."""
import fx, kit, numpy as np, polars as pl
df = kit.load()
c = fx.Ctx(df)
F = pl.read_parquet("feeds.parquet")
s = c.base
M = df["M"].to_numpy().astype(float)
cn = df["cat_n_in_feed"].to_numpy().astype(float)
cr = df["cat_rank_in_feed"].to_numpy().astype(float)
Mf = F["M"].to_numpy()
segs = {"all": np.ones(c.nf, bool), "M<20": Mf < 20, "M20-50": (Mf >= 20) & (Mf < 50), "M50-100": (Mf >= 50) & (Mf < 100),
        "M100-200": (Mf >= 100) & (Mf < 200), "M200+": Mf >= 200}
grid = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
variants = {
    "log_cn": np.log(cn),
    "log_cn_minus_log_cr": np.log(cn) - np.log(cr),
    "sqrt_share": np.sqrt(cn / M),
    "log1p_cn_cap30": np.log(np.minimum(cn, 30)),
}
for nm, x in variants.items():
    D = np.stack([c.ndcg(s + w * x) - c.base_nd for w in grid])
    print("==", nm)
    for sn, m in segs.items():
        dm = D[:, m].mean(1)
        print(f"  {sn:9s} n={m.sum():5d} " + " ".join(f"{d*1e4:+6.1f}" for d in dm))
    mean, se, z, ch, _ = c.crossfit(lambda w: s + w * x, grid)
    print(f"  crossfit {mean*1e4:+.2f}e-4 se {se*1e4:.2f} z {z:.2f} chosen {ch}")
