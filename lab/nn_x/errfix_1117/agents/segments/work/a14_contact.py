"""Contact-heavy vs view-heavy histories: segment-specific blends, cross-fitted per segment."""
import fx, kit, numpy as np, polars as pl
df = kit.load()
c = fx.Ctx(df)
s = c.base
# feed-level action share over the categories present in the feed (approximation of the history mix)
u = df.select("fidx", "cat", "cat_views", "cat_actions").unique().group_by("fidx").agg(
    pl.col("cat_views").sum().alias("v"), pl.col("cat_actions").sum().alias("a")).sort("fidx")
v = u["v"].to_numpy().astype(float); a = u["a"].to_numpy().astype(float)
ash = a / np.maximum(a + v, 1)
print("action share quantiles", np.quantile(ash, [.1, .25, .5, .75, .9]))
F = pl.read_parquet("feeds.parquet")
assert (u["fidx"].to_numpy() == F["fidx"].to_numpy()).all()
seg = np.digitize(ash, np.quantile(ash, [1 / 3, 2 / 3]))
nd = c.base_nd
for g in range(3):
    m = seg == g
    print("seg", g, m.sum(), "ndcg", nd[m].mean(), "M", F["M"].to_numpy()[m].mean())

def nz(x, fill):
    x = np.asarray(x, float); return np.where(np.isfinite(x), x, fill)
feats = {
    "c_contact": nz(df["c_contact"].to_numpy(), 0.3), "c_fav": nz(df["c_fav"].to_numpy(), 0.3),
    "c_vfeed": nz(df["c_vfeed"].to_numpy(), 0.3), "c_vsearch": nz(df["c_vsearch"].to_numpy(), 0.3),
    "h1": nz(df["h1"].to_numpy(), 0.45), "hmax": nz(df["hmax"].to_numpy(), 0.8), "f1": df["f1"].to_numpy().astype(float),
    "f3": df["f3"].to_numpy().astype(float), "ucos": nz(df["ucos"].to_numpy(), 0.48),
    "log_ith": np.log1p(df["it_hist_all_luo"].to_numpy().astype(float)), "xp": df["xp_imp_c50"].to_numpy().astype(float),
    "f4": df["f4"].to_numpy().astype(float), "f5": df["f5"].to_numpy().astype(float),
}
grid = [-0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2]
for nm, x in feats.items():
    xs = fx.zf(x, c.off, c.sz, c.feed)
    D = np.stack([c.ndcg(s + w * xs) - nd for w in grid])
    oof = np.zeros(c.nf); ch = {}
    for k in range(5):
        tr = c.fold != k; te = ~tr
        for g in range(3):
            m = seg == g
            b = int(np.argmax(D[:, tr & m].mean(1))); ch.setdefault(g, []).append(grid[b])
            oof[te & m] = D[b, te & m]
    se = oof.std(ddof=1) / np.sqrt(c.nf)
    ins = " | ".join(f"s{g}:" + ",".join(f"{d*1e4:+.0f}" for d in D[:, seg == g].mean(1)) for g in range(3))
    print(f"{nm:10s} CF {oof.mean()*1e4:+.2f}e-4 se {se*1e4:.2f} z {oof.mean()/se:+.2f} ch {ch}\n     {ins}")
