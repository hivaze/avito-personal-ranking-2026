"""Broad lens scan: for each segment scheme (feed-level) and each feature, w per segment chosen on 4 folds (grid incl. 0),
evaluated on the 5th. Reports CF delta; also the best single-segment in-sample effect for orientation."""
import fx, kit, numpy as np, polars as pl
df = kit.load()
c = fx.Ctx(df)
s = c.base; nd = c.base_nd
F = pl.read_parquet("feeds.parquet")
q = lambda x, k: np.digitize(x, np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]))
schemes = {
    "hour4": np.digitize(F["hour"].to_numpy(), [6, 12, 18]),
    "weekday": (F["weekday"].to_numpy() == 1).astype(int),
    "nfc3": q(F["n_feed_cats"].to_numpy().astype(float), 3),
    "g110_3": q(F["g110"].to_numpy(), 3),
    "g12_3": q(F["g12"].to_numpy(), 3),
    "ntype3": np.select([F["newest_type"].to_numpy() == 1, F["newest_type"].to_numpy() == 10], [0, 1], 2),
    "cold": (F["guest"].to_numpy() | F["uvec_missing"].to_numpy() | (F["H"].to_numpy() < 5)).astype(int),
    "nhc3": q(F["n_hist_cats32"].to_numpy().astype(float), 3),
    "corr3": q(F["corr_ft_to"].to_numpy(), 3),
}
def nz(x, fill):
    x = np.asarray(x, float); return np.where(np.isfinite(x), x, fill)
feats = {
    "ucos": nz(df["ucos"].to_numpy(), 0.48), "h1": nz(df["h1"].to_numpy(), 0.45), "h5": nz(df["h5"].to_numpy(), 0.6),
    "hmax": nz(df["hmax"].to_numpy(), 0.8), "f1": df["f1"].to_numpy().astype(float), "f2": df["f2"].to_numpy().astype(float),
    "f2_home": df["f2_home"].to_numpy().astype(float), "f3": df["f3"].to_numpy().astype(float),
    "f4": df["f4"].to_numpy().astype(float), "f5": df["f5"].to_numpy().astype(float),
    "xp": df["xp_imp_c50"].to_numpy().astype(float), "log_ith": np.log1p(df["it_hist_all_luo"].to_numpy().astype(float)),
    "seq_cv": df["seq_cv_invd20"].to_numpy().astype(float), "pre2_fe_z": df["pre2_fe_z"].to_numpy().astype(float),
    "cmx": nz(df["cand_maxcos"].to_numpy(), 0.85), "sft_m_sto": (df["s_ft"].to_numpy() - df["s_to"].to_numpy()).astype(float),
}
grid = [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1]
rows = []
Dcache = {}
for fn, x in feats.items():
    xs = fx.zf(x, c.off, c.sz, c.feed)
    Dcache[fn] = np.stack([c.ndcg(s + w * xs) - nd for w in grid])
for sn, seg in schemes.items():
    for fn in feats:
        D = Dcache[fn]
        oof = np.zeros(c.nf)
        for k in range(5):
            tr = c.fold != k; te = ~tr
            for g in np.unique(seg):
                m = seg == g
                b = int(np.argmax(D[:, tr & m].mean(1)))
                oof[te & m] = D[b, te & m]
        se = oof.std(ddof=1) / np.sqrt(c.nf)
        rows.append(dict(scheme=sn, feat=fn, cf=oof.mean() * 1e4, se=se * 1e4, z=oof.mean() / se if se > 0 else 0.0))
T = pl.DataFrame(rows)
pl.Config.set_tbl_rows(300)
print(T.sort("z", descending=True).head(15))
print("mean CF over all", T["cf"].mean(), "n variants", len(T), "n with z>2", (T["z"] > 2).sum())
