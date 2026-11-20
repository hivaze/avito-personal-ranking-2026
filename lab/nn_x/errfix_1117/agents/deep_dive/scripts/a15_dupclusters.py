"""Near-duplicate candidates competing for one click? Reconstruct pairs via identical cand_maxcos within (feed, cat)."""
from util import *

df = load()
c = Ctx(df)
E0 = np.load("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/E_calib.npy")
df = df.with_columns(pl.Series("E0", E0), pl.Series("ri", np.arange(df.height)))
df = df.with_columns((pl.col("E0") * pl.col("label").sum().over("fidx") / pl.col("E0").sum().over("fidx")).alias("E"))
d = df.filter(pl.col("n_cand_dup95") >= 1)
d = d.with_columns(pl.len().over(["fidx", "cat", "cand_maxcos"]).alias("grp_n"),
                   pl.col("score").rank("ordinal", descending=True).over(["fidx", "cat", "cand_maxcos"]).alias("grp_rank"))
print("rows with n_cand_dup95>=1:", d.height, " in reconstructed groups of 2+:", d.filter(pl.col("grp_n") >= 2).height)
g = d.filter(pl.col("grp_n") >= 2)
print(g.group_by("grp_rank").agg(pl.len(), pl.col("label").sum().alias("O"), pl.col("E").sum().alias("E"),
                                  pl.col("rank").mean().alias("mean_rank")).with_columns((pl.col("O") / pl.col("E")).alias("OE")).sort("grp_rank").head(6))
# pairs: both positive?
pairs = g.filter(pl.col("grp_n") == 2).group_by(["fidx", "cat", "cand_maxcos"]).agg(pl.col("label").sum().alias("npos"), pl.col("rank").min().alias("r_hi"), pl.col("rank").max().alias("r_lo"))
print("pairs:", pairs.height, pairs.group_by("npos").len().sort("npos"))
# compare with expectation if independent: P(both) vs P(a)P(b)
# demotion rule for non-leaders in groups
idx_nonlead = g.filter(pl.col("grp_rank") >= 2)["ri"].to_numpy()
x = np.zeros(df.height); x[idx_nonlead] = 1.0
r = c.crossfit(lambda w: c.base + w * x, [-0.3, -0.2, -0.1, -0.05, 0.05, 0.1])
print(f"demote non-leaders: cf={r['cf']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} chosen={r['chosen']} ins_best={r['ins_best']} ins={r['ins_delta']:+.5f}")
idx_lead = g.filter(pl.col("grp_rank") == 1)["ri"].to_numpy()
x2 = np.zeros(df.height); x2[idx_lead] = 1.0
r = c.crossfit(lambda w: c.base + w * x2, [-0.1, -0.05, 0.05, 0.1, 0.2])
print(f"boost leaders: cf={r['cf']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} chosen={r['chosen']} ins_best={r['ins_best']} ins={r['ins_delta']:+.5f}")
# confident rank-1 with a near-dup in feed: P(r1 pos) by cand_maxcos, M-controlled
w = pl.read_parquet("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/head_wide.parquet")
w = w.with_columns(pl.col("M").cut([30, 60, 120, 200]).cast(pl.Utf8).alias("Mb"), (pl.col("cand_maxcos_1") >= 0.9).alias("r1_has_close"))
print(w.group_by(["Mb", "r1_has_close"]).agg(pl.len(), pl.col("label_1").mean().round(4), pl.col("label_2").mean().round(4)).sort(["Mb", "r1_has_close"]))
