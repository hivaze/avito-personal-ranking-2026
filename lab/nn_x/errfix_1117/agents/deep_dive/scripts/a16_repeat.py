"""Same user, several feeds: is an item shown again to the same user (in another feed) clicked less/more?"""
from util import *

df = load()
c = Ctx(df)
pl.Config.set_tbl_rows(40)
fl = df.group_by("fidx", maintain_order=True).agg(pl.col("user_id").first(), pl.col("hour").first(), pl.col("weekday").first(),
                                                  pl.col("H").first(), pl.col("M").first())
nfu = fl.filter(pl.col("user_id") != 0).group_by("user_id").len()
print("non-guest users", nfu.height, "feeds per user dist:", nfu["len"].value_counts().sort("len").head(10))
d = df.filter(pl.col("user_id") != 0).with_columns(pl.col("fidx").n_unique().over("user_id").alias("nfeeds"),
                                                   pl.len().over(["user_id", "item_id"]).alias("n_shown_user"))
print("rows of users with 2+ feeds:", d.filter(pl.col("nfeeds") >= 2).height, "rows whose item is shown to the same user in another feed:",
      d.filter(pl.col("n_shown_user") >= 2).height)
E = np.load("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/E_calib.npy")
dd = df.with_columns(pl.Series("E0", E)).with_columns((pl.col("E0") * pl.col("label").sum().over("fidx") / pl.col("E0").sum().over("fidx")).alias("E"))
dd = dd.with_columns(pl.when(pl.col("user_id") == 0).then(1).otherwise(pl.len().over(["user_id", "item_id"])).alias("n_shown_user"),
                     pl.when(pl.col("user_id") == 0).then(1).otherwise(pl.col("fidx").n_unique().over("user_id")).alias("nfeeds"))
print(dd.filter(pl.col("nfeeds") >= 2).group_by((pl.col("n_shown_user") >= 2).alias("repeat")).agg(
    pl.len(), pl.col("label").mean().alias("rate"), pl.col("label").sum().alias("O"), pl.col("E").sum().alias("E"),
    pl.col("rank").mean()).with_columns((pl.col("O") / pl.col("E")).alias("OE")))
# order of feeds within user: by fidx (check whether fidx is time-ordered: compare H, weekday/hour across a user's feeds)
u2 = fl.filter(pl.col("user_id") != 0).sort(["user_id", "fidx"]).with_columns(pl.col("fidx").rank("ordinal").over("user_id").alias("k"),
                                                                              pl.len().over("user_id").alias("n"))
u2 = u2.filter(pl.col("n") == 2)
a = u2.filter(pl.col("k") == 1); b = u2.filter(pl.col("k") == 2)
j = a.join(b, on="user_id", suffix="_b")
print("2-feed users", j.height, "H later>=earlier:", (j["H_b"] >= j["H"]).mean(), "weekday later>=earlier", (j["weekday_b"] >= j["weekday"]).mean(),
      "same weekday & hour later>=", ((j["weekday_b"] == j["weekday"]) & (j["hour_b"] >= j["hour"])).sum(), ((j["weekday_b"] == j["weekday"]) & (j["hour_b"] < j["hour"])).sum())
