"""Lens selection: (a) confident wrong rank-1, (b) positives at rank 50+ in feeds < 120, and good feeds."""
from util import *

df = load()
c = Ctx(df)
fl = (df.group_by("fidx", maintain_order=True)
      .agg(pl.col("M").first(), pl.col("H").first(), pl.col("guest").first(),
           pl.col("label").sum().alias("npos"),
           pl.col("rank").filter(pl.col("label") == 1).min().alias("best_pos_rank"),
           pl.col("label").filter(pl.col("rank") == 1).first().alias("lab_r1"),
           pl.col("gap_next").filter(pl.col("rank") == 1).first().alias("gap12"),
           pl.col("score").filter(pl.col("rank") == 1).first().alias("s1"),
           pl.col("newest_type").first(), pl.col("newest_src").first(), pl.col("newest_cat").first(),
           pl.col("n_hist_cats32").first(), pl.col("n_feed_cats").first(), pl.col("hour").first(),
           pl.col("uvec_missing").first(), pl.col("user_id").first()))
fl = fl.with_columns(pl.Series("ndcg", c.base_ndcg))
print(fl.describe())
print("gap12 quantiles", fl["gap12"].quantile(0.5), fl["gap12"].quantile(0.75), fl["gap12"].quantile(0.9))
# P(rank1 positive) by gap12 bins
fl = fl.with_columns(pl.col("gap12").qcut(10, allow_duplicates=True).alias("gbin"))
print(fl.group_by("gbin").agg(pl.len(), pl.col("lab_r1").mean(), pl.col("ndcg").mean(), pl.col("M").mean()).sort("gbin"))
# lens a: rank1 negative, gap12 in top quartile, best pos rank >= 30
qa = fl["gap12"].quantile(0.75)
A = fl.filter((pl.col("lab_r1") == 0) & (pl.col("gap12") >= qa) & (pl.col("best_pos_rank") >= 30))
print("lens a feeds", A.height, "of", fl.height)
# lens b: M < 120, some positive at rank >= 50
B = df.filter((pl.col("M") < 120) & (pl.col("label") == 1) & (pl.col("rank") >= 50))
print("lens b positives", B.height, "feeds", B["fidx"].n_unique(), "feeds M<120", fl.filter(pl.col("M") < 120).height,
      "positives M<120", df.filter((pl.col('M') < 120) & (pl.col('label') == 1)).height)
G = fl.filter(pl.col("best_pos_rank") <= 3)
print("good feeds", G.height)
fl.write_parquet("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/feedlevel.parquet")
