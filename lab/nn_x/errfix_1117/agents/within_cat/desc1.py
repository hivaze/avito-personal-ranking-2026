from wc import *
df = load(); d = features(df)
off, sz = kit.feed_layout(d)
y = d["label"].to_numpy(); s = d["score"].to_numpy().astype(np.float64)
nb = kit.ndcg_per_feed(s, y, off, sz)
# per positive: rank, cat rank, cat_n
pos = d.filter(pl.col("label") == 1)
print("positives", len(pos))
pos = pos.with_columns(
    pl.when(pl.col("rank") <= 3).then(pl.lit("a 1-3")).when(pl.col("rank") <= 10).then(pl.lit("b 4-10")).when(pl.col("rank") <= 30).then(pl.lit("c 11-30")).otherwise(pl.lit("d 31+")).alias("rb"))
print(pos.group_by("rb").agg(pl.len(), pl.col("cat_rank_in_feed").median().alias("crank_med"), (pl.col("cat_rank_in_feed") == 1).mean().alias("is_catlead"),
      ((pl.col("cat_rank_in_feed") - 1) / (pl.col("rank") - 1).clip(1, None)).mean().alias("share_above_samecat"),
      pl.col("cat_n_in_feed").median().alias("catn_med"), pl.col("cat_user_rank").eq(1).mean().alias("topcat"),
      pl.col("cat_user_rank").eq(0).mean().alias("absentcat"), pl.col("M").median()).sort("rb"))
# For positives at rank 31+: how many above are same-category?
p31 = pos.filter(pl.col("rank") > 30)
print("rank>30 positives: crank>10 share", (p31["cat_rank_in_feed"] > 10).mean(), " crank==1 share", (p31["cat_rank_in_feed"] == 1).mean())
# lost DCG decomposition: for each positive, ideal is top; count items above it same cat vs other cat
pos2 = pos.with_columns(((pl.col("cat_rank_in_feed") - 1)).alias("above_same"), (pl.col("rank") - pl.col("cat_rank_in_feed")).alias("above_other"))
print(pos2.select(pl.col("above_same").sum() / (pl.col("above_same") + pl.col("above_other")).sum()))
