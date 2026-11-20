"""Feed-relative category standing: is the item's category the most recent / most frequent (by the user's
history) among the categories PRESENT in this feed? The model sees absolute values (cat_newest_idx,
cat_user_rank) per candidate; the feed-relative version needs the whole feed."""
import numpy as np
import polars as pl
import cu

SP = "<repo>/lab/nn_x/errfix_1117/agents/category/work"
C = cu.Ctx()
df = C.df
s = C.s
E = np.load(f"{SP}/E_flex.npy")
y = C.y.astype(float)
d = df.with_columns(
    cni_eff=pl.when(pl.col("cat_newest_idx") < 0).then(9999).otherwise(pl.col("cat_newest_idx")),
    ur_eff=pl.when(pl.col("cat_user_rank") == 0).then(9999).otherwise(pl.col("cat_user_rank")),
)
d = d.with_columns(
    min_cni=pl.col("cni_eff").min().over("fidx"), min_ur=pl.col("ur_eff").min().over("fidx"),
    has_newest=pl.col("same_cat_newest").any().over("fidx"),
)
# dense rank of the category's recency among present categories (1 = most recent present)
cr = (d.select("fidx", "cat", "cni_eff", "ur_eff").unique(["fidx", "cat"])
      .with_columns(rec_rank=pl.col("cni_eff").rank("dense").over("fidx"),
                    fq_rank=pl.col("ur_eff").rank("dense").over("fidx")))
d = d.join(cr.select("fidx", "cat", "rec_rank", "fq_rank"), on=["fidx", "cat"], how="left", maintain_order="left")
d = d.with_columns(pl.Series("E2", E))
mrp = (d["cni_eff"] == d["min_cni"]).to_numpy() & (d["min_cni"] < 9999).to_numpy()
pl.Config.set_tbl_rows(60)


def oe(dd, by):
    return (dd.group_by(by).agg(n=pl.len(), feeds=pl.col("fidx").n_unique(), O=pl.col("label").sum(), E=pl.col("E2").sum())
            .with_columns(OE=pl.col("O") / pl.col("E"), z=(pl.col("O") - pl.col("E")) / pl.col("E").sqrt()).sort(by))


d = d.with_columns(rr=pl.col("rec_rank").clip(1, 5), fr=pl.col("fq_rank").clip(1, 5),
                   rb=pl.when(pl.col("rank") <= 3).then(1).when(pl.col("rank") <= 10).then(2).when(pl.col("rank") <= 30).then(3).otherwise(4))
print("no newest-category item in feed: by recency rank among present categories")
print(oe(d.filter(~pl.col("has_newest")), ["rr"]))
print(oe(d.filter(~pl.col("has_newest")), ["rr", "rb"]))
print("by frequency rank among present categories, all feeds")
print(oe(d, ["fr"]))
print(oe(d.filter(pl.col("ur_eff") > 1), ["fr"]))
hn = d["has_newest"].to_numpy()
rec1 = (d["rec_rank"].to_numpy() == 1) & (d["min_cni"].to_numpy() < 9999)
fq1 = (d["fq_rank"].to_numpy() == 1) & (d["min_ur"].to_numpy() < 9999)
ws = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
C.crossfit(ws, lambda w: s + w * (rec1 & ~hn), label="most-recent present cat, no newest-cat item in feed")
C.crossfit(ws, lambda w: s + w * fq1, label="most-frequent present cat")
C.crossfit(ws, lambda w: s + w * (fq1 & (d["ur_eff"].to_numpy() > 1)), label="most-frequent present cat when user's #1 absent")
print("variants", C.n_tried)
