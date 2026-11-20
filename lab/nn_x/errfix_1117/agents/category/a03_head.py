"""Head (ranks 1-10) calibration of same-category-as-newest, by newest event type/source and history state."""
import numpy as np
import polars as pl
import cu

SP = "<repo>/lab/nn_x/errfix_1117/agents/category/work"
df = pl.read_parquet(f"{SP}/calib.parquet").with_columns(pl.Series("E2", np.load(f"{SP}/E_flex.npy")))
pl.Config.set_tbl_rows(400)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_cols(20)
print(df.select((pl.col("same_cat_newest") == (pl.col("cat_newest_idx") == 0)).mean()))


def oe(d, by, minE=10):
    return (d.group_by(by).agg(n=pl.len(), O=pl.col("label").sum(), E=pl.col("E2").sum(), rate=pl.col("label").mean())
            .with_columns(OE=pl.col("O") / pl.col("E"), z=(pl.col("O") - pl.col("E")) / pl.col("E").sqrt())
            .filter(pl.col("E") >= minE).sort(by))


h = df.filter(pl.col("rank") <= 10)
print(oe(h, ["rank", "same_cat_newest"]))
h3 = df.filter(pl.col("rank") <= 3)
print(oe(h3, ["newest_src", "same_cat_newest"]))
print(oe(h3, ["newest_type", "same_cat_newest"]))
# how many top-3 slots in feed are same-cat-newest
fe = h3.group_by("fidx").agg(k=pl.col("same_cat_newest").sum())
h3 = h3.join(fe, on="fidx")
print(oe(h3, ["k", "same_cat_newest"]))
print(oe(h3.with_columns(Hb=pl.col("n_hist_cats32").clip(0, 8)), ["Hb", "same_cat_newest"]))
print(oe(h3.with_columns(Mb=(pl.col("M") // 50)), ["Mb", "same_cat_newest"]))
