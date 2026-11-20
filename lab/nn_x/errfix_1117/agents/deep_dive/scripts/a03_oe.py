"""Observed/expected positive rate by feature bins, expected = rate of (rank bucket x M bucket) cell."""
from util import *

df = load()
rb = np.digitize(df["rank"].to_numpy(), [2, 3, 4, 6, 11, 21, 31, 51, 101, 151])
mb = np.digitize(df["M"].to_numpy(), [30, 60, 120, 200])
cell = rb * 10 + mb
y = df["label"].to_numpy().astype(float)
cs = np.bincount(cell, weights=y, minlength=200) / np.maximum(np.bincount(cell, minlength=200), 1)
E = cs[cell]
df = df.with_columns(pl.Series("E", E), pl.Series("rb", rb))
fl = pl.read_parquet("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/feedlevel.parquet")
df = df.join(fl.select("fidx", "gap12", "lab_r1"), on="fidx", how="left")
df = df.with_columns((pl.col("rank") <= 10).alias("top10"))
df.write_parquet("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/withE.parquet")


def oe(expr, name, by_top=True):
    d = df.with_columns(expr.alias("bin"))
    keys = ["bin", "top10"] if by_top else ["bin"]
    g = d.group_by(keys).agg(pl.len().alias("n"), pl.col("label").sum().alias("O"), pl.col("E").sum().alias("E"))
    g = g.with_columns((pl.col("O") / pl.col("E")).alias("OE"),
                       ((pl.col("O") - pl.col("E")) / pl.col("E").sqrt()).alias("zz")).sort(keys)
    print(f"--- {name}")
    print(g)


pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(200)
oe(pl.col("xp_imp_c50") == 0, "cold xp=0")
oe(pl.col("f4").qcut(10, allow_duplicates=True), "f4 decile")
oe(pl.col("n_hdup95").clip(0, 50).cut([0, 1, 3, 10, 30]), "n_hdup95")
oe(pl.col("hmax").cut([0.8, 0.9, 0.95, 0.98, 0.99]), "hmax")
oe(pl.col("cat_user_rank").clip(0, 6), "cat_user_rank")
oe(pl.col("cat_n_in_feed").cut([1, 2, 5, 10, 30, 80]), "cat_n_in_feed")
oe(pl.col("same_cat_newest"), "same_cat_newest")
oe(pl.col("guest"), "guest")
oe(pl.col("f2_home"), "f2_home")
oe(pl.col("f5").qcut(5, allow_duplicates=True), "f5 q")
oe(pl.col("cand_maxcos").cut([0.8, 0.9, 0.95, 0.98]), "cand_maxcos")
oe(pl.col("n_cand_dup95").clip(0, 10).cut([0, 1, 3]), "n_cand_dup95")
