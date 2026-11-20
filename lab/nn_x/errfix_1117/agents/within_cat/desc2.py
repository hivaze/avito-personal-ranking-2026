from wc import *
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20)
df = load(); d = add_expect(features(df))
d = d.with_columns(pl.col("n_cand_dup95").clip(0, 10).alias("dupc"),
                   (pl.col("cand_maxcos") * 50).floor().clip(40, 50).alias("cmc"))
print(oe(d, "dupc"))
print(oe(d, "cmc"))
# dup x rank band
d = d.with_columns(pl.when(pl.col("rank") <= 10).then(10).when(pl.col("rank") <= 30).then(30).when(pl.col("rank") <= 100).then(100).otherwise(300).alias("rb"),
                   pl.when(pl.col("n_cand_dup95") == 0).then(0).when(pl.col("n_cand_dup95") <= 2).then(2).otherwise(9).alias("dupb"))
print(oe(d, ["rb", "dupb"]))
