"""s08: is type 10 a platform marker? O/E(type10 present, anonymous) within activity strata; source shares of
type-1 views in feeds with vs without type-10 events and anonymous vs logged-in (val+test)."""
import sys, numpy as np, polars as pl
sys.path.insert(0, "<repo>/nn_x/codes/sources")
from scommon import load_hist, pshow
v, fv = load_hist("val"); t, ft = load_hist("test")
df = pl.concat([v, t.with_columns(pl.col("fidx") + 10_000_000)]); fs = pl.concat([fv, ft.with_columns(pl.col("fidx") + 10_000_000)])
ff = fs.select("fidx", "user_id", "H").join(df.group_by("fidx").agg((pl.col("et") == 10).any().alias("t10"), pl.col("et").is_in([5, 6, 8]).any().alias("msg"), pl.len().alias("n")), on="fidx", how="left").fill_null(False)
ff = ff.with_columns((pl.col("user_id") == 0).alias("anon"), pl.col("H").qcut(10, allow_duplicates=True).alias("hb"))
oe = ff.group_by("hb").agg(pl.len().alias("n"), (pl.col("t10") & pl.col("anon")).sum().alias("o"), (pl.col("t10").mean() * pl.col("anon").mean() * pl.len()).alias("e"))
print("O/E(type10 & anonymous) within H deciles:", float(oe["o"].sum() / oe["e"].sum()), " P(t10|anon)", float(ff.filter(pl.col("anon"))["t10"].mean()), " P(t10|logged)", float(ff.filter(~pl.col("anon"))["t10"].mean()))
v1 = df.filter(pl.col("et") == 1).join(ff.select("fidx", "t10", "anon"), on="fidx")
for col in ["t10", "anon"]:
    s = v1.group_by([col, "es"]).len().with_columns((pl.col("len") / pl.col("len").sum().over(col)).alias("p")).filter(pl.col("es") <= 16)
    pshow(s.pivot(values="p", index="es", on=col, sort_columns=True).sort("es").with_columns((pl.col("true") / pl.col("false")).alias("ratio")))
