"""s06: categories with / without message-like events (types 5/6/8), and each source's share of type-1 views in each group (val+test)."""
import sys, numpy as np, polars as pl
sys.path.insert(0, "<repo>/nn_x/codes/sources")
from scommon import load_hist, pshow, OUT
v, _ = load_hist("val"); t, _ = load_hist("test")
df = pl.concat([v, t])
cm = df.group_by("cat").agg(pl.len().alias("n"), pl.col("et").is_in([5, 6, 8]).mean().alias("msg_share"), (pl.col("et") == 4).mean().alias("t4_share"),
                            (pl.col("et") == 7).mean().alias("t7_share"), pl.col("et").is_in([2, 3]).mean().alias("contact_share")).sort("n", descending=True)
pshow(cm, 60)
msgcats = cm.filter(pl.col("msg_share") > 0.001)["cat"].to_list()
print("categories WITH 5/6/8 events:", sorted(msgcats), "event share of these cats:", float(cm.filter(pl.col("cat").is_in(msgcats))["n"].sum() / cm["n"].sum()))
v1 = df.filter(pl.col("et") == 1).with_columns(pl.col("cat").is_in(msgcats).alias("msgcat"))
r = v1.group_by("es").agg(pl.len().alias("n_views"), pl.col("msgcat").mean().alias("share_views_in_msg_cats")).sort("es").filter(pl.col("es") <= 22)
pshow(r)
r.write_csv(f"{OUT}/s06_msgcat_share.csv")
cm.write_csv(f"{OUT}/s06_cat_eventmix.csv")
