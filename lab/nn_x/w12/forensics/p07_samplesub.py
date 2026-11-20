import os; os.environ["POLARS_MAX_THREADS"]="12"
import numpy as np, polars as pl, glob, pyarrow.parquet as pq
from scipy.stats import spearmanr
R="<repo>/data/competition/"
pf=pq.ParquetFile(R+"sample_submission.parquet"); print(pf.metadata.created_by, pf.metadata.num_row_groups, pf.schema_arrow, pf.schema_arrow.metadata)
s=pl.read_parquet(R+"sample_submission.parquet"); print(s.height, s["score"].min(), s["score"].max(), s["score"].n_unique())
files=sorted(glob.glob(R+"test/impressions/*.parquet"))
t=pl.concat([pl.read_parquet(fp,columns=["feed_id","item_id","timestamp","user_id"]).with_columns(pl.lit(i).alias("part")) for i,fp in enumerate(files)]).with_columns(pl.int_range(pl.len()).alias("g"))
te=t.explode("item_id").with_columns(pl.int_range(pl.len()).over("feed_id").alias("j"))
s=s.with_columns(pl.int_range(pl.len()).alias("r"))
# feed contiguity in sample sub
fs=s.group_by("feed_id",maintain_order=True).agg(pl.col("r").min().alias("r0"),pl.col("r").max().alias("r1"),pl.len().alias("m"))
print("feeds",fs.height,"contiguous",((fs["r1"]-fs["r0"]+1)==fs["m"]).all())
j=fs.with_columns(pl.int_range(pl.len()).alias("sf")).join(t.select("feed_id","g","timestamp","user_id","part"),on="feed_id")
print("feed order sample vs raw global row: spearman %.4f ; vs ts %.4f ; vs feed_id %.4f; vs uid %.4f; vs M %.4f"%tuple(spearmanr(j["sf"],j[c])[0] for c in ("g","timestamp","feed_id","user_id","m")))
print("same order as raw:",(j.sort("sf")["g"].to_numpy()==np.arange(j.height)).mean())
s2=s.with_columns((pl.col("r")-pl.col("r").min().over("feed_id")).alias("js")).join(te.select("feed_id","item_id","j"),on=["feed_id","item_id"],how="left")
print("null joins",s2["j"].null_count(),"within-feed order identical frac",(s2["js"]==s2["j"]).mean())
# dup (feed,item)
print("dup pairs in sample",s.height-s.select("feed_id","item_id").unique().height)
c=[spearmanr(g["js"],g["j"])[0] for g in s2.filter(pl.col("feed_id").is_in(fs["feed_id"][:300].to_list())).partition_by("feed_id") if g.height>5]
print("within-feed spearman sample-pos vs raw-pos: mean %.4f"%np.mean(c))
c=[spearmanr(g["js"],g["item_id"])[0] for g in s2.filter(pl.col("feed_id").is_in(fs["feed_id"][:300].to_list())).partition_by("feed_id") if g.height>5]
print("within-feed spearman sample-pos vs item_id: mean %.4f"%np.mean(c))
