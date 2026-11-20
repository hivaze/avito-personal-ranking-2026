import numpy as np, polars as pl
FL="<repo>/data/flat/val/"
MSK=3*3600
f=pl.read_parquet(FL+"feeds.parquet")
ts=f["timestamp"].to_numpy().astype(np.int64); uid=f["user_id"].to_numpy()
cf=np.load(FL+"cand_fidx.npy"); y=np.load(FL+"cand_target.npy")
c=pl.DataFrame({"row":np.arange(len(cf)),"fq":cf,"item":np.load(FL+"cand_item_id.npy"),"tq":ts[cf],"uq":uid[cf],"y":y})
hf=np.load(FL+"hist_fidx.npy"); hp=np.load(FL+"hist_hpos.npy")
m=hp<10
h=pl.DataFrame({"fe":hf[m],"item":np.load(FL+"hist_item_id.npy")[m],"te":ts[hf[m]],"ue":uid[hf[m]],"et":np.load(FL+"hist_event_type.npy")[m]})
j=c.join(h,on="item")
j=j.filter((pl.col("fe")!=pl.col("fq")) & ((pl.col("ue")!=pl.col("uq"))|(pl.col("uq")==0)))
j=j.with_columns(((pl.col("tq")+MSK)//86400==(pl.col("te")+MSK)//86400).alias("sameday"))
j=j.filter(pl.col("sameday"))
g=j.group_by("row").agg((pl.col("te")>pl.col("tq")).sum().alias("fut"),(pl.col("te")<pl.col("tq")).sum().alias("past"),(pl.col("te")==pl.col("tq")).sum().alias("same"),
   ((pl.col("te")>pl.col("tq"))&(pl.col("et")==1)).sum().alias("fut_e1"),(pl.col("uq")==0).first().alias("u0"))
c=c.join(g,on="row",how="left").fill_null(0)
base=y.mean()
print("base pos rate",round(base,5),"rows",len(c))
for nm,cond in [("any",pl.col("fut")+pl.col("past")+pl.col("same")>0),("future only",(pl.col("fut")>0)&(pl.col("past")==0)),("past only",(pl.col("past")>0)&(pl.col("fut")==0)),
                ("future e1",pl.col("fut_e1")>0),("user0 any",(pl.col("u0"))&(pl.col("fut")+pl.col("past")>0))]:
    s=c.filter(cond); print(f"{nm:12s} n={len(s):6d} share={len(s)/len(c):.4f} pos_rate={s['y'].mean() if len(s) else float('nan'):.4f} npos={int(s['y'].sum())}")
# within-feed: does the positive have higher fut than negatives in the same feed? (tiny-coverage) 
