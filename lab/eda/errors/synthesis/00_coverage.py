import os; os.environ["POLARS_MAX_THREADS"]="16"
import numpy as np, polars as pl, time
F="<repo>/data/flat"
t0=time.time()
tr_item=np.load(f"{F}/train/cand_item_id.npy", mmap_mode="r")
tr_t=np.load(f"{F}/train/cand_target.npy", mmap_mode="r")
df=pl.DataFrame({"item":np.asarray(tr_item),"t":np.asarray(tr_t).astype(np.int8)})
agg=df.group_by("item").agg(pl.len().alias("n"), pl.col("t").sum().alias("c"))
print("train cand rows",len(df),"unique items",len(agg),"items with click",(agg["c"]>0).sum(), f"{time.time()-t0:.0f}s")
for sp in ["val","test"]:
    it=np.load(f"{F}/{sp}/cand_item_id.npy")
    d=pl.DataFrame({"item":it}).join(agg,on="item",how="left").fill_null(0)
    n=d["n"].to_numpy(); c=d["c"].to_numpy()
    msg=f"{sp}: rows {len(it)} share with train impr {np.mean(n>0):.3f}  n>=5 {np.mean(n>=5):.3f} n>=20 {np.mean(n>=20):.3f}  with train click {np.mean(c>0):.3f}"
    if sp=="val":
        y=np.load(f"{F}/val/cand_target.npy")
        msg+=f" | positives: impr>0 {np.mean(n[y==1]>0):.3f} n>=5 {np.mean(n[y==1]>=5):.3f} click>0 {np.mean(c[y==1]>0):.3f}; median n|n>0 {np.median(n[n>0])}"
    print(msg)
# history items of train (other users' histories) coverage
