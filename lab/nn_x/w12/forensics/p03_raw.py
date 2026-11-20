import os; os.environ["OMP_NUM_THREADS"]="12"; os.environ["POLARS_MAX_THREADS"]="12"
import sys; sys.path.insert(0,"<repo>/src")
import numpy as np, polars as pl, pyarrow.parquet as pq, glob
from scipy.stats import spearmanr
R="<repo>/data/competition/"
for split in ["val","test"]:
    files=sorted(glob.glob(R+f"{split}/impressions/*.parquet"))
    md=pq.ParquetFile(files[0]); print(split,len(files),md.schema_arrow.names); print(" kv meta:",{k:(v[:200] if len(v)>200 else v) for k,v in (md.schema_arrow.metadata or {}).items()}); print(" created_by",md.metadata.created_by,"rowgroups",md.metadata.num_row_groups, "rows",md.metadata.num_rows)
    cols=["feed_id","timestamp","user_id","item_id","item_id__embedding_idx","history__item_id"]+(["target"] if split=="val" else [])
    dfs=[]
    for pi,fp in enumerate(files):
        d=pl.read_parquet(fp,columns=cols).with_columns(pl.lit(pi).alias("part"),pl.int_range(pl.len()).alias("row"))
        dfs.append(d)
    df=pl.concat(dfs).with_columns(pl.int_range(pl.len()).alias("g"))
    n=df.height
    fid=df["feed_id"].to_numpy(); ts=df["timestamp"].to_numpy(); uid=df["user_id"].to_numpy(); part=df["part"].to_numpy(); g=df["g"].to_numpy()
    M=df["item_id"].list.len().to_numpy(); H=df["history__item_id"].list.len().to_numpy()
    print(" feed_id range",fid.min(),fid.max(),"n",n,"unique",len(np.unique(fid)))
    print(" spearman global row vs feed_id %.4f ts %.4f uid %.4f M %.4f H %.4f"%tuple(spearmanr(g,x)[0] for x in (fid,ts,uid,M,H)))
    print(" spearman feed_id vs ts %.4f uid %.4f M %.4f H %.4f"%tuple(spearmanr(fid,x)[0] for x in (ts,uid,M,H)))
    # within-part row order
    r=df["row"].to_numpy()
    for nm,x in (("feed_id",fid),("ts",ts),("uid",uid),("M",M)):
        cs=[spearmanr(r[part==p],x[part==p])[0] for p in range(len(files))]
        print("  within-part spearman row vs",nm,"mean %.4f sd %.4f"%(np.mean(cs),np.std(cs)))
    print(" part sizes",np.bincount(part)[:8],"feed_id%nparts==part?",(fid%len(files)==part).mean())
    # is part a hash of feed_id / user?
    for nm,x in (("feed_id",fid),("uid",uid),("ts",ts)):
        dd=pl.DataFrame({"x":x,"p":part}).group_by("x").agg(pl.col("p").n_unique().alias("k"),pl.len().alias("c")).filter(pl.col("c")>1)
        print("  key",nm,"groups with >1 rows",dd.height,"share single-part",(dd["k"]==1).mean() if dd.height else None)
    if split=="val":
        npos=df["target"].list.sum().to_numpy()
        print(" spearman npos vs global row %.4f feed_id %.4f; npos/M vs feed_id %.4f"%(spearmanr(g,npos)[0],spearmanr(fid,npos)[0],spearmanr(fid,npos/M)[0]))
        # position of positives within stored list
        ex=df.select("item_id","target","item_id__embedding_idx").with_columns(pl.int_range(pl.len()).alias("f")).explode(["item_id","target","item_id__embedding_idx"])
        ex=ex.with_columns((pl.int_range(pl.len()).over("f")).alias("j"),pl.len().over("f").alias("M"))
        ex=ex.with_columns(((pl.col("j")+0.5)/pl.col("M")).alias("rel"),
                           ((pl.col("item_id__embedding_idx").rank().over("f")-0.5)/pl.col("M")).alias("erel"),
                           ((pl.col("item_id").rank().over("f")-0.5)/pl.col("M")).alias("irel"))
        p=ex.filter(pl.col("target")==1)
        for c in ("rel","erel","irel"):
            v=p[c].to_numpy(); print("  positives mean",c,"%.4f +- %.4f"%(v.mean(),v.std()/np.sqrt(len(v))), "deciles",np.round(np.histogram(v,bins=10,range=(0,1))[0]/len(v),4))
    # embedding table referencing
    N=int(open(R+f"{split}/item_embeddings/data.meta").read().split('[')[1].split(',')[0])
    ce=df["item_id__embedding_idx"].explode().to_numpy(); ci=df["item_id"].explode().to_numpy()
    hd=pl.read_parquet(files,columns=["history__item_id","history__item_id__embedding_idx"])
    he=hd["history__item_id__embedding_idx"].explode().drop_nulls().to_numpy(); hi=hd["history__item_id"].explode().drop_nulls().to_numpy()
    ref=np.zeros(N,bool); ref[ce[ce>=0]]=True; refc=ref.copy(); ref[he[he>=0]]=True
    print(" emb table N",N,"referenced",ref.sum(),"by cands",refc.sum(),"unreferenced",N-ref.sum())
    # mapping idx<->item
    m=pl.DataFrame({"e":np.r_[ce,he],"i":np.r_[ci,hi]}).filter(pl.col("e")>=0).unique()
    print(" unique (idx,item) pairs",m.height,"unique idx",m["e"].n_unique(),"unique item",m["i"].n_unique())
    ms=m.sort("e"); e=ms["e"].to_numpy(); it=ms["i"].to_numpy()
    print(" spearman idx vs item_id %.4f"%spearmanr(e,it)[0], "monotone frac",(np.diff(it)>0).mean())
    # first-seen order: global row of first occurrence as candidate
    fe=pl.DataFrame({"e":ce,"g":np.repeat(g,M),"t":np.repeat(ts,M)}).filter(pl.col("e")>=0).group_by("e").agg(pl.col("g").min(),pl.col("t").min().alias("tmin"),pl.len().alias("n"))
    print(" spearman idx vs first row %.4f, vs first ts %.4f, vs ncand %.4f"%(spearmanr(fe["e"],fe["g"])[0],spearmanr(fe["e"],fe["tmin"])[0],spearmanr(fe["e"],fe["n"])[0]))
    isc=np.zeros(N,bool); isc[ce[ce>=0]]=True; ish=np.zeros(N,bool); ish[he[he>=0]]=True
    print(" idx quartile share cand-only / hist-only / both:",[ (round((isc&~ish)[a:b].mean(),3),round((~isc&ish)[a:b].mean(),3),round((isc&ish)[a:b].mean(),3)) for a,b in [(0,N//4),(N//4,N//2),(N//2,3*N//4),(3*N//4,N)]])
    np.save(f"embref_{split}.npy",ref)
