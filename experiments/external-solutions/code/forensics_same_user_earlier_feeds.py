# Forensics probe 8: does a user's own earlier labelled train feed help on val? Builds five signals (item
# shown to this user before, the same weighted by its best production position, the user's click rate,
# click count and exposure by category) and reads each through the nested harness on the 8-student pool.
# Run on CPU from lab/nn_x/w12/forensics (it imports fx.py); the command line is not on record.
# Output saved as p08.log (in ../results/forensics_same_user_earlier_feeds.log).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/forensics/p08_transfer.py
#
from fx import *
T=FLAT/"train"
ftr=pl.read_parquet(T/"feeds.parquet"); print(ftr.columns)
tu=ftr["user_id"].to_numpy(); tM=ftr["M"].to_numpy().astype(np.int64); toff=ftr["cand_off"].to_numpy(); tts=ftr["timestamp"].to_numpy()
ti=np.load(T/"cand_item_id.npy",mmap_mode="r"); ty=np.load(T/"cand_target.npy",mmap_mode="r"); tc=np.load(T/"cand_category_id.npy",mmap_mode="r"); tp=np.load(T/"cand_pos.npy",mmap_mode="r")
tf4=np.load(T/"cand_f4.npy",mmap_mode="r")
meta={s:feed_meta(s) for s in ("val","test")}
allu=np.unique(np.r_[meta["val"][0]["user_id"].to_numpy(),meta["test"][0]["user_id"].to_numpy()]); allu=allu[allu!=0]
sel=np.nonzero(np.isin(tu,allu))[0]; print("train feeds of val/test users",len(sel))
rows=np.concatenate([np.arange(toff[i],toff[i]+tM[i]) for i in sel])
B=pl.DataFrame({"u":np.repeat(tu[sel],tM[sel]),"ts0":np.repeat(tts[sel],tM[sel]),"item":ti[rows],"y":ty[rows].astype(np.int32),"c":tc[rows].astype(np.int32),"pos":tp[rows].astype(np.int32),"M0":np.repeat(tM[sel],tM[sel]).astype(np.int32)})
# add val as bank for test
fv,Mv,offv=meta["val"]; yv=np.load(FLAT/"val/cand_target.npy")
Bv=pl.DataFrame({"u":np.repeat(fv["user_id"].to_numpy(),Mv),"ts0":np.repeat(fv["timestamp"].to_numpy(),Mv),"item":np.load(FLAT/"val/cand_item_id.npy"),"y":yv.astype(np.int32),"c":np.load(FLAT/"val/cand_category_id.npy").astype(np.int32),"pos":np.full(len(yv),-1,np.int32),"M0":np.repeat(Mv,Mv).astype(np.int32)}).filter(pl.col("u")!=0)
B=B.with_columns(pl.col("u").cast(pl.Int64),pl.col("item").cast(pl.Int64)); Bv=Bv.with_columns(pl.col("u").cast(pl.Int64),pl.col("item").cast(pl.Int64))
print("bank rows",B.height,"pos rate",B["y"].mean())
out={}
for split in ("val","test"):
    f,M,off=meta[split]; n=len(M); fid=np.repeat(np.arange(n),M)
    bank=B if split=="val" else pl.concat([B,Bv])
    C=pl.DataFrame({"row":np.arange(M.sum()),"f":fid,"u":np.repeat(f["user_id"].to_numpy(),M).astype(np.int64),"ts":np.repeat(f["timestamp"].to_numpy(),M),"item":np.load(FLAT/f"{split}/cand_item_id.npy").astype(np.int64),"c":np.load(FLAT/f"{split}/cand_category_id.npy").astype(np.int32)})
    # exact pair
    pa=bank.group_by(["u","item"]).agg(pl.len().alias("n_shown"),pl.col("y").sum().alias("n_click"),pl.col("pos").min().alias("minpos"),(pl.col("pos")/pl.col("M0")).min().alias("minrel"))
    uc=bank.group_by(["u","c"]).agg(pl.len().alias("uc_n"),pl.col("y").sum().alias("uc_y"))
    uu=bank.group_by("u").agg(pl.len().alias("u_n"),pl.col("y").sum().alias("u_y"),pl.col("ts0").n_unique().alias("u_feeds"))
    C=C.join(pa,on=["u","item"],how="left").join(uc,on=["u","c"],how="left").join(uu,on="u",how="left").sort("row")
    C=C.with_columns(pl.col("n_shown").fill_null(0),pl.col("n_click").fill_null(0),pl.col("uc_n").fill_null(0),pl.col("uc_y").fill_null(0),pl.col("u_n").fill_null(0),pl.col("u_y").fill_null(0))
    # signals
    r0=(C["u_y"]+1)/(C["u_n"]+50)
    C=C.with_columns(
        (pl.col("n_shown")>0).cast(pl.Float64).alias("s_shown"),
        pl.when(pl.col("n_shown")>0).then(1-pl.col("minrel").fill_null(0.5)).otherwise(0).alias("s_shown_toprel"),
        # user-category click lift (shrunk): log((uc_y+a*r)/(uc_n+a)) - log r
        ((pl.col("uc_y")+20*(pl.col("u_y")+1)/(pl.col("u_n")+50))/(pl.col("uc_n")+20)).log().alias("s_uc_ctr"),
        (pl.col("uc_y")+0.0).log1p().alias("s_uc_clicks"),
        ((pl.col("uc_n")+1)/(pl.col("u_n")+50)).log().alias("s_uc_expo"),
    )
    hasu=(C["u_n"]>0).to_numpy()
    feed_has=np.bincount(fid,hasu,n)>0; feed_pair=np.bincount(fid,C["s_shown"].to_numpy(),n)>0
    ts=f["timestamp"].to_numpy()
    if split=="test":
        pub=ts<PUB_TS
        print("test coverage: user has earlier labelled feeds: public %.4f private %.4f ; feeds with >=1 re-shown pair: public %.4f private %.4f ; rows re-shown %.5f"%(feed_has[pub].mean(),feed_has[~pub].mean(),feed_pair[pub].mean(),feed_pair[~pub].mean(),C["s_shown"].mean()))
        print("  n_click>0 rows",(C["n_click"]>0).sum())
    else:
        y=yv; uid=f["user_id"].to_numpy()
        _,_,_,ft,to=load_w8("val"); base=0.5*zfeed(ft,off,M)+0.5*zfeed(to,off,M)
        print("val coverage: user has earlier labelled feeds %.4f ; feeds with re-shown pair %.4f ; rows re-shown %.5f ; n_click>0 rows %d"%(feed_has.mean(),feed_pair.mean(),C["s_shown"].mean(),(C["n_click"]>0).sum()))
        sh=C["s_shown"].to_numpy()>0
        print("  target rate re-shown %.4f (n %d) vs not, same feeds %.4f"%(y[sh].mean(),sh.sum(),y[(~sh)&feed_pair[fid]].mean()))
        # model-relative: mean base z of re-shown rows, and residual: positives rate vs expected by softmax of base? use within-feed: compare y among reshown vs matched by base z decile
        bz=base; print("  mean W8 z of re-shown rows %.3f"%bz[sh].mean())
        for s in ("s_shown","s_shown_toprel","s_uc_ctr","s_uc_clicks","s_uc_expo"):
            x=C[s].to_numpy().astype(np.float64); z=zfeed(x,off,M); z[~feed_has[fid]]=0
            d,se,wc,(wb,db)=nested(base,z,y,off,M,uid)
            print("  %-16s nested %+.5f +- %.5f  w=%s  in-sample best w=%s %+.5f ; on covered feeds %+.5f"%(s,d,se,wc,wb,db,db/ max(feed_has.mean(),1e-9)))
