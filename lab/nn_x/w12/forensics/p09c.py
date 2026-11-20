from fx import *
FL=0.0162
def load(s):
    f,M,off=feed_meta(s)
    return pl.DataFrame({"row":np.arange(M.sum()),"f":np.repeat(np.arange(len(M)),M),"u":np.repeat(f["user_id"].to_numpy(),M),"ts":np.repeat(f["timestamp"].to_numpy(),M).astype(np.int64),"item":np.load(FLAT/f"{s}/cand_item_id.npy"),"f5":np.load(FLAT/f"{s}/cand_f5.npy").astype(np.float64)})
V=load("val"); T=load("test")
def sig(C,bank):
    b=bank.select("item",pl.col("ts").alias("ts2"),pl.col("f5").alias("f5b"),pl.col("row").alias("row2"))
    j=C.select("row","item","ts","f5").join(b,on="item",how="inner").filter(pl.col("ts2")>pl.col("ts"))
    g=j.group_by("row").agg(pl.col("f5").first(),
        (pl.col("f5b")<=FL).any().alias("later_floor"),                      # some later obs still at floor
        pl.col("ts2").filter(pl.col("f5b")<=FL).max().alias("t_lastfloor"),
        pl.col("ts2").filter(pl.col("f5b")>FL).min().alias("t_firstup"),
        pl.col("ts2").min().alias("t_next"),pl.col("ts").first(),pl.len().alias("n"),
        (pl.col("f5b")<pl.col("f5")).any().alias("later_lower"),(pl.col("f5b").min()).alias("f5min"))
    return g
f,M,off,ft,to=load_w8("val"); y=np.load(FLAT/"val/cand_target.npy"); uid=f["user_id"].to_numpy(); n=len(M); fid=np.repeat(np.arange(n),M)
base=0.5*zfeed(ft,off,M)+0.5*zfeed(to,off,M)
f5v=V["f5"].to_numpy()
print("val rows at floor %.4f target rate %.4f"%((f5v<=FL).mean(),y[f5v<=FL].mean()))
for tag,bank in (("val-only bank",V),("val+test bank",pl.concat([V,T.with_columns(pl.col("row")+10**8)]))):
    g=sig(V,bank); r=g["row"].to_numpy(); now=g["f5"].to_numpy(); lf=g["later_floor"].to_numpy()
    A=(now<=FL)&lf           # floor now, still floor later -> not clicked
    Bm=(now<=FL)&~lf         # floor now, all later obs above floor
    gapup=((g["t_firstup"]-g["ts"]).to_numpy())/3600.
    print(f"\n[{tag}] floor rows with a later obs: {((now<=FL)).sum()} ({(now<=FL).sum()/len(y):.4f} of rows)")
    print("  still-floor-later: n %d rate %.5f | up-later: n %d rate %.4f ; up within 6h n %d rate %.4f ; up within 24h n %d rate %.4f"%(A.sum(),y[r[A]].mean(),Bm.sum(),y[r[Bm]].mean(),(Bm&(gapup<6)).sum(),y[r[Bm&(gapup<6)]].mean(),(Bm&(gapup<24)).sum(),y[r[Bm&(gapup<24)]].mean()))
    print("  W8 z of still-floor rows: mean %.3f ; positives among them %d ; feeds touched %d"%(base[r[A]].mean(),y[r[A]].sum(),len(np.unique(fid[r[A]]))))
    # generic: later obs lower than now (rate fell => impressions w/o click)
    nf=now>FL; lower=g["later_lower"].to_numpy()
    print("  non-floor rows: later lower n %d rate %.4f | never lower n %d rate %.4f"%((nf&lower).sum(),y[r[nf&lower]].mean(),(nf&~lower).sum(),y[r[nf&~lower]].mean()))
    for nm,x_ in (("demote still-floor",-(A.astype(float))),("floor: up(+1)/still(-1)",Bm.astype(float)-A.astype(float)),("floor: up within 24h",(Bm&(gapup<24)).astype(float)),("hard demote still-floor (w fixed)",None)):
        if x_ is None:
            s2=base.copy(); s2[r[A]]-=100; p0=ndcg_per_feed(base,y,off,M); p1=ndcg_per_feed(s2,y,off,M); d=p1-p0
            print("  %-34s delta %+.5f +- %.5f (no tuning)"%(nm,d.mean(),d.std()/np.sqrt(n))); continue
        x=np.zeros(len(y)); x[r]=x_
        # use raw (not z) so that effect is per-row constant
        dl,se,wc,(wb,db)=nested(base,x,y,off,M,uid,ws=(0,0.1,0.25,0.5,1,2,4,8))
        print("  %-34s nested %+.5f +- %.5f w=%s in-sample w=%s %+.5f"%(nm,dl,se,wc,wb,db))
# test coverage
ft_,Mt,offt=feed_meta("test"); pub=(ft_["timestamp"].to_numpy()<PUB_TS); fidt=np.repeat(np.arange(len(Mt)),Mt); pr=pub[fidt]
g=sig(T,T); r=g["row"].to_numpy(); now=g["f5"].to_numpy(); lf=g["later_floor"].to_numpy(); A=(now<=FL)&lf; Bm=(now<=FL)&~lf
f5t=T["f5"].to_numpy()
print("\ntest floor share: public %.4f private %.4f"%((f5t[pr]<=FL).mean(),(f5t[~pr]<=FL).mean()))
print("test still-floor-later rows: public %.5f private %.5f ; up-later rows: public %.5f private %.5f"%(pr[r[A]].sum()/pr.sum(),(~pr[r[A]]).sum()/(~pr).sum(),pr[r[Bm]].sum()/pr.sum(),(~pr[r[Bm]]).sum()/(~pr).sum()))
g=sig(V,V); r=g["row"].to_numpy(); now=g["f5"].to_numpy(); lf=g["later_floor"].to_numpy(); A=(now<=FL)&lf; Bm=(now<=FL)&~lf
print("val (val-only bank) still-floor-later rows %.5f ; up-later rows %.5f"%(A.sum()/len(y),Bm.sum()/len(y)))
