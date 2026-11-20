# Forensics probe 9, a leak-class channel sized in order to rule it out: feature f5 is a slowly updated item
# counter, so the value another feed shows for the same item LATER tells whether the item was clicked in
# between. The probe joins every val row to later rows of the same item and reads sign and size of the
# change through the nested harness. This reads the future and was never an input of any model.
# Run on CPU from lab/nn_x/w12/forensics (it imports fx.py); the command line is not on record.
# Output saved as p09.log (in ../results/forensics_f5_forward_echo.log).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/forensics/p09_f5echo.py
#
from fx import *
D={}
for s in ("train","val","test"):
    f,M,off=feed_meta(s)
    if s=="train":
        # only last 3 days of train needed for 'earlier' context; skip train for later-echo
        continue
    D[s]=pl.DataFrame({"split":s,"row":np.arange(M.sum()),"f":np.repeat(np.arange(len(M)),M),"u":np.repeat(f["user_id"].to_numpy(),M),"ts":np.repeat(f["timestamp"].to_numpy(),M).astype(np.int64),"item":np.load(FLAT/f"{s}/cand_item_id.npy"),"f5":np.load(FLAT/f"{s}/cand_f5.npy").astype(np.float64),"f4":np.load(FLAT/f"{s}/cand_f4.npy").astype(np.float64)})
def later_feats(C,bank,tag):
    # for each row of C: later occurrences in bank (other feeds, ts>ts): next f5, max f5, count
    b=bank.select("item",pl.col("ts").alias("ts2"),pl.col("f5").alias("f5b"),pl.col("u").alias("u2"))
    j=C.select("row","item","ts","f5","u").join(b,on="item",how="inner").filter((pl.col("ts2")>pl.col("ts")))
    j=j.filter((pl.col("u2")!=pl.col("u"))|(pl.col("u")==0))
    g=j.sort("ts2").group_by("row").agg(pl.col("f5b").first().alias("f5_next"),pl.col("f5b").max().alias("f5_max"),pl.col("ts2").first().alias("ts_next"),pl.len().alias("n_later"),pl.col("f5").first(),pl.col("ts").first())
    return g
print("f5 value stats val: n unique",D["val"]["f5"].n_unique(),"share ==0 %.3f"%(D["val"]["f5"]==0).mean())
f,M,off,ft,to=load_w8("val"); y=np.load(FLAT/"val/cand_target.npy"); uid=f["user_id"].to_numpy(); n=len(M); fid=np.repeat(np.arange(n),M)
base=0.5*zfeed(ft,off,M)+0.5*zfeed(to,off,M)
for tag,bank in (("later in val only",D["val"]),("later in val+test",pl.concat([D["val"],D["test"]]))):
    g=later_feats(D["val"],bank,tag)
    r=g["row"].to_numpy(); d=(g["f5_next"]-g["f5"]).to_numpy(); gap=(g["ts_next"]-g["ts"]).to_numpy()/3600
    print(f"\n[{tag}] rows covered %.4f, feeds with >=1 covered %.4f"%(len(r)/len(y),len(np.unique(fid[r]))/n))
    for nm,m in (("all",np.ones(len(r),bool)),("gap<6h",gap<6),("gap<24h",gap<24),("gap>=24h",gap>=24)):
        if m.sum()<50: continue
        yy=y[r[m]]; dd=d[m]
        print("  %-8s n %6d  target rate: f5 up %.4f (n %d) | same %.4f (n %d) | down %.4f (n %d)"%(nm,m.sum(),yy[dd>0].mean(),(dd>0).sum(),yy[dd==0].mean() if (dd==0).any() else np.nan,(dd==0).sum(),yy[dd<0].mean(),(dd<0).sum()))
    for nm,fn in (("sign",lambda d,gap:np.sign(d)),("delta",lambda d,gap:d),("sign/gap<24h",lambda d,gap:np.sign(d)*(gap<24)),("delta*exp(-gap/12h)",lambda d,gap:d*np.exp(-gap/12))):
        x=np.zeros(len(y)); x[r]=fn(d,gap); z=zfeed(x,off,M)
        dl,se,wc,(wb,db)=nested(base,z,y,off,M,uid)
        print("  %-22s nested %+.5f +- %.5f w=%s  in-sample w=%s %+.5f"%(nm,dl,se,wc,wb,db))
# test coverage
ft_,Mt,offt=feed_meta("test"); tst=ft_["timestamp"].to_numpy(); pub=tst<PUB_TS; fidt=np.repeat(np.arange(len(Mt)),Mt)
g=later_feats(D["test"],D["test"],"test")
r=g["row"].to_numpy(); gap=(g["ts_next"]-g["ts"]).to_numpy()/3600; d=(g["f5_next"]-g["f5"]).to_numpy()
pr=pub[fidt]
print("\ntest rows covered: public %.4f private %.4f ; gap<24h: public %.4f private %.4f ; f5-up rows: public %.4f private %.4f"%(pr[r].sum()/pr.sum(),(~pr[r]).sum()/(~pr).sum(),pr[r[gap<24]].sum()/pr.sum(),(~pr[r[gap<24]]).sum()/(~pr).sum(),pr[r[d>0]].sum()/pr.sum(),(~pr[r[d>0]]).sum()/(~pr).sum()))
gv=later_feats(D["val"],D["val"],"v"); rv=gv["row"].to_numpy(); dv=(gv["f5_next"]-gv["f5"]).to_numpy()
print("val (within-val) rows covered %.4f ; f5-up rows %.4f"%(len(rv)/len(y),(dv>0).sum()/len(y)))
