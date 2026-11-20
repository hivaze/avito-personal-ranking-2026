from fx import *
s="val"; f,M,off=feed_meta(s); fid=np.repeat(np.arange(len(M)),M)
C=pl.DataFrame({"f":fid,"item":np.load(FLAT/f"{s}/cand_item_id.npy"),"c":np.load(FLAT/f"{s}/cand_category_id.npy").astype(np.int32),"f1":np.load(FLAT/f"{s}/cand_f1.npy"),"f3":np.load(FLAT/f"{s}/cand_f3.npy"),"y":np.load(FLAT/f"{s}/cand_target.npy")})
v,c=np.unique(C["f1"].to_numpy(),return_counts=True); o=np.argsort(-c)[:6]; print("f1 atoms",[(float(v[i]),int(c[i])) for i in o],"n unique",len(v))
v3,c3=np.unique(C["f3"].to_numpy(),return_counts=True); o=np.argsort(-c3)[:4]; print("f3 atoms",[(float(v3[i]),int(c3[i])) for i in o],"n unique",len(v3))
g=C.group_by(["f","c"]).agg(pl.len().alias("n"),pl.col("f1").n_unique().alias("k1"),pl.col("f3").n_unique().alias("k3"))
print("(feed,cat) groups",g.height,"mean n %.2f mean distinct f1 %.2f ; f3 const %.4f"%(g["n"].mean(),g["k1"].mean(),(g["k3"]==1).mean()))
gg=C.group_by(["f","c","f1"]).agg(pl.len().alias("n"))
print("(feed,cat,f1) groups",gg.height,"size dist",np.bincount(np.minimum(gg["n"].to_numpy(),10))[1:])
# does f1 value repeat across categories within a feed (i.e. is f1 a value not a key)?
x=C.group_by(["f","f1"]).agg(pl.col("c").n_unique().alias("kc"),pl.len().alias("n")).filter(pl.col("f1")>v[0]+1e-9)
print("non-floor (feed,f1) groups: share spanning >1 category %.4f (n %d)"%((x["kc"]>1).mean(),x.height))
# consistency across feeds: item pairs co-occurring in 2+ feeds, non-floor f1 in both
FL1=float(v[np.argmax(c)]); print("f1 floor atom",FL1)
E=C.filter(pl.col("f1")!=FL1).select("f","c","f1","item")
# items appearing in >=2 feeds
multi=E.group_by("item").agg(pl.len().alias("k")).filter(pl.col("k")>=2)["item"]
Em=E.filter(pl.col("item").is_in(multi.implode()))
P=Em.join(Em,on="f",suffix="_b").filter(pl.col("item")<pl.col("item_b")).with_columns((pl.col("f1")==pl.col("f1_b")).alias("same"),(pl.col("c")==pl.col("c_b")).alias("samec"))
pp=P.group_by(["item","item_b"]).agg(pl.len().alias("k"),pl.col("same").sum().alias("ns"),pl.col("samec").first()).filter(pl.col("k")>=2)
print("item pairs co-occurring in >=2 feeds (both non-floor):",pp.height,"same-cat",pp["samec"].sum())
q=pp.filter(pl.col("samec"))
print(" same-cat pairs: always same f1 %d, never %d, mixed %d"%((q["ns"]==q["k"]).sum(),(q["ns"]==0).sum(),((q["ns"]>0)&(q["ns"]<q["k"])).sum()))
