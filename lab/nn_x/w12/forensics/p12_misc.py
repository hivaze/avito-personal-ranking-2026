from fx import *
f,M,off=feed_meta("val"); uid=f["user_id"].to_numpy(); p=np.load("val_w8_ndcg.npy"); n=len(M); H=f["H"].to_numpy()
tr=pl.read_parquet(FLAT/"train/feeds.parquet",columns=["user_id","timestamp"])
cnt=tr.group_by("user_id").agg(pl.len().alias("k")); d=dict(zip(cnt["user_id"].to_list(),cnt["k"].to_list()))
k=np.array([d.get(int(u),0) if u!=0 else 0 for u in uid])
y=np.load(FLAT/"val/cand_target.npy"); fid=np.repeat(np.arange(n),M)
rng=np.random.default_rng(0); pr=np.mean([ndcg_per_feed(rng.random(len(y)),y,off,M) for _ in range(20)],0)
# M-adjusted: regress p on random-expected NDCG (captures M and npos)
import numpy.linalg as la
X=np.c_[np.ones(n),pr,np.log(M)]; b=la.lstsq(X,p,rcond=None)[0]; res=p-X@b
for nm,m in (("guest",uid==0),("new user",(uid!=0)&(k==0)),("seen 1 train feed",k==1),("seen 2-3",(k>=2)&(k<=3)),("seen 4+",k>=4)):
    print("%-18s n %5d  NDCG %.4f  random %.4f  adj residual %+.4f +- %.4f  mean M %.0f H %.0f"%(nm,m.sum(),p[m].mean(),pr[m].mean(),res[m].mean(),res[m].std()/np.sqrt(m.sum()),M[m].mean(),H[m].mean()))
# test composition of same groups
ft,Mt,_=feed_meta("test"); ut=ft["user_id"].to_numpy(); pub=ft["timestamp"].to_numpy()<PUB_TS
kt=np.array([d.get(int(u),0) if u!=0 else 0 for u in ut])
for nm,m in (("guest",ut==0),("new user",(ut!=0)&(kt==0)),("seen 1",kt==1),("seen 2-3",(kt>=2)&(kt<=3)),("seen 4+",kt>=4)):
    print("test %-10s public %.3f private %.3f"%(nm,m[pub].mean(),m[~pub].mean()))
