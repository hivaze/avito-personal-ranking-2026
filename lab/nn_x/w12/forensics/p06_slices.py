from fx import *
f,M,off,ft,to=load_w8("val"); y=np.load(FLAT/"val/cand_target.npy"); uid=f["user_id"].to_numpy(); ts=f["timestamp"].to_numpy(); H=f["H"].to_numpy()
n=len(M); fid=np.repeat(np.arange(n),M)
base=0.5*zfeed(ft,off,M)+0.5*zfeed(to,off,M)
p=ndcg_per_feed(base,y,off,M); L=1-p; print("W8 val",p.mean(),"total loss",L.mean())
ft_=pl.read_parquet(FLAT/"train/feeds.parquet",columns=["user_id"]); tru=np.unique(ft_["user_id"].to_numpy())
seen=np.isin(uid,tru)&(uid!=0)
cat=np.load(FLAT/"val/cand_category_id.npy"); item=np.load(FLAT/"val/cand_item_id.npy")
luo=np.load(FEATS/"val/it_hist_all_luo.npy"); f4=np.load(FLAT/"val/cand_f4.npy"); f5=np.load(FLAT/"val/cand_f5.npy")
ctr=np.load(FEATS/"val/it_cand_cnt_train.npy")
npos=np.bincount(fid,y,n).astype(int)
ncat=pl.DataFrame({"f":fid,"c":cat}).group_by("f").agg(pl.col("c").n_unique()).sort("f")["c"].to_numpy()
cold=np.bincount(fid,(ctr==0),n)/M
msk=(ts+3*3600)%86400//3600; day=(ts+3*3600)//86400
# positive-level attributes (first positive = best-ranked positive)
r=np.empty(len(base)); o=np.lexsort((y,-base,fid)); r[o]=np.arange(len(base))-np.repeat(off,M)
big=np.where(y>0,r,1e9); bestr=np.minimum.reduceat(big,off)
def posmean(x): return np.bincount(fid,x*(y>0),n)/npos
pos_cold=posmean((ctr==0).astype(float)); pos_luo0=posmean((luo==0).astype(float))
f4r=rankz(f4,off,M); pos_f4z=posmean(f4r)
pos_cat=np.zeros(n,int); idx=np.nonzero(y>0)[0]; pos_cat[fid[idx][::-1]]=cat[idx][::-1]
# share of feed in the positive's category
catshare=pl.DataFrame({"f":fid,"c":cat}).with_columns((pl.len().over(["f","c"])/pl.len().over("f")).alias("s"))["s"].to_numpy()
pos_cs=posmean(catshare)
def tab(name,g,labels=None):
    print(f"\n== {name}")
    print("%-18s %7s %7s %8s %9s %8s"%("slice","feeds","share","NDCG","loss/N","loss%"))
    for v in np.unique(g):
        m=g==v; lab=labels[v] if labels else str(v)
        print("%-18s %7d %7.3f %8.4f %9.5f %7.1f%%  se %.4f"%(lab,m.sum(),m.mean(),p[m].mean(),L[m].sum()/n,100*L[m].sum()/L.sum(),p[m].std()/np.sqrt(m.sum())))
def q(x,edges): return np.searchsorted(edges,x,side="right")
tab("M",q(M,[10,20,30,50,100,200]),["<10","10-19","20-29","30-49","50-99","100-199","200+"])
tab("H",q(H,[1,16,64,128,256]),["0","1-15","16-63","64-127","128-255","256"])
tab("user",np.where(uid==0,0,np.where(seen,1,2)),["guest(u0)","seen in train","new user"])
tab("npos",np.minimum(npos,4))
tab("n categories",q(ncat,[3,6,10,15,20]))
tab("cold share of feed",q(cold,[0.1,0.3,0.5,0.7]))
tab("best positive is cold (not a train cand)",(pos_cold>0.5).astype(int))
tab("positive LUO-pop==0",(pos_luo0>0.5).astype(int))
tab("positive age z (f4 rank-z)",q(pos_f4z,[-1,-0.3,0.3,1]))
tab("positive's category share in feed",q(pos_cs,[0.05,0.1,0.2,0.4,0.7]))
tab("day",day-day.min()); tab("hour MSK /4",msk//4)
tab("best-positive rank under W8",q(bestr,[1,3,8,20,50]),["0","1-2","3-7","8-19","20-49","50+"])
# two-way: M x pos cat share
print("\n== positive category: top 12 by loss")
rows=[]
for c in np.unique(pos_cat):
    m=pos_cat==c; rows.append((L[m].sum()/n,c,m.sum(),p[m].mean()))
for l,c,k,pm in sorted(rows)[::-1][:12]: print("cat %2d feeds %5d NDCG %.4f loss/N %.5f"%(c,k,pm,l))
# relative-to-M expectation: random NDCG per feed analytic approx via simulation
rng=np.random.default_rng(0); pr=np.mean([ndcg_per_feed(rng.random(len(base)),y,off,M) for _ in range(5)],0)
print("\nrandom NDCG",pr.mean()," model lift over random by M bucket:")
g=q(M,[10,20,30,50,100,200])
for v in np.unique(g):
    m=g==v; print(v,"rand %.4f model %.4f"%(pr[m].mean(),p[m].mean()))
np.save("val_w8_ndcg.npy",p)
