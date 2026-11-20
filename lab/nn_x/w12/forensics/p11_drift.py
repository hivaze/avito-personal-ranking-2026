from fx import *
import json
RC="<repo>/data/competition/"
def table(s):
    sh=json.load(open(RC+f"{s}/item_embeddings/data.meta"))["shape"]; return np.memmap(RC+f"{s}/item_embeddings/data.memmap",dtype=np.int8,mode="r",shape=tuple(sh))
f,M,off,ft,to=load_w8("val"); y=np.load(FLAT/"val/cand_target.npy"); uid=f["user_id"].to_numpy(); n=len(M); fid=np.repeat(np.arange(n),M)
base=0.5*zfeed(ft,off,M)+0.5*zfeed(to,off,M)
vi=np.load(FLAT/"val/cand_item_id.npy"); ve=np.load(FLAT/"val/cand_emb_idx.npy")
ti=np.load(FLAT/"train/cand_item_id.npy"); te=np.load(FLAT/"train/cand_emb_idx.npy")
# last occurrence mapping item->train emb idx (table is per split so any occurrence is the same)
o=np.argsort(ti,kind="stable"); tis=ti[o]; pos=np.searchsorted(tis,vi); pos[pos>=len(tis)]=0
hit=(tis[pos]==vi); tidx=np.where(hit,te[o[pos]],-1); ok=(tidx>=0)&(ve>=0)
print("val cand rows with a train embedding: %.4f"%ok.mean())
Tv=table("val"); Tt=table("train")
r=np.nonzero(ok)[0]
ev=np.asarray(Tv[ve[r]],np.float32); et=np.asarray(Tt[np.sort(tidx[r])],np.float32)  # sorted read
inv=np.argsort(np.argsort(tidx[r])); et=et[inv]
d=ev-et; dn=np.linalg.norm(d,axis=1); print("drift: share exactly equal %.4f ; |d| quantiles"%(dn==0).mean(),np.quantile(dn,[.5,.9,.99]).round(2),"norm ev median %.1f"%np.median(np.linalg.norm(ev,axis=1)))
ue=np.load(FLAT/"val/user_emb.npy").astype(np.float32); u=ue[fid[r]]
ud=(u*d).sum(1)/(np.linalg.norm(u,axis=1)+1e-9)
cs=lambda a,b:(a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-9)
dcos=cs(u,ev)-cs(u,et)
print("target rate by drift: none %.4f (n %d) | drifted %.4f (n %d)"%(y[r[dn==0]].mean(),(dn==0).sum(),y[r[dn>0]].mean(),(dn>0).sum()))
m=dn>0; q=np.quantile(dcos[m],[.2,.4,.6,.8]); b=np.searchsorted(q,dcos[m]); print("drifted rows: target rate by quintile of cos(u,e_val)-cos(u,e_train):",[round(float(y[r[m]][b==k].mean()),4) for k in range(5)])
for nm,x_ in (("|drift|",dn),("u.d (directional)",ud),("dcos",dcos),("drifted flag",(dn>0).astype(float))):
    x=np.zeros(len(y)); x[r]=x_; z=zfeed(x,off,M)
    dl,se,wc,(wb,db)=nested(base,z,y,off,M,uid)
    print("  %-20s nested %+.5f +- %.5f w=%s in-sample w=%s %+.5f"%(nm,dl,se,wc,wb,db))
