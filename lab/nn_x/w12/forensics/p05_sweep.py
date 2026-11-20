from fx import *
from concurrent.futures import ProcessPoolExecutor
os.environ["OMP_NUM_THREADS"]="1"
f,M,off,ft,to=load_w8("val"); y=np.load(FLAT/"val/cand_target.npy"); uid=f["user_id"].to_numpy()
base=0.5*zfeed(ft,off,M)+0.5*zfeed(to,off,M)
names=sorted(p.stem for p in (FEATS/"val").glob("*.npy"))
tn=set(p.stem for p in (FEATS/"test").glob("*.npy"))
def run(n):
    x=np.load(FEATS/"val"/f"{n}.npy").astype(np.float64)
    if x.shape[0]!=len(base): return None
    x=np.nan_to_num(x,nan=0.0,posinf=0,neginf=0)
    s=zfeed(x,off,M)
    if np.abs(s).max()==0: return None
    d,se,wc,(wb,db)=nested(base,s,y,off,M,uid)
    return (n,d,se,wc,wb,db,n in tn)
if __name__=="__main__":
    with ProcessPoolExecutor(12) as ex: res=[r for r in ex.map(run,names) if r]
    res.sort(key=lambda r:-r[1]/max(r[2],1e-9))
    with open("p05_sweep.tsv","w") as fo:
        for r in res: fo.write("%s\t%+.5f\t%.5f\t%s\t%s\t%+.5f\t%s\n"%r)
    print("n feats",len(res)); 
    for r in res[:15]: print("%-28s nested %+.5f +- %.5f w=%s in-sample w=%s %+.5f test=%s"%r)
    z=np.array([r[1]/max(r[2],1e-9) for r in res]); print("z quantiles",np.quantile(z,[0,.1,.5,.9,1]).round(2), "n nonzero",(np.array([r[1] for r in res])!=0).sum())
