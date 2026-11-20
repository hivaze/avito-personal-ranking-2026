# Forensics probe 7: re-reads every stored feature column that has a val file (248 of them) as a residual
# on the 8-student pool, through the nested harness of fx.py, 12 processes in parallel, and sorts the
# columns by z. The best read, +0.00044 +- 0.00020, is about what the largest of 248 null reads gives by
# chance (lab/reports/TOP1_GAP_FORENSICS_1114.md, row 7 of the verdict table).
# Run on CPU from lab/nn_x/w12/forensics (it imports fx.py); the command line is not on record.
# Produced p05_sweep.tsv (in ../results/forensics_248_column_sweep.tsv).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/forensics/p05_sweep.py
#
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
