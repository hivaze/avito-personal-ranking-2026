# Shared harness of the raw-data forensics (2026-11-14). nested() adds a candidate signal s to the pool
# score as z(pool) + w * z(s), chooses w on one random half of the val users, reads the NDCG change on the
# other half, and does the same in the other direction. Every probe p01..p12 reports this number with its
# paired per-feed standard error over the 14,423 val feeds.
# Imported by the probe scripts of lab/nn_x/w12/forensics; not run on its own.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/forensics/fx.py
#
import os; os.environ.setdefault("OMP_NUM_THREADS","12"); os.environ.setdefault("POLARS_MAX_THREADS","12"); os.environ.setdefault("NUMBA_NUM_THREADS","12")
import sys; sys.path.insert(0,"<repo>/src")
import numpy as np, polars as pl
from common import feed_meta, FLAT, FEATS
from metrics import ndcg_per_feed
R="<repo>/"
PUB_TS=1790497920
def zfeed(x,off,M):
    x=np.asarray(x,np.float64); fid=np.repeat(np.arange(len(M)),M)
    mu=np.bincount(fid,x)/M; sd=np.sqrt(np.maximum(np.bincount(fid,x*x)/M-mu*mu,0))+1e-9
    return (x-mu[fid])/sd[fid]
def rankz(x,off,M):
    # per-feed average-rank then z
    x=np.asarray(x,np.float64); fid=np.repeat(np.arange(len(M)),M)
    o=np.lexsort((x,fid)); r=np.empty(len(x)); r[o]=np.arange(len(x))-np.repeat(off,M)
    # average ties
    df=pl.DataFrame({"f":fid,"x":x,"r":r}).with_columns(pl.col("r").mean().over(["f","x"]))
    return zfeed(df["r"].to_numpy(),off,M)
def load_w8(split):
    f,M,off=feed_meta(split)
    from stack_val import feed_rank_z
    ft=np.load(R+f"preds/ens_W8kd_ft_{split}.npy"); to=np.load(R+f"preds/ens_W8kd_to_{split}.npy")
    return f,M,off,ft,to
def nested(base,s,y,off,M,uid,ws=(-0.3,-0.2,-0.1,-0.05,-0.02,0,0.02,0.05,0.1,0.2,0.3,0.5),seed=0,mask=None):
    """base, s already per-feed z. returns delta mean, se, chosen ws. user-half split both directions."""
    fid=np.repeat(np.arange(len(M)),M)
    p0=ndcg_per_feed(base,y,off,M)
    P=np.stack([ndcg_per_feed(base+w*s,y,off,M) for w in ws])
    rng=np.random.default_rng(seed); uu=np.unique(uid); h=rng.random(len(uu))<0.5
    half=h[np.searchsorted(uu,uid)]
    # user 0 feeds: assign by feed hash
    z0=uid==0; half[z0]=rng.random(z0.sum())<0.5
    d=np.zeros(len(M)); wc=[]
    for a in (True,False):
        tr=half==a; te=~tr
        k=int(np.argmax(P[:,tr].mean(1))); wc.append(ws[k]); d[te]=P[k,te]-p0[te]
    best=int(np.argmax(P.mean(1)))
    return d.mean(), d.std()/np.sqrt(len(d)), wc, (ws[best], (P[best]-p0).mean())
