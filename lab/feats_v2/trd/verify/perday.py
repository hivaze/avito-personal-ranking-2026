import sys, numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
from pathlib import Path
F=Path("<repo>/data/feats"); FL=Path("<repo>/data/flat")
names=sorted(p.stem for p in (F/"val").glob("trd_*.npy"))
f=pl.read_parquet(FL/"val/feeds.parquet"); M=f["M"].to_numpy(); off=f["cand_off"].to_numpy(); ts=f["timestamp"].to_numpy()
y=np.load(FL/"val/cand_target.npy"); day=(ts>=1790197200)
rng=np.random.default_rng(0)
ytr=np.load(FL/"train/cand_target.npy", mmap_mode="r")
idx=np.sort(rng.choice(len(ytr),3_000_000,replace=False))
for nm in names:
    x=np.load(F/"val"/f"{nm}.npy").astype(np.float64)
    x=np.where(np.isfinite(x),x,np.nanmin(x)-1)
    r=[]
    for sgn in (1,-1):
        pf=np.mean([ndcg_per_feed(sgn*x+rng.random(len(x))*1e-6*(np.nanstd(x)+1e-9),y,off,M) for _ in range(3)],axis=0)
        r.append((pf[~day].mean(),pf[day].mean(),pf.mean()))
    nzv=(x>0).mean() if not nm.startswith(("trd_dsr","trd_max")) else np.nan
    t=np.load(F/"test"/f"{nm}.npy"); tr=np.load(F/"train"/f"{nm}.npy",mmap_mode="r")[idx]
    nzt=(t>0).mean() if not nm.startswith(("trd_dsr","trd_max")) else np.nan
    nztr=(tr>0).mean() if not nm.startswith(("trd_dsr","trd_max")) else np.nan
    print(f"{nm:18s} asc Tue {r[0][0]:.4f} Thu {r[0][1]:.4f} all {r[0][2]:.4f} | desc all {r[1][2]:.4f} | nonzero val {nzv:.3f} test {nzt:.3f} train {nztr:.3f}", flush=True)
