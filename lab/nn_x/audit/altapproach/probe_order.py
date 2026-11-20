"""Eye-fatigue check: does the STORED within-feed candidate order (or rank-from-end of the stored order) carry label
information on val/test-like data? (Val/test candidates were uniformly shuffled before display and the files were
shuffled by one common permutation; if the stored order were the display order, the terminal click would be visible.)"""
import os
for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","POLARS_MAX_THREADS","NUMBA_NUM_THREADS"]: os.environ[k]="12"
import sys, numpy as np, polars as pl
sys.path.insert(0,"<repo>/src")
from metrics import ndcg_per_feed
f=pl.read_parquet("<repo>/data/flat/val/feeds.parquet"); M=f["M"].to_numpy().astype(np.int64); off=f["cand_off"].to_numpy().astype(np.int64)
y=np.load("<repo>/data/flat/val/cand_target.npy").astype(np.float64); n=len(M); fid=np.repeat(np.arange(n),M)
pos=np.arange(len(y))-np.repeat(off,M); rkend=np.repeat(M,M)-1-pos
rng=np.random.default_rng(0); noise=1e-6*rng.random(len(y))
rnd=np.mean([ndcg_per_feed(rng.random(len(y)),y,off,M).mean() for _ in range(5)])
print(f"random {rnd:.4f}  stored-order asc {ndcg_per_feed(-pos+noise,y,off,M).mean():.4f}  desc {ndcg_per_feed(pos+noise,y,off,M).mean():.4f}  "
      f"last-first {ndcg_per_feed((rkend==0)*10.0-pos*1e-3+noise,y,off,M).mean():.4f}")
for name, r in [("stored pos", pos), ("stored rank-from-end", rkend)]:
    print(name, "target rate by bucket:", {b: round(float(y[(r>=lo)&(r<hi)].mean()),4) for b,(lo,hi) in {"0":(0,1),"1":(1,2),"2":(2,3),"3-9":(3,10),"10-49":(10,50),"50+":(50,10**6)}.items()}, "base", round(float(y.mean()),4))
# feed_id order and item_id order as scores (eda_history_leak 3.5 said random); emb_idx too
ci=np.load("<repo>/data/flat/val/cand_item_id.npy").astype(np.float64); ce=np.load("<repo>/data/flat/val/cand_emb_idx.npy").astype(np.float64)
print(f"item_id asc {ndcg_per_feed(-ci+noise,y,off,M).mean():.4f} desc {ndcg_per_feed(ci+noise,y,off,M).mean():.4f} | emb_idx asc {ndcg_per_feed(-ce+noise,y,off,M).mean():.4f} desc {ndcg_per_feed(ce+noise,y,off,M).mean():.4f}")
