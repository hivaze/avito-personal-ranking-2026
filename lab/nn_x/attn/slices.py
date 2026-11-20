"""Per-slice paired deltas on val (treat vs base, pooled over seeds): by history length H and feed size M.
  python slices.py attn_gate [--base_prefix ...]"""
import argparse, sys
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
p = argparse.ArgumentParser(); p.add_argument("treat"); p.add_argument("--seeds", default="23,24")
p.add_argument("--base_prefix", default="<repo>/preds_x/optim/nn_base_s{seed}")
p.add_argument("--treat_prefix", default="<repo>/preds_x/attn/nn_{t}_s{seed}")
a = p.parse_args()
FL = "<repo>/data/flat"
f = pl.read_parquet(f"{FL}/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64); H = f["H"].to_numpy()
u0 = f["user_id"].to_numpy() == 0
y = np.load(f"{FL}/val/cand_target.npy").astype(np.float64)
ok = np.add.reduceat(y, off) > 0
ds = []
for s in a.seeds.split(","):
    b = ndcg_per_feed(np.load(a.base_prefix.format(seed=s) + "_trainonly_val.npy"), y, off, M)
    t = ndcg_per_feed(np.load(a.treat_prefix.format(t=a.treat, seed=s) + "_trainonly_val.npy"), y, off, M)
    ds.append(t - b)
d = np.mean(ds, 0)
sl = {"all": np.ones_like(ok), "H=0": H == 0, "H 1-16": (H >= 1) & (H <= 16), "H 17-128": (H > 16) & (H <= 128),
      "H>128": H > 128, "user0": u0, "M<=32": M <= 32, "M 33-96": (M > 32) & (M <= 96), "M>96": M > 96}
for k, m in sl.items():
    m = m.astype(bool)
    print(f"{k:10s} n={m.sum():6d}  d={d[m].mean():+.5f}  se={d[m].std(ddof=1)/np.sqrt(m.sum()):.5f}")
