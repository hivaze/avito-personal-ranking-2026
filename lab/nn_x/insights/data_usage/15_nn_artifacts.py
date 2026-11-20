"""Ready-made inputs for the proposed NN arms (CPU, read-only on data; writes out/nn_artifacts/):
  train_rank_true.npy   int16 [246.7M]  within-feed rank of the shown position (0 = top), in flat train order
  train_rkend_true.npy  int16 [246.7M]  true rank from the end (0 = last shown item); NOT M-1-pos (gaps!)
  trunc_lift.json       log click lift of rkend 0/1/2 vs rkend>=3 by group (prod M<50, prod M>=50, random-like p>0.5 M>=50)
  theta_rand_rank.npy   (copy) examination lift by rank from random-like M>=50 feeds (last 3 ranks excluded)"""
import json, shutil
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path("<repo>"); D = ROOT / "nn_x/insights/data_usage/out"; A = D / "nn_artifacts"; A.mkdir(exist_ok=True)
t = pl.read_parquet(D / "train_feeds.parquet", columns=["M", "cls"])
M = t["M"].to_numpy().astype(np.int64); cls = t["cls"].to_numpy(); n = len(M)
fid = np.repeat(np.arange(n), M)
pos = np.load(ROOT / "data/flat/train/cand_pos.npy").astype(np.int64)
y = np.load(ROOT / "data/flat/train/cand_target.npy").astype(np.int8)
off = np.concatenate([[0], np.cumsum(M)[:-1]])
o = np.lexsort((pos, fid)); rank = np.empty(len(pos), np.int64); rank[o] = np.arange(len(pos)) - off[fid[o]]; del o
rk = M[fid] - 1 - rank
np.save(A / "train_rank_true.npy", rank.astype(np.int16)); np.save(A / "train_rkend_true.npy", rk.astype(np.int16))
grp = np.where(cls[fid] >= 1, 2, np.where(M[fid] < 50, 0, 1))
L = {}
for g, nm in enumerate(["prod_M<50", "prod_M>=50", "random_like"]):
    m = grp == g; base = y[m & (rk >= 3)].mean()
    L[nm] = {f"rkend{k}": float(np.log(y[m & (rk == k)].mean() / base)) for k in range(3)}
    L[nm]["base_rate_rkend>=3"] = float(base)
json.dump(L, open(A / "trunc_lift.json", "w"), indent=1)
shutil.copy(D / "theta_rand_rank.npy", A / "theta_rand_rank.npy")
print(json.dumps(L, indent=1))
