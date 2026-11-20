"""Segment samples for the preprocessing review (featrev_prep): feed indices + candidate rows per segment.
Segments: trR = train random-like (p_random > 0.9, M >= 50), trP = train production (p_random < 0.05 or M < 50),
val = all val feeds, pub = test ts < 1790497920, priv = test ts >= 1790497920.
Output: <scratch>/out/segments.npz (feed idx per segment; train samples of N feeds, seed 7)."""
import numpy as np, polars as pl, sys
OUT = sys.argv[1]
ROOT = "<repo>"
rng = np.random.default_rng(7)
N = 6000
f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet")
M = f["M"].to_numpy()
pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"], nan=0.0)
pr[M < 50] = 0.0
trR = np.sort(rng.choice(np.where((pr > 0.9) & (M >= 50))[0], N, replace=False))
trP = np.sort(rng.choice(np.where(pr < 0.05)[0], N, replace=False))
v = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
t = pl.read_parquet(f"{ROOT}/data/flat/test/feeds.parquet")
ts = t["timestamp"].to_numpy()
pub = np.where(ts < 1790497920)[0]
priv = np.where(ts >= 1790497920)[0]
np.savez(f"{OUT}/segments.npz", trR=trR, trP=trP, val=np.arange(len(v)), pub=pub, priv=priv)
for k, a in dict(trR=trR, trP=trP, val=np.arange(len(v)), pub=pub, priv=priv).items():
    print(k, len(a))
