"""winner_model recheck, check 7: feed-size distribution that train.py --aug_all --aug_cand A produces (cd_arms.py:73-80:
per feed q ~ U(A,1), every positive kept, each negative kept w.p. q) versus val / test. CPU, 1 thread."""
import numpy as np, polars as pl
R = "<repo>/data/flat/"
def meta(sp):
    f = pl.read_parquet(R + sp + "/feeds.parquet"); return f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64)
M, off = meta("train")
y = np.load(R + "train/cand_target.npy", mmap_mode="r")
npos = np.add.reduceat(np.asarray(y, dtype=np.int64), off)
rng = np.random.default_rng(0)
def desc(name, m):
    q = np.percentile(m, [10, 25, 50, 75, 90])
    print(f"{name:34s} mean {m.mean():6.1f} | M<20 {np.mean(m < 20):.3f} | M<50 {np.mean(m < 50):.3f} | M>=100 {np.mean(m >= 100):.3f} | q10/25/50/75/90 " + "/".join(f"{x:.0f}" for x in q))
desc("train, as stored", M)
for a in (0.8, 0.5, 0.2):
    q = a + (1 - a) * rng.random(len(M))
    desc(f"train, --aug_cand {a}", npos + rng.binomial(M - npos, q))
for sp in ("val", "test"):
    desc(sp, meta(sp)[0])
