"""Segment x category offsets fitted by conditional logit with L2, cross-fitted on NDCG."""
import fx, kit, numpy as np, polars as pl, clogit, sys
from scipy import sparse
df = kit.load()
c = fx.Ctx(df)
s = c.base; y = c.y; feed = c.feed
cat = df["cat"].to_numpy().astype(int)
F = pl.read_parquet("feeds.parquet")
row = lambda a: np.asarray(a)[feed]
hour = row(F["hour"].to_numpy()); Mf = row(F["M"].to_numpy()); Hf = row(F["H"].to_numpy())
segs = {
    "cat": np.zeros(len(y), int),
    "hour4xcat": np.digitize(hour, [6, 12, 18]),
    "M3xcat": np.digitize(Mf, [30, 100]),
    "H3xcat": np.digitize(Hf, [50, 200]),
    "guestxcat": row(F["guest"].to_numpy()).astype(int),
    "wdxcat": (row(F["weekday"].to_numpy()) == 1).astype(int),
    "ntypexcat": np.select([row(F["newest_type"].to_numpy()) == 1, row(F["newest_type"].to_numpy()) == 10], [0, 1], 2),
    "nsrcxcat": np.select([row(F["newest_src"].to_numpy()) == 1, row(F["newest_src"].to_numpy()) == 2], [0, 1], 2),
}
l2s = [float(x) for x in sys.argv[1:]] or [3.0, 10.0, 30.0]
for nm, sg in segs.items():
    key = sg * 64 + cat
    uk, ki = np.unique(key, return_inverse=True)
    X = sparse.csr_matrix((np.ones(len(y)), (np.arange(len(y)), ki)), shape=(len(y), len(uk)))
    for l2 in l2s:
        oof_new = np.zeros(len(y))
        for k in range(5):
            tr = c.fold_row != k
            a, w, _ = clogit.fit(s, X, y, feed, c.nf, rowmask=tr, l2=l2, a0=1.0)
            te = ~tr
            oof_new[te] = s[te] + (X[te] @ w) / a
        d = c.ndcg(oof_new) - c.base_nd
        se = d.std(ddof=1) / np.sqrt(c.nf)
        print(f"{nm:10s} l2={l2:5.1f} groups={len(uk)} CF {d.mean()*1e4:+.2f}e-4 se {se*1e4:.2f} z {d.mean()/se:+.2f}", flush=True)
