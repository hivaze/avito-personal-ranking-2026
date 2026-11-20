"""Cross-fitted branch weight per segment: for each segmentation, choose w in each segment on 4 folds
(by mean delta), evaluate on the 5th. Also the 1-d smooth version: w = 0.5 + k * (z(segment var))."""
import hk, numpy as np, polars as pl

c = hk.Ctx(); df = c.df
a = c.col("s_ft").astype(float); b = c.col("s_to").astype(float)
ws = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ND = {w: c.nd(w * a + (1 - w) * b) - c.base for w in ws}
fe = df.group_by("fidx", maintain_order=True).agg(pl.all().first())
segs = {
    "M": np.digitize(fe["M"].to_numpy(), [30, 60, 100, 200]),
    "H": np.digitize(fe["H"].to_numpy(), [50, 150, 255]),
    "newest_type": np.where(np.isin(fe["newest_type"].to_numpy(), [1, 2, 3, 4]), fe["newest_type"].to_numpy(), 0),
    "nhc32": np.digitize(fe["n_hist_cats32"].to_numpy(), [2, 3, 5, 9]),
    "nfc": np.digitize(fe["n_feed_cats"].to_numpy(), [4, 7, 11]),
    "newest_src": np.where(np.isin(fe["newest_src"].to_numpy(), [0, 1, 2]), fe["newest_src"].to_numpy(), 3),
    "guest": fe["guest"].to_numpy().astype(int),
}
ff = c.fold_feed
for name, s in segs.items():
    oof = np.zeros(len(ff)); ch = []
    for f in range(5):
        tr = ff != f; te = ff == f
        for v in np.unique(s):
            m = s == v
            best = max(ws, key=lambda w: ND[w][tr & m].mean() if (tr & m).any() else 0)
            ch.append((v, best))
            oof[te & m] = ND[best][te & m]
    se = oof.std(ddof=1) / np.sqrt(len(oof))
    insample = sum(max(ND[w][s == v].sum() for w in ws) for v in np.unique(s)) / len(s)
    print(f"{name:12s} crossfit {oof.mean():+.5f} se {se:.5f} z {oof.mean()/se:+.2f}   in-sample best {insample:+.5f}")
