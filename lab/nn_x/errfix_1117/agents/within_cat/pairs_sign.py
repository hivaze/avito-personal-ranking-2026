"""Mutual near-duplicate pairs (equal cand_maxcos within a feed, k=2) with exactly one positive: sign tests."""
from wc import *
df = load(); d = features(df)
d = d.filter(pl.col("n_cand_dup95") >= 1).with_columns(pl.len().over(["fidx", "cand_maxcos"]).alias("k"), pl.col("label").sum().over(["fidx", "cand_maxcos"]).alias("gp"))
d = d.filter((pl.col("k") == 2) & (pl.col("gp") == 1)).sort(["fidx", "cand_maxcos", "label"], descending=[False, False, True])
P = d[0::2]; N = d[1::2]
assert (P["label"] == 1).all() and (N["label"] == 0).all()
cm = P["cand_maxcos"].to_numpy()
cols = ["score", "s_ft", "s_to", "f1", "f4", "f5", "ucos", "hmax", "h1", "h5", "h10", "lhidx", "xp_imp_c50", "lpop", "lpop_et2", "lpop_et4", "pre2_fe_z", "pre2_new_z", "lcv", "f2", "n_hdup95", "ldup", "rank"]
for lo, hi in [(0.95, 0.98), (0.98, 0.995), (0.995, 1.1)]:
    m = (cm >= lo) & (cm < hi)
    print(f"--- cand_maxcos in [{lo},{hi}): pairs {m.sum()}")
    for c in cols:
        a = P[c].to_numpy()[m].astype(float); b = N[c].to_numpy()[m].astype(float)
        ok = np.isfinite(a) & np.isfinite(b) & (a != b)
        w = (a[ok] > b[ok]).mean() if ok.sum() else np.nan
        z = (w - 0.5) / np.sqrt(0.25 / max(ok.sum(), 1))
        print("  %-11s P(pos > neg)=%.3f  n=%5d  z=%+.2f  mean diff %+.4f" % (c, w, ok.sum(), z, np.nanmean(a - b)))
