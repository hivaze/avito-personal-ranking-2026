"""Out-of-fold listwise log-likelihood vs NDCG for full-list clogit corrections."""
from wc import *
df = load(); d = features(df)
d = d.with_columns((pl.col("score") - pl.col("score").mean().over(["fidx", "cat"])).alias("s_dm"))
d = d.with_columns((pl.col("s_dm") * pl.col("lcatn")).alias("sdm_lcn"))
off, sz = kit.feed_layout(d)
y = d["label"].to_numpy(); s = d["score"].to_numpy().astype(np.float64)
fo = kit.folds(d); ff = fo[off]
seg = np.repeat(np.arange(len(sz)), sz)
nb = kit.ndcg_per_feed(s, y, off, sz)
def feed_ll(z):
    zmax = np.maximum.reduceat(z, off); e = np.exp(z - zmax[seg]); S = np.add.reduceat(e, off)
    npos = np.add.reduceat(y.astype(float), off)
    return np.add.reduceat(y * z, off) - npos * (np.log(S) + zmax)
for nm, cols in [("lcn", ["lcatn"]), ("sdm+sdm_lcn", ["s_dm", "sdm_lcn"]), ("cat_hist_share", ["cat_hist_share"]), ("ldup", ["ldup"])]:
    X = np.nan_to_num(d.select(cols).to_numpy().astype(np.float64))
    llb = np.zeros(len(sz)); lln = np.zeros(len(sz)); new = s.copy()
    for k in range(5):
        tr = fo != k; te = ~tr
        dt = d.filter(pl.Series(tr)); o2, s2 = kit.feed_layout(dt)
        a0, _, _, _ = clogit(np.zeros((tr.sum(), 0)), s[tr], y[tr].astype(float), o2, s2)
        a, b, se, L = clogit(X[tr], s[tr], y[tr].astype(float), o2, s2, l2=0)
        z0 = a0 * s; z1 = a * s + X @ b
        fte = ff == k
        llb[fte] = feed_ll(z0)[fte]; lln[fte] = feed_ll(z1)[fte]
        new[te] = (z1 / a)[te]
    dl = lln - llb
    nn = kit.ndcg_per_feed(new, y, off, sz); dd = nn - nb
    print("%-15s oof dLL/feed %.5f (z %.2f) total %.1f | ndcg %.5f (z %.2f)" % (nm, dl.mean(), dl.mean() / (dl.std(ddof=1) / np.sqrt(len(dl))), dl.sum(), dd.mean(), dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))))
