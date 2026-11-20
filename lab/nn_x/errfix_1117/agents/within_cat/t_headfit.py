"""Cross-fitted head re-ranker: clogit on the top-K of train folds with a few a-priori features, applied to the held-out fold's top-K."""
from wc import *
df = load(); d = features(df)
d = rel(d, ["f1", "ldup", "pre2_fe_z", "ucos", "h5", "xp_imp_c50"])
d = d.with_columns((pl.col("score") - pl.col("score").mean().over(["fidx", "cat"])).alias("s_dm"))
off, sz = kit.feed_layout(d)
y = d["label"].to_numpy(); s = d["score"].to_numpy().astype(np.float64); rk = d["rank"].to_numpy()
fo = kit.folds(d); ff = fo[off]
nb = kit.ndcg_per_feed(s, y, off, sz)
sets = {
    "lcn": ["lcatn"],
    "lcn+share": ["lcatn", "cat_hist_share"],
    "dup": ["ldup", "ldup_pr"],
    "cat+dup": ["lcatn", "cat_hist_share", "ldup"],
    "rich": ["lcatn", "cat_hist_share", "ldup", "f1_pr", "pre2_fe_z", "lcv", "f2", "cand_maxcos"],
}
sdm = d["s_dm"].to_numpy().astype(np.float64); lcn = d["lcatn"].to_numpy().astype(np.float64)
for K in [10, 20, 40]:
    for nm, cols in sets.items():
        X = d.select(cols).to_numpy().astype(np.float64)
        X = np.nan_to_num(X)
        mu, sd_ = X.mean(0), X.std(0); X = (X - mu) / sd_
        new = s.copy()
        head = rk <= K
        for k in range(5):
            tr = (fo != k) & head
            dt = d.filter(pl.Series(tr))
            o2, s2 = kit.feed_layout(dt)
            yt = y[tr].astype(np.float64)
            keep = np.add.reduceat(yt, o2) > 0
            # clogit ignores feeds with no positive automatically (npos=0)
            a, b, se, L = clogit(X[tr], s[tr], yt, o2, s2, l2=1e-4)
            te = (fo == k) & head
            new[te] = s[te] + (X[te] @ b) / a
        nn = kit.ndcg_per_feed(new, y, off, sz); dd = nn - nb
        print("K=%d %-10s oof %.5f se %.5f z %.2f" % (K, nm, dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd)), dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))), flush=True)
