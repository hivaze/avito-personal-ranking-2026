"""Category-specific slopes: new = score + sum_c [cat==c] * b_c * zfeat, b fitted by clogit with L2 on 4 folds, NDCG on the 5th."""
import fx, kit, numpy as np, polars as pl, clogit, sys
from scipy import sparse
df = kit.load()
c = fx.Ctx(df)
s = c.base; y = c.y; feed = c.feed
cat = df["cat"].to_numpy().astype(int)
uk, ki = np.unique(cat, return_inverse=True)
def nz(x, fill):
    x = np.asarray(x, float); return np.where(np.isfinite(x), x, fill)
feats = {
    "f1": df["f1"].to_numpy().astype(float), "f2": df["f2"].to_numpy().astype(float),
    "f2_home": df["f2_home"].to_numpy().astype(float), "f4": df["f4"].to_numpy().astype(float),
    "f5": df["f5"].to_numpy().astype(float), "ucos": nz(df["ucos"].to_numpy(), 0.48), "h1": nz(df["h1"].to_numpy(), 0.45),
    "hmax": nz(df["hmax"].to_numpy(), 0.8), "xp": df["xp_imp_c50"].to_numpy().astype(float),
    "log_ith": np.log1p(df["it_hist_all_luo"].to_numpy().astype(float)), "pre2_fe_z": df["pre2_fe_z"].to_numpy().astype(float),
}
l2s = [float(x) for x in sys.argv[1:]] or [10.0, 100.0]
for nm, x in feats.items():
    xs = fx.zf(x, c.off, c.sz, feed)
    X = sparse.csr_matrix((xs, (np.arange(len(y)), ki)), shape=(len(y), len(uk)))
    for l2 in l2s:
        oof_new = np.zeros(len(y))
        for k in range(5):
            tr = c.fold_row != k
            a, w, _ = clogit.fit(s, X, y, feed, c.nf, rowmask=tr, l2=l2, a0=1.0)
            te = ~tr
            oof_new[te] = s[te] + (X[te] @ w) / a
        d = c.ndcg(oof_new) - c.base_nd
        se = d.std(ddof=1) / np.sqrt(c.nf)
        a, w, _ = clogit.fit(s, X, y, feed, c.nf, l2=l2, a0=1.0)
        top = np.argsort(-np.abs(w))[:5]
        print(f"{nm:9s} l2={l2:6.1f} CF {d.mean()*1e4:+.2f}e-4 se {se*1e4:.2f} z {d.mean()/se:+.2f}  top slopes " +
              " ".join(f"c{uk[t]}:{w[t]/a:+.3f}" for t in top), flush=True)
