"""Power control at the decision threshold: planted signals worth about +0.0010 and +0.0005 (oracle, all-feed) on the 30% covered feeds."""
from hx import *
D = load_val(); y, M, off, fid = D["y"], D["M"], D["off"], D["fid"]; fh = np.load(OUT / "bank_val.npz")["feed_has"]
ctx = ctx_cols(D); Xc = np.stack([ctx[k] for k in ctx], 1).astype(np.float32); nc = list(ctx)
cov = fh[fid]; sgn = np.where(np.repeat(M, M) >= np.median(M[fh]), 1.0, -1.0); p0 = ndcg_per_feed(D["base"], y, off, M)
for dl in (0.175, 0.125):
    for rep in (0, 1):
        eps = np.random.default_rng(1000 + rep).standard_normal(len(y))
        for name, s, sg in ((f"P1 additive delta {dl} rep {rep}", np.where(cov, eps + dl * y, 0.0), 1.0), (f"P2 sign-flip delta {dl} rep {rep}", np.where(cov, eps + dl * y * sgn, 0.0), sgn)):
            z = np.where(cov, zfeed(s, off, M), 0.0); d, wc = nested_lin(D, z)
            g = max((ndcg_per_feed(D["base"] + w * z * sg, y, off, M) - p0).mean() for w in (0.02, 0.05, 0.1, 0.2))
            X = np.hstack([Xc, s[:, None].astype(np.float32), z[:, None].astype(np.float32)])
            d1, det, _ = resid_cv(D, X, nc + ["s", "s_fz"], name, log=lambda *a: None)
            hf = user_half(D["uid"], 0)
            print(f"{name}: oracle in-sample {g:+.5f} | linear nested {fmt(d)} | LGB residual {fmt(d1)} (z {np.nanmean(d1) / se(d1):+.2f}); halves {fmt(d1[hf])} / {fmt(d1[~hf])}; rounds {[x['rounds'] for x in det]}", flush=True)
