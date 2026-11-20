"""Power / positive control of the residual harness: PLANTED signals on val (30% covered feeds = the real coverage mask of signal (a)).
P1 additive: s = eps + delta*y on covered feeds. P2 interaction-only: s = eps + delta*y*sign(M - median M) (a linear feed-agnostic weight sees ~0).
delta is set so that the planted information is worth about +0.001 all-feed for an oracle user of it."""
from hx import *
D = load_val(); y, M, off, fid = D["y"], D["M"], D["off"], D["fid"]; fh = np.load(OUT / "bank_val.npz")["feed_has"]
ctx = ctx_cols(D); Xc = np.stack([ctx[k] for k in ctx], 1).astype(np.float32); nc = list(ctx)
rng = np.random.default_rng(123); eps = rng.standard_normal(len(y)); cov = fh[fid]; sgn = np.where(np.repeat(M, M) >= np.median(M[fh]), 1.0, -1.0)
p0 = ndcg_per_feed(D["base"], y, off, M)
print("delta scan (P1 additive, oracle in-sample best linear weight, all-feed):")
best = None
for dl in (0.3, 0.4, 0.5, 0.65, 0.8):
    s = np.where(cov, eps + dl * y, 0.0); z = np.where(cov, zfeed(s, off, M), 0.0)
    g = max((ndcg_per_feed(D["base"] + w * z, y, off, M) - p0).mean() for w in (0.02, 0.05, 0.1, 0.2, 0.3))
    print(f"  delta {dl}: oracle linear gain {g:+.5f}")
    if best is None or abs(g - 0.001) < abs(best[1] - 0.001): best = (dl, g)
dl = best[0]; print(f"using delta {dl} (oracle linear {best[1]:+.5f})", flush=True)
for name, s in (("P1 additive", np.where(cov, eps + dl * y, 0.0)), ("P2 sign-flip by M", np.where(cov, eps + dl * y * sgn, 0.0))):
    z = np.where(cov, zfeed(s, off, M), 0.0)
    d, wc = nested_lin(D, z); print(f"{name}: linear nested {fmt(d)} w={wc}")
    # oracle for the interaction: per-sign weights, in-sample
    zs = z * sgn if "P2" in name else z
    g = max((ndcg_per_feed(D["base"] + w * zs, y, off, M) - p0).mean() for w in (0.02, 0.05, 0.1, 0.2, 0.3)); print(f"   oracle (knows the interaction), in-sample: {g:+.5f}")
    X = np.hstack([Xc, s[:, None].astype(np.float32), z[:, None].astype(np.float32)])
    d1, det, _ = resid_cv(D, X, nc + ["s", "s_fz"], name + " / LGB residual all feeds")
    d2, det, _ = resid_cv(D, X, nc + ["s", "s_fz"], name + " / LGB residual covered feeds", feed_mask=fh)
    dd = np.where(fh, d2, 0.0); print(f"   covered-only model as all-feed delta {fmt(dd)}  (recovers {dd.mean() / g:.0%} of the oracle; all-feed model {np.nanmean(d1) / g:.0%})", flush=True)
