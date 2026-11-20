"""Follow-up to (d): TO beats FT on seen users when the positive sits in a category the user clicked in an earlier labelled feed (label-conditioned).
Label-free candidate-level version: add w * (z_TO - z_FT) * g, g in {seen feed, candidate's category clicked before, candidate's category shown before and
never clicked}, w picked on the fit half (linear nested), and all g-terms + bank signals in the LightGBM residual."""
from hx import *
V = load_val(); y, M, off, n, uid, fid = V["y"], V["M"], V["off"], V["n"], V["uid"], V["fid"]
Z = np.load(OUT / "bank_val.npz"); Xb = Z["X"]; nb_ = [str(s) for s in Z["names"]]; g = lambda k: Xb[:, nb_.index(k)].astype(np.float64)
seen = np.repeat((uid != 0), M) & (g("u_feeds") > 0); diff = V["to"] - V["ft"]
G = {"all candidates (= global w_ft shift)": np.ones(len(y)), "seen-feed candidates": seen.astype(float), "category clicked before": (g("uc_y") > 0).astype(float),
     "category shown before, never clicked": ((g("uc_n") > 0) & (g("uc_y") == 0)).astype(float), "category not shown before (seen feeds)": (seen & (g("uc_n") == 0)).astype(float)}
print("signal = (z_TO - z_FT) x g ; linear nested, 5 user-split seeds")
for k, gg in G.items():
    r = []
    for sd in range(5):
        d, wc = nested_lin(V, diff * gg, ws=(-0.5, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5), seed=sd); hf = user_half(uid, sd); r.append((d.mean(), se(d), d[hf].mean(), d[~hf].mean(), wc))
    print(f"  {k:40s} rows {gg.mean():.3f} | " + " | ".join(f"s{sd}: {a:+.5f}+-{b:.5f} (halves {c:+.5f}/{e:+.5f}) w={w}" for sd, (a, b, c, e, w) in enumerate(r[:2])) + f" | mean over 5 seeds {np.mean([x[0] for x in r]):+.5f}", flush=True)
# also the plain signals 'category clicked before' on each branch
for k in ("category clicked before",):
    for sd in (0, 1): d, wc = nested_lin(V, np.where(seen, zfeed(G[k], off, M), 0.0), seed=sd); print(f"  plain flag '{k}' (feed z): seed {sd} {fmt(d)} w={wc}")
ctx = ctx_cols(V); cols = dict(ctx); cols.update(to_minus_ft=diff, seen=seen.astype(float))
for k, gg in list(G.items())[1:]: cols["d_x_" + k.replace(" ", "_")[:24]] = diff * gg
for k in ("uc_y", "uc_n", "uc_ctr_log_fz", "u_feeds", "u_y"): cols[k] = g(k)
X = np.stack([np.asarray(v, np.float32) for v in cols.values()], 1)
for sd in (0, 1): d, det, _ = resid_cv(V, X, list(cols), f"branch-trust LGB residual, split seed {sd}", seed=sd)
