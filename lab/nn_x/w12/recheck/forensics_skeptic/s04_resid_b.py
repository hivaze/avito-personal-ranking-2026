"""(b) the existing feature columns as a BLOCK on top of W8 (LightGBM lambdarank residual, user-split 2-fold).
val arms: b_top1 (cos_het1 + its feed z), b_sel12 (12 columns selected by the linear screen INSIDE the fit half), b_all (all leak-safe columns),
          b_own (own-feed label-free columns only = the block that exists identically on the holdout).
hold arm: b_own (within-holdout 2-fold).   Usage: s04_resid_b.py val|hold"""
from hx import *
from concurrent.futures import ThreadPoolExecutor
which = sys.argv[1]
D = load_val() if which == "val" else load_hold(); y, M, off = D["y"], D["M"], D["off"]
ctx = ctx_cols(D); Xc = np.stack([ctx[k] for k in ctx], 1).astype(np.float32); nc = list(ctx)
src = FEATS / ("val" if which == "val" else "train"); nrow = len(y)
OWN = ("cos_", "dot_", "seq_", "ctx_", "cat_", "maxsim", "meansim", "attn32", "centrality", "item_norm", "emb_missing", "u_het", "hist_n_uniq_cat", "feed_cat_share")
OWN_EXACT = {f"f{i}{s}" for i in range(1, 6) for s in ("", "_z", "_rk", "_gap")}
BAN = lambda n: ("id_contrib" in n) or n.startswith("kv64") or n in ("M", "H", "hour", "category_id") or (n.startswith("it_hist_all") and "luo" not in n) or (n.startswith("it_hist_et") and not n.endswith("_luo"))
names = sorted(p.stem for p in src.glob("*.npy")); own = [n for n in names if (n.startswith(OWN) or n in OWN_EXACT) and not BAN(n) and not n.startswith("seq_cv")]  # seq_cv = global co-visitation, not own-feed
def ld(n):
    x = np.load(src / f"{n}.npy", mmap_mode="r")
    if x.shape[0] != (nrow if D["rows"] is None else 246745836) or x.ndim != 1: return None
    x = np.asarray(x if D["rows"] is None else x[D["rows"]], np.float32); return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
print(f"== (b) {which}: base {ndcg_per_feed(D['base'], y, off, M).mean():.5f}", flush=True)
if which == "val":
    safe = [n for n in names if not BAN(n)]; cols = {}
    for n in safe:
        x = ld(n)
        if x is not None and x.std() > 0: cols[n] = x
    safe = list(cols); print("leak-safe row columns", len(safe), "own-feed label-free", len([n for n in own if n in cols]), flush=True)
    # linear screen: per-feed gain of base + w z(col) for 4 weights, all feeds; the selection below uses fit-half feeds only
    p0 = ndcg_per_feed(D["base"], y, off, M); ws = (-0.05, -0.02, 0.02, 0.05)
    def scr(n):
        z = zfeed(cols[n], off, M); return np.stack([ndcg_per_feed(D["base"] + w * z, y, off, M) - p0 for w in ws]).astype(np.float32)
    with ThreadPoolExecutor(12) as ex: G = dict(zip(safe, ex.map(scr, safe)))
    half = user_half(D["uid"], 0)
    full = sorted(((G[n].mean(1).max(), n) for n in safe), reverse=True)[:8]; print("in-sample top columns (all feeds, best of 4 w):", [(n, round(float(g), 5)) for g, n in full], flush=True)
    def run_sel(K, tag):
        """selection inside the fit half: done by running resid_cv per direction with its own column set"""
        d = np.full(len(M), np.nan); out = []
        for a in (True, False):
            fitm = half == a; top = [n for _, n in sorted(((G[n][:, fitm].mean(1).max(), n) for n in safe), reverse=True)[:K]]
            X = np.hstack([Xc] + [np.stack([cols[n], zfeed(cols[n], off, M).astype(np.float32)], 1) for n in top]); nm = nc + [s for n in top for s in (n, n + "_fz")]
            dd, det, _ = resid_cv(D, X, nm, f"{tag} dir fit-half {int(a)} (cols {top[:6]}...)", seed=0)
            d[half != a] = dd[half != a]; out.append(det[0 if a else 1])
        print(f"  [{tag}] BOTH halves with in-half selection: {fmt(d)} (z {d.mean() / max(se(d), 1e-12):+.2f}); per direction: " + "; ".join(f"fit {int(o['fit_half'])} -> {o['delta']:+.5f}+-{o['se']:.5f} rounds {o['rounds']}" for o in out), flush=True)
        return d
    X1 = np.hstack([Xc, cols["cos_het1"][:, None], zfeed(cols["cos_het1"], off, M).astype(np.float32)[:, None]])
    d1, _, _ = resid_cv(D, X1, nc + ["cos_het1", "cos_het1_fz"], "b_top1 cos_het1")
    dl, wc = nested_lin(D, zfeed(cols["cos_het1"], off, M)); print(f"  linear nested cos_het1 (report: +0.00044 +- 0.00020): {fmt(dl)} w={wc}; per half: {fmt(dl[half])} / {fmt(dl[~half])}", flush=True)
    d12 = run_sel(12, "b_sel12"); d4 = run_sel(4, "b_sel4")
    Xa = np.hstack([Xc, np.stack([cols[n] for n in safe], 1)]); da, _, _ = resid_cv(D, Xa, nc + safe, "b_all (all leak-safe columns)")
    ow = [n for n in own if n in cols]; Xo = np.hstack([Xc, np.stack([cols[n] for n in ow], 1)]); do, _, _ = resid_cv(D, Xo, nc + ow, "b_own (own-feed label-free block)")
    np.savez(OUT / "s04_val.npz", d1=d1, d12=d12, d4=d4, da=da, do=do)
else:
    cols = {}
    for n in own:
        x = ld(n)
        if x is not None and x.std() > 0: cols[n] = x
    ow = list(cols); print("own-feed label-free columns on the holdout", len(ow), flush=True)
    for n in ("cos_het1", "cos_h50", "meansim32", "seq_max_old", "cat_min_hpos", "seq_dcos_deep"):
        if n in cols:
            dl, wc = nested_lin(D, zfeed(cols[n], off, M)); print(f"  linear nested {n:14s} {fmt(dl)} w={wc}", flush=True)
    X1 = np.hstack([Xc, cols["cos_het1"][:, None], zfeed(cols["cos_het1"], off, M).astype(np.float32)[:, None]])
    d1, _, _ = resid_cv(D, X1, nc + ["cos_het1", "cos_het1_fz"], "b_top1 cos_het1 (holdout)")
    Xo = np.hstack([Xc, np.stack([cols[n] for n in ow], 1)]); do, _, _ = resid_cv(D, Xo, nc + ow, "b_own (holdout)", cfgs=CFGS[:2], ckpt=(0, 10, 20, 40, 80, 150), inner_k=2, bag=1)  # reduced grid: 2.3M rows x ~110 columns per half
    np.savez(OUT / "s04_hold.npz", d1=d1, do=do)
