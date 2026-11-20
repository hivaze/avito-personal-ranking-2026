"""Reviewer addendum: centred identity of the mean files; student in-sample gap from the h1/h2 dumps; OOF teacher as a control."""
import numpy as np, polars as pl
R = "<repo>"; K = f"{R}/nn_x/w12/kd/out"
def P(*a): print(*a, flush=True)
f = pl.read_parquet(f"{R}/data/flat/train/feeds.parquet"); Mt = f["M"].to_numpy().astype(np.int64); offt = f["cand_off"].to_numpy().astype(np.int64)
p = np.nan_to_num(np.load(f"{R}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0); p[Mt < 50] = 0.0
cand = np.where((p > 0.9) & (Mt >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
ins = np.sort(np.random.default_rng(4242).choice(np.setdiff1d(cand, hold), 30000, replace=False))
rows = lambda ids: np.concatenate([np.arange(o, o + m) for o, m in zip(offt[ids], Mt[ids])])
hr, ir = rows(hold), rows(ins); Mh, Mi = Mt[hold], Mt[ins]
yt = np.load(f"{R}/data/flat/train/cand_target.npy", mmap_mode="r"); yh = np.asarray(yt[hr], np.float64); yi = np.asarray(yt[ir], np.float64)
def nd(s, t, M):
    n = len(M); fid = np.repeat(np.arange(n), M); off = np.concatenate([[0], np.cumsum(M)[:-1]])
    o = np.lexsort((t, -s.astype(np.float64), fid)); rk = np.arange(len(s)) - np.repeat(off, M); d = 1.0 / np.log2(rk + 2.0)
    dcg = np.bincount(fid, weights=t[o] * d, minlength=n); npos = np.bincount(fid, weights=t, minlength=n)
    return dcg / np.bincount(fid, weights=(rk < np.repeat(npos, M)) * d, minlength=n)
def cen(s, M):
    fid = np.repeat(np.arange(len(M)), M); return s - (np.bincount(fid, weights=s) / M)[fid]
L = lambda t, r: np.asarray(np.load(f"{K}/teacher_{t}.npy", mmap_mode="r")[r], np.float64)
H = {t: L(t, hr) for t in ("a", "b", "c", "d", "e", "f", "g1", "g2", "p1", "p2")}
for name, parts, n in (("mean11", "abcdef", 11), ("mean25", ["a", "b", "c", "d", "e", "f", "g1", "g2", "p1", "p2"], 25)):
    h = L(name, hr); s = cen(sum(H[t] for t in parts) / n, Mh)
    P(f"teacher_{name} holdout rows vs per-feed-centred sum(components)/{n}: max abs diff {np.abs(h - s).max():.2e}; corr {np.corrcoef(h, s)[0, 1]:.10f}; max |per-feed mean of the file| {np.abs(h - cen(h, Mh)).max():.2e}")
se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
res = {}
for t in ("h1", "h2", "oof", "mean11", "mean25"):
    a, b = nd(L(t, hr), yh, Mh), nd(L(t, ir), yi, Mi); res[t] = (a, b)
    P(f"{t:7s}: NDCG holdout {a.mean():.5f} | 30,000 in-sample draw-set feeds {b.mean():.5f} | gap {b.mean() - a.mean():+.5f} +- {np.sqrt(se(a) ** 2 + se(b) ** 2):.5f}")
for s, t in (("h1", "mean11"), ("h2", "mean25"), ("h1", "oof"), ("mean11", "oof")):
    dh, di = res[s][0] - res[t][0], res[s][1] - res[t][1]
    P(f"{s} - {t}: holdout {dh.mean():+.5f} +- {se(dh):.5f} | in-sample feeds {di.mean():+.5f} +- {se(di):.5f} | diff-in-diff {di.mean() - dh.mean():+.5f} +- {np.sqrt(se(dh) ** 2 + se(di) ** 2):.5f}")
