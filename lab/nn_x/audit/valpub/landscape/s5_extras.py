"""s5: robustness extras. (a) S2's low Pearson is a rank-transform artefact: within-feed Spearman; (b) likelihood-ratio tests of
1:1 val->public transfer (b = 1) against a free transfer factor, with and without a common offset (reference-free), and the same
with PF7x as the reference; (c) PRE-REGISTERED predictions of the public delta vs G3 for the unscored pools under the competing
models, with the public paired SE and the separation each pool offers; (d) holdout private-like subset (Sun >= 12h, Mon, Tue)
for all pools; (e) order-of-magnitude of the val-selection optimism (winner's curse) for a val-picked pool."""
from lib import *  # noqa
import calib
from scipy.stats import chi2 as chi2dist, norm, rankdata

f, M, off = meta("val"); y = val_labels()
ft, Mt, ot = meta("test"); pubm = public_mask(); n_pub = int(pubm.sum())
Wpub = calib.weights("public")
hd = holdout()
fid_v = np.repeat(np.arange(len(M)), M); fid_t = np.repeat(np.arange(len(Mt)), Mt)

# (a) Spearman S2 vs G3
def feed_rank(s, M_, o_):
    r = np.empty(len(s), np.float64)
    for i in range(len(M_)):
        a, b = o_[i], o_[i] + M_[i]; r[a:b] = rankdata(s[a:b])
    return r
for split, M_, o_, fid, sel in [("val", M, off, fid_v, {"all": np.ones(len(M), bool)}),
                                ("test", Mt, ot, fid_t, {"public": pubm, "private": ~pubm})]:
    rs2 = zf(o_, M_, feed_rank(pool_scores("S2", split), M_, o_)); rg3 = zf(o_, M_, feed_rank(pool_scores("G3", split), M_, o_))
    rm2 = zf(o_, M_, feed_rank(pool_scores("M2", split), M_, o_))
    c = np.bincount(fid, rs2 * rg3, len(M_)) / M_; c2 = np.bincount(fid, rm2 * rg3, len(M_)) / M_
    for k, m in sel.items():
        print(f"(a) within-feed SPEARMAN vs G3 on {split}/{k}: S2 {c[m].mean():.4f}  (M2 {c2[m].mean():.4f})")

# (b) LR tests
CON = [n for n in SCORED if n != "G3"]
nd = {n: perfeed_val(n) for n in SCORED}
def contrasts(ref):
    xs, rs, ses = [], [], []
    for n in SCORED:
        if n == ref:
            continue
        d = nd[n] - nd[ref]; m_v, se_v = mse(d); m_pub, _ = mse(d, Wpub)
        se_pub = np.sqrt(np.average((d - m_pub) ** 2, weights=Wpub) / n_pub)
        xs.append(m_pub); rs.append(PUBLIC[n] - PUBLIC[ref]); ses.append(np.hypot(se_pub, se_v))
    return np.array(xs), np.array(rs), np.array(ses)
def wls(X, yv, ww):
    A = X.T @ (ww[:, None] * X); b = np.linalg.solve(A, X.T @ (ww * yv)); return b, np.linalg.inv(A)
for ref in ["G3", "PF7x"]:
    x, r, se = contrasts(ref); w = 1 / se ** 2
    c1 = ((r - x) ** 2 * w).sum()
    k = (w * x * r).sum() / (w * x * x).sum(); ck = ((r - k * x) ** 2 * w).sum()
    X = np.column_stack([np.ones(len(x)), x]); b, C = wls(X, r, w); cb = ((r - X @ b) ** 2 * w).sum()
    c1o = ((r - x - np.average(r - x, weights=w)) ** 2 * w).sum()
    print(f"\n(b) reference {ref} ({len(x)} contrasts): chi2(1:1) {c1:.2f} | free factor k {k:+.2f}: chi2 {ck:.2f}, LR vs k=1: dchi2 {c1 - ck:.2f} p {1 - chi2dist.cdf(c1 - ck, 1):.4f}"
          f" | with offset: 1:1+c chi2 {c1o:.2f} vs a+b*x (b {b[1]:+.2f}±{np.sqrt(C[1, 1]):.2f}) chi2 {cb:.2f}, LR vs b=1: dchi2 {c1o - cb:.2f} p {1 - chi2dist.cdf(c1o - cb, 1):.4f}"
          f" | 95% CI of b: [{b[1] - 1.96 * np.sqrt(C[1, 1]):+.2f}, {b[1] + 1.96 * np.sqrt(C[1, 1]):+.2f}]")

# (c) pre-registered predictions for unscored pools
x, r, se = contrasts("G3"); w = 1 / se ** 2
X = np.column_stack([np.ones(len(x)), x]); b3, _ = wls(X, r, w)
hold = {n: perfeed_holdout(n) for n in SCORED + UNSCORED}; g3h = hold["G3"]
hold_d6 = np.array([(hold[n] - g3h).mean() if hold.get(n) is not None else (0.0 if n == "TOonly" else -0.00006) for n in CON])
c5 = np.average(r - hold_d6, weights=w)
print(f"\n(c) PRE-REGISTERED public-delta predictions vs G3 for the UNSCORED pools (models fitted on the 6 scored contrasts; written before any of these is submitted):")
print(f"    H1 1:1 = pubM d ; H3 = {b3[0]:+.5f} + {b3[1]:+.3f} * pubM d ; H5o = holdout d {c5:+.5f}")
print(f"    {'pool':8s} {'pubM d':>9s} {'hold d':>9s} {'H1':>9s} {'H3':>9s} {'H5o':>9s} {'SE_pub':>8s} {'|H1-H3|/SE':>10s} {'|H1-H5o|/SE':>11s} {'|H3-H5o|/SE':>11s}")
for n in UNSCORED:
    d = perfeed_val(n) - nd["G3"]; m_pub, _ = mse(d, Wpub); se_pub = np.sqrt(np.average((d - m_pub) ** 2, weights=Wpub) / n_pub)
    hdl = (hold[n] - g3h).mean() if hold.get(n) is not None else np.nan
    h1 = m_pub; h3 = b3[0] + b3[1] * m_pub; h5 = hdl + c5
    print(f"    {n:8s} {m_pub:+9.5f} {hdl:+9.5f} {h1:+9.5f} {h3:+9.5f} {h5:+9.5f} {se_pub:8.5f} {abs(h1 - h3) / se_pub:10.2f} {abs(h1 - h5) / se_pub:11.2f} {abs(h3 - h5) / se_pub:11.2f}")

# (d) holdout private-like subset
wd = shiftlib.weekday(hd["ts"]); hr = shiftlib.msk_hour(hd["ts"])
privlike = ((wd == 6) & (hr >= 12)) | (wd == 0) | (wd == 1)
publike = (wd == 5) | ((wd == 6) & (hr < 12))
print(f"\n(d) holdout subsets: privlike (Sun>=12h, Mon, Tue) n {privlike.sum()}, publike (Sat, Sun<12h) n {publike.sum()}, other n {(~privlike & ~publike).sum()}; deltas vs G3:")
for n in SCORED + UNSCORED:
    if hold.get(n) is None or n == "G3":
        continue
    h = hold[n] - g3h
    print(f"    {n:8s} all {fmt(*mse(h))} | privlike {fmt(*mse(h[privlike]))} | publike {fmt(*mse(h[publike]))} | priv-pub {h[privlike].mean() - h[publike].mean():+.5f}±{np.hypot(mse(h[privlike])[1], mse(h[publike])[1]):.5f}")

# (e) winner's curse order of magnitude
print("\n(e) val-selection optimism: expected max of N iid N(0,1) draws = 1.54 (N 10), 2.04 (N 30), 2.25 (N 50); at SE_val(M2-G3) = 0.00034 the")
print("    expected optimism of the val-best of 30 near-copies is ~0.0007, the size of M2's public shortfall (-0.0010 ± 0.0006).")
for N in [10, 30, 50]:
    sims = np.random.default_rng(0).standard_normal((20000, N)).max(1).mean()
    print(f"    N {N:2d}: E[max] {sims:.2f} -> optimism at SE 0.00034: {sims * 0.00034:+.5f}; at SE 0.00028 (S2): {sims * 0.00028:+.5f}")
