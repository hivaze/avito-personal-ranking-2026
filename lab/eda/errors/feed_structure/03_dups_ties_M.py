"""(a) near-duplicate groups: label sharing, model score spread, who wins inside a group; (b) exact / near ties in the base
scores and their NDCG cost; (c) feed size M vs npos, positive coherence and model quality; (d) oracle bound for top-1 propagation."""
from fs_common import *
import numba as nb

BASE = sys.argv[1] if len(sys.argv) > 1 else "nn_final2_val_oof"
d = load_val()
X, miss = load_emb_norm(d["emb_idx"])
M, off, t, fid = d["M"], d["off"], d["target"].astype(np.int64), d["fid"]
use = ~d["leak"]
npos = np.bincount(fid, t, len(M))
s = np.load(PREDS / f"{BASE}.npy").astype(np.float32)
z = zfeed(s, off, M)
f4r = np.load(CACHE / "f4r.npy"); f5r = np.load(CACHE / "f5r.npy")
base = ndcg_per_feed(s, t, off, M)
print(f"base {BASE} non-leak {base[use].mean():.5f}")


@nb.njit(parallel=True, cache=True)
def dup_stats(X, miss, off, M, t, z, thr, use):
    # per positive: has dup (cos>thr), n dups, n positive dups, rank of positive z within its group (0 = best), group size
    n = len(t)
    has = np.zeros(n, np.int64); ndup = np.zeros(n, np.int64); npd = np.zeros(n, np.int64)
    rk = np.full(n, -1, np.int64); maxdz = np.full(n, np.nan)
    for f in nb.prange(len(M)):
        if not use[f]: continue
        o = off[f]; m = M[f]
        Xf = X[o:o + m]
        C = Xf @ Xf.T
        for i in range(m):
            if miss[o + i]: continue
            r = 0; mx = 0.0
            for j in range(m):
                if j == i or miss[o + j]: continue
                if C[i, j] > thr:
                    ndup[o + i] += 1
                    npd[o + i] += t[o + j]
                    if z[o + j] > z[o + i]: r += 1
                    dz = abs(z[o + j] - z[o + i])
                    if dz > mx: mx = dz
            if ndup[o + i] > 0:
                has[o + i] = 1; rk[o + i] = r; maxdz[o + i] = mx
    return has, ndup, npd, rk, maxdz


out = {}
row = use[fid]
print("\n(a) near-duplicate groups (candidate-level, non-leak feeds)")
for thr in (0.9, 0.95, 0.98):
    has, ndup, npd, rk, maxdz = dup_stats(X, miss, off, M, t, z, thr, use)
    P = row & (t == 1); N = row & (t == 0)
    hp = has[P].mean(); hn = has[N].mean()
    share = npd[P & (has == 1)].sum() / ndup[P & (has == 1)].sum()
    anyp = (npd[P & (has == 1)] > 0).mean()
    # positive rank within its group: expected under random = ndup/2
    sel = P & (has == 1)
    rk_obs = rk[sel].mean(); rk_rand = ndup[sel].mean() / 2
    p_best = (rk[sel] == 0).mean(); p_best_rand = (1.0 / (ndup[sel] + 1)).mean()
    # median |dz| between a positive and its dups vs between a random negative and its dups
    print(f"  cos>{thr}: P(has dup | pos) {hp:.4f} vs | neg {hn:.4f}; positives with dup: {sel.sum()}; share of their dups that are positive {share:.4f}; "
          f"P(>=1 positive dup) {anyp:.4f}; pos is best of its group {p_best:.4f} (random {p_best_rand:.4f}); mean #dups above pos {rk_obs:.2f} (random {rk_rand:.2f}); "
          f"median max|dz| to dups: pos {np.nanmedian(maxdz[sel]):.3f}, neg {np.nanmedian(maxdz[N & (has==1)]):.3f}")
    out[f"dup{thr}"] = dict(p_has_pos=hp, p_has_neg=hn, n_pos_with_dup=int(sel.sum()), share_dups_pos=share, p_any_pos_dup=anyp,
                            p_best=p_best, p_best_rand=p_best_rand, rk_obs=rk_obs, rk_rand=rk_rand)
    if thr == 0.95:
        # inside groups with a positive: which features separate the positive from its (negative) duplicates?
        # pairwise: for each positive with dups, compare to the mean of its negative dups via feature ranks
        pass

# (a2) within dup groups: feature comparison positive vs its negative duplicates (pairwise win-rate)
@nb.njit(parallel=True, cache=True)
def dup_pair_wins(X, miss, off, M, t, feats, thr, use):
    K = feats.shape[1]
    W = np.zeros((len(M), K)); Tt = np.zeros((len(M), K)); Np = np.zeros(len(M))
    for f in nb.prange(len(M)):
        if not use[f]: continue
        o = off[f]; m = M[f]
        Xf = X[o:o + m]; C = Xf @ Xf.T
        for i in range(m):
            if t[o + i] != 1 or miss[o + i]: continue
            for j in range(m):
                if j == i or t[o + j] == 1 or miss[o + j] or C[i, j] <= thr: continue
                Np[f] += 1
                for k in range(K):
                    if feats[o + i, k] > feats[o + j, k]: W[f, k] += 1
                    elif feats[o + i, k] == feats[o + j, k]: Tt[f, k] += 1
    return W, Tt, Np

fn = ["z_base", "f1", "f2", "f3", "f4", "f5"]
F = np.stack([z, d["f1"], d["f2"], d["f3"], d["f4"], d["f5"]], 1).astype(np.float64)
for thr in (0.95,):
    W, Tt, Np = dup_pair_wins(X, miss, off, M, t, F, thr, use)
    print(f"  pos-vs-negative-duplicate pairs (cos>{thr}): {int(Np.sum())} pairs in {(Np>0).sum()} feeds; P(pos > dup) / P(tie):")
    for k, nm in enumerate(fn):
        print(f"    {nm:7s} win {W[:,k].sum()/Np.sum():.4f}  tie {Tt[:,k].sum()/Np.sum():.4f}")
    # compare to pos-vs-any-negative pairs in the same feeds, for z
    out["dup_pair_wins"] = {nm: dict(win=W[:, k].sum() / Np.sum(), tie=Tt[:, k].sum() / Np.sum()) for k, nm in enumerate(fn)}

# (b) ties in base scores
print("\n(b) ties")
@nb.njit(parallel=True, cache=True)
def tie_stats(off, M, s, t, eps):
    # per feed: number of positive-negative pairs with |ds| <= eps, total pos-neg pairs
    n = len(M); a = np.zeros(n); b = np.zeros(n); ex = np.zeros(n)
    for f in nb.prange(n):
        o = off[f]; m = M[f]
        for i in range(m):
            if t[o + i] != 1: continue
            for j in range(m):
                if t[o + j] == 1: continue
                b[f] += 1
                dd = abs(s[o + i] - s[o + j])
                if dd <= eps: a[f] += 1
                if dd == 0: ex[f] += 1
    return a, b, ex
for nm in [BASE, "nn_final2_trainonly_val", "stk2_res2_r3_val"]:
    ss = np.load(PREDS / f"{nm}.npy").astype(np.float32)
    zz = zfeed(ss, off, M)
    pess = ndcg_per_feed(ss, t, off, M)
    rng = np.random.default_rng(0)
    opt = ndcg_per_feed(ss.astype(np.float64) + 0.0, 1 - t, off, M)  # placeholder (not used)
    # optimistic ties: sort with target desc on ties -> use lexsort trick: add tiny positive to positives below float32 resolution
    srt = ss.astype(np.float64) + t * 1e-12
    optim = ndcg_per_feed(srt, t, off, M)
    a, b, ex = tie_stats(off, M, zz, t, 1e-3)
    a2, _, _ = tie_stats(off, M, zz, t, 1e-2)
    n_tied = 0
    print(f"  {nm}: pessimistic {pess[use].mean():.5f} optimistic {optim[use].mean():.5f} (gap {optim[use].mean()-pess[use].mean():.6f}); "
          f"exact pos-neg ties {ex[use].sum():.0f} of {b[use].sum():.0f} pairs; |dz|<=1e-3: {a[use].sum()/b[use].sum():.5f}; |dz|<=1e-2: {a2[use].sum()/b[use].sum():.5f}")
    out[f"ties_{nm}"] = dict(pess=pess[use].mean(), optim=optim[use].mean(), exact=ex[use].sum(), frac_1e3=a[use].sum() / b[use].sum(), frac_1e2=a2[use].sum() / b[use].sum())

# (b2) near-tie tiebreak by a second model: s' = z + eps * z2 for small eps
z2s = {nm: zfeed(np.load(PREDS / f"{nm}.npy"), off, M) for nm in ["nn_final2_trainonly_val", "stk2_res2_r3_val", "cb_tr300k_none_val"]}
for nm, z2 in z2s.items():
    for eps in (1e-4, 1e-2, 0.05, 0.2):
        p = ndcg_per_feed(z + eps * z2, t, off, M)
        mu, se, n = paired(p, base, use)
        print(f"  tiebreak/blend z + {eps:g} z({nm}): {mu:+.5f} ± {se:.5f}")

# (c) feed size M
print("\n(c) feed size M (non-leak)")
bins = [0, 10, 20, 30, 50, 100, 150, 200, 250, 301]
top1 = np.zeros(len(M), bool); prk = np.full(len(M), np.nan)
for f in range(len(M)):
    o, m = off[f], M[f]
    zf = z[o:o + m]; tf = t[o:o + m]
    top1[f] = tf[np.argmax(zf)] == 1
    r = np.argsort(np.argsort(-zf))
    prk[f] = (r[tf == 1] / max(m - 1, 1)).mean() if tf.sum() > 0 else np.nan
# random NDCG closed form per feed
def rand_ndcg(m, k):
    disc = 1 / np.log2(np.arange(m) + 2)
    return k * disc.mean() / disc[:k].sum()
rnd = np.array([rand_ndcg(M[f], int(npos[f])) if npos[f] > 0 else 1.0 for f in range(len(M))])
S = np.load(CACHE / "pair_S.npy")
with np.errstate(invalid="ignore", divide="ignore"):
    sc_pp = S[:, 0, 1] / S[:, 0, 0]; sc_pn = S[:, 1, 1] / S[:, 1, 0]
rows = []
for lo, hi in zip(bins[:-1], bins[1:]):
    mb = use & (M >= lo) & (M < hi)
    mm = mb & (npos >= 2)
    r = dict(M=f"[{lo},{hi})", n=int(mb.sum()), share=mb.sum() / use.sum(), mean_npos=npos[mb].mean(), frac_multi=(npos[mb] >= 2).mean(),
             ndcg=base[mb].mean(), rand=rnd[mb].mean(), lift=base[mb].mean() - rnd[mb].mean(), top1=top1[mb].mean(),
             pos_rank_pct=np.nanmean(prk[mb]), samecat_pp=np.nanmean(sc_pp[mm]), samecat_pn=np.nanmean(sc_pn[mm]))
    rows.append(r)
    print(f"  M {r['M']:10s} n={r['n']:5d} npos {r['mean_npos']:.2f} multi {r['frac_multi']:.3f} | NDCG {r['ndcg']:.4f} random {r['rand']:.4f} lift {r['lift']:+.4f} | "
          f"P(top1+) {r['top1']:.3f} | mean pos rank pct {r['pos_rank_pct']:.3f} | multi-pos feeds: same-cat pp {r['samecat_pp']:.3f} vs pn {r['samecat_pn']:.3f}")
out["M_bins"] = rows
print("  NDCG by npos:", {int(k): round(float(base[use & (npos == k)].mean()), 4) for k in range(1, 6)},
      " random:", {int(k): round(float(rnd[use & (npos == k)].mean()), 4) for k in range(1, 6)})
print("  corr(npos, M) non-leak: %.3f" % np.corrcoef(npos[use], M[use])[0, 1])

# (d) oracle-conditioned propagation bound: keep top-1 fixed, add a*cos(i, top1) to the others.
@nb.njit(parallel=True, cache=True)
def prop_keep_top1(X, miss, off, M, z, a, samecat_only, cat):
    out = z.copy()
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        if m < 2: continue
        t1 = np.argmax(z[o:o + m])
        top = z[o + t1]
        for i in range(m):
            if i == t1 or miss[o + i] or miss[o + t1]: continue
            if samecat_only and cat[o + i] != cat[o + t1]: continue
            c = 0.0
            for k in range(X.shape[1]): c += X[o + i, k] * X[o + t1, k]
            out[o + i] = z[o + i] + a * c
        mx = -1e18
        for i in range(m):
            if i != t1 and out[o + i] > mx: mx = out[o + i]
        if mx >= top: out[o + t1] = mx + 1.0
    return out
print("\n(d) propagation from the top-1 with top-1 kept first; split by whether top-1 is truly positive (oracle split)")
cat = d["category_id"].astype(np.int64)
for a in (0.2, 0.5, 1.0, 2.0):
    p = ndcg_per_feed(prop_keep_top1(X, miss, off, M, z, a, False, cat), t, off, M)
    dd = p - base
    g1 = dd[use & top1 & (npos >= 2)].mean(); g0 = dd[use & ~top1].mean()
    mu, se, n = paired(p, base, use)
    p2 = ndcg_per_feed(prop_keep_top1(X, miss, off, M, z, a, True, cat), t, off, M); dd2 = p2 - base
    print(f"     same-cat-only a={a}: overall {dd2[use].mean():+.5f} ± {dd2[use].std(ddof=1)/np.sqrt(use.sum()):.5f} | top1+ & npos>=2 {dd2[use & top1 & (npos >= 2)].mean():+.5f} | top1- {dd2[use & ~top1].mean():+.5f}")
    print(f"  a={a}: overall {mu:+.5f} ± {se:.5f} | top1+ & npos>=2 (n={int((use&top1&(npos>=2)).sum())}) {g1:+.5f} | top1- (n={int((use&~top1).sum())}) {g0:+.5f}")
json.dump(out, open(OUT / f"out_03_{BASE}.json", "w"), indent=1, default=float)
