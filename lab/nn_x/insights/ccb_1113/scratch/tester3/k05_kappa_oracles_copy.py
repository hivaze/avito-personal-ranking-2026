"""(i) Ceiling of perfect disambiguation inside near-duplicate groups (oracle). (ii) NDCG-optimal multi-positive correction:
E[NDCG] = sum_i disc(r_i) E[t_i / IDCG(N)], so the optimal key is P(t_i) * E[1/IDCG(N) | t_i = 1], not P(t_i) * E[1/N | t_i = 1]
(what a listwise softmax with target y/N learns). Estimate E[N | t_i=1] = 1 + sum_j min(1, p_j * lift(cos_ij, samecat)) with p
from a logistic calibration of z and lift from table C, both fitted on one half of the feeds and applied to the other (2-fold)."""
from fs_common import *
import numba as nb
from sklearn.linear_model import LogisticRegression

BASE = sys.argv[1] if len(sys.argv) > 1 else "nn_final2_val_oof"
EDGES = np.array([-1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.999, 1.01], np.float32)
d = load_val()
X, miss = load_emb_norm(d["emb_idx"])
M, off, t, fid = d["M"], d["off"], d["target"].astype(np.int64), d["fid"]
cat = d["category_id"].astype(np.int64)
use = ~d["leak"]
npos = np.bincount(fid, t, len(M))
s = np.load(PREDS / f"{BASE}.npy").astype(np.float64)
z = zfeed(s, off, M)
base = ndcg_per_feed(z, t, off, M)
print(f"base {BASE} non-leak {base[use].mean():.5f}")
# within-feed sd of raw logits (scale of the base score)
sd = np.sqrt(np.bincount(fid, s * s, len(M)) / M - (np.bincount(fid, s, len(M)) / M) ** 2)
print(f"raw score within-feed sd: median {np.median(sd[use]):.3f} (q10 {np.quantile(sd[use],0.1):.3f}, q90 {np.quantile(sd[use],0.9):.3f})")


@nb.njit(parallel=True, cache=True)
def oracle_dup(X, miss, off, M, t, z, thr):
    out = z.copy()
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        Xf = X[o:o + m]; C = Xf @ Xf.T
        for i in range(m):
            if t[o + i] != 1 or miss[o + i]: continue
            mx = z[o + i]
            for j in range(m):
                if j != i and (not miss[o + j]) and C[i, j] > thr and z[o + j] > mx: mx = z[o + j]
            if mx > z[o + i]: out[o + i] = mx + 1e-6
    return out

print("\n(i) oracle: each positive lifted to the best score of its near-duplicate group")
for thr in (0.9, 0.95, 0.98):
    p = ndcg_per_feed(oracle_dup(X, miss, off, M, t, z, thr), t, off, M)
    mu, se, n = paired(p, base, use)
    print(f"  cos>{thr}: {mu:+.5f} ± {se:.5f}")


@nb.njit(parallel=True, cache=True)
def expN(X, miss, off, M, p, cat, edges, L):
    out = np.ones(len(p))
    nb_ = len(edges) - 1
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        Xf = X[o:o + m]; C = Xf @ Xf.T
        for i in range(m):
            e = 1.0
            for j in range(m):
                if j == i: continue
                if miss[o + i] or miss[o + j]:
                    lf = 1.0
                else:
                    b = np.searchsorted(edges, C[i, j], side="right") - 1
                    b = min(max(b, 0), nb_ - 1)
                    lf = L[1 if cat[o + i] == cat[o + j] else 0, b]
                e += min(1.0, p[o + j] * lf)
            out[o + i] = e
    return out


def lift_table(H):
    # H: (2 anchor, 2 samecat, NB, 2 jlab) counts; lift = P(t_j|t_i=1,b,sc) / P(t_j|b,sc); shrink to 1 with prior 200 pairs
    L = np.ones((2, H.shape[2]))
    for sc in range(2):
        for b in range(H.shape[2]):
            n1 = H[1, sc, b].sum(); k1 = H[1, sc, b, 1]
            na = H[:, sc, b].sum(); ka = H[:, sc, b, 1].sum()
            pa = ka / max(na, 1)
            if pa <= 0: continue
            p1 = (k1 + 200 * pa) / (n1 + 200)
            L[sc, b] = p1 / pa
    return L

H = np.load(CACHE / "pair_H.npy")  # per feed
Ns = np.arange(1, 400)
idcg = np.cumsum(1 / np.log2(Ns + 1))
even = np.arange(len(M)) % 2 == 0
keys = {k: np.zeros(len(z)) for k in ("A_marg_idcg", "B_ratio")}
for half_fit in (even, ~even):
    fit = use & half_fit; app = ~half_fit
    rf = fit[fid]
    lr = LogisticRegression(C=100.0).fit(np.column_stack([z[rf], np.log(M[fid][rf])]), t[rf])
    p = lr.predict_proba(np.column_stack([z, np.log(M[fid])]))[:, 1]
    L = lift_table(H[fit].sum(0))
    EN = expN(X, miss, off, M, p, cat, EDGES, L)
    inv_idcg = np.interp(EN, Ns, 1 / idcg); inv_n = 1 / EN
    ra = app[fid]
    keys["A_marg_idcg"][ra] = (np.log(p) + np.log(inv_idcg))[ra]
    keys["B_ratio"][ra] = (np.log(p) + np.log(inv_idcg) - np.log(inv_n))[ra]
print("\n(ii) NDCG-optimal multi-positive correction (2-fold fitted)")
print(f"  lift table (fit on all non-leak, for reference): samecat=1 {np.round(lift_table(H[use].sum(0))[1],2)}")
print(f"  E[N|t_i=1] estimate: median {np.median(EN):.3f}, q90 {np.quantile(EN,0.9):.3f}")
for k, v in keys.items():
    pk = ndcg_per_feed(v, t, off, M)
    mu, se, n = paired(pk, base, use)
    dd = pk - base
    print(f"  {k}: {mu:+.5f} ± {se:.5f} | multi {dd[use&(npos>=2)].mean():+.5f} single {dd[use&(npos==1)].mean():+.5f}")
