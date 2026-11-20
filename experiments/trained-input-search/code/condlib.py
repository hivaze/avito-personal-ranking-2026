# The calibrated conditional score test (2026-11-16). It asks whether a stored column carries information
# about the click that is not already in the pool's score and in 84 of the ranker's own input columns: a null
# model is fitted on those, and the column is scored against its residual. Under 'nothing new' the statistic
# is close to N(0,1), which was checked with placebo columns.
# Imported by s06_cond.py and s07_shortlist.py in lab/reports/INPUTS_1116/residual_rows/; not run on its own.
# It found 25 of 284 stored columns on both witnesses; one survived the parity and lag checks.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/reports/INPUTS_1116/residual_rows/condlib.py
"""INVENTORY B: the conditional machinery of s06_cond.py as a module (same code: control block of the ranker's own inputs, null model, score test).
Used by s07_shortlist.py. CPU only, read-only."""
import sys, json, time
sys.path.insert(0, "<repo>/reports/INPUTS_1116/residual_rows")
from lens import *
import cand_specs as cs

sp = "val"   # callers set condlib.sp to the split of the data they pass ("val" | "hold")
POP8 = "seq_cv_invd20 seq_cv_pop seq_cv_norm it_hist_all_luo it_hist_et1_luo it_hist_et2_luo it_hist_et3_luo it_hist_et4_luo".split()
PRE2 = "pre2_fe_z pre2_new_z pre2_hmax32".split()
CTX = ("ctx_nd_hcent ctx_nd_knn_f1 ctx_nd_knn_h ctx_nd_knn_h_dev ctx_nd_knn_u ctx_nd_knn_u_dev ctx_nd_max ctx_nd_top5h ctx_nd_top5u ctx_nd_ucent "
       "ctx_hn_top1w ctx_hn_rrw ctx_hn_bestrk ctx_hn_margin ctx_hn_soft05 ctx_hn_e1_rrw ctx_hs_mutual ctx_hs_k128 ctx_hs_e1 ctx_hs_t10").split()
INMODEL = ("f1 f2 f3 f4 f5 emb_missing item_norm cos_u dot_u seq_max_all seq_max_new8 maxsim32 meansim32 attn32 seq_top3 "
           + " ".join(f"cos_het{t}" for t in range(1, 11)) + " cat_cnt_all cat_share_all " + " ".join(f"cat_cnt_et{t}" for t in range(1, 11)) + " cat_min_hpos feed_cat_share").split()
RK = "f1 f2 f3 f4 f5 cos_u seq_max_all attn32 seq_top3 cat_cnt_all seq_cv_invd20 it_hist_all_luo pre2_fe_z pre2_new_z pre2_hmax32".split()
STORE = {n: s for g, n, s in cs.specs() if g == "store"}


@nb.njit(parallel=True, cache=True)
def clogit_big(X, y, off, M, theta, use, T):
    """as lens.clogit, with the information matrix accumulated per thread chunk (K up to ~100). Also returns row probabilities and per-feed E_p[X]."""
    n = len(M); K = X.shape[1]
    ll = np.zeros(n); G = np.zeros((n, K)); Hc = np.zeros((T, K, K)); p_row = np.zeros(X.shape[0], np.float32); MU = np.zeros((n, K), np.float32)
    for c in nb.prange(T):
        lo = (n * c) // T; hi = (n * (c + 1)) // T
        mu = np.zeros(K); S2 = np.zeros((K, K))
        for f in range(lo, hi):
            o = off[f]; m = M[f]; npos = 0.0
            for i in range(m): npos += y[o + i]
            s = np.empty(m); mx = -1e30
            for i in range(m):
                v = 0.0
                for k in range(K): v += X[o + i, k] * theta[k]
                s[i] = v
                if v > mx: mx = v
            Z = 0.0
            for i in range(m):
                s[i] = np.exp(s[i] - mx); Z += s[i]
            mu[:] = 0.0
            for i in range(m):
                p = s[i] / Z; p_row[o + i] = p
                for k in range(K): mu[k] += p * X[o + i, k]
            for k in range(K): MU[f, k] = mu[k]
            if npos == 0.0 or not use[f]: continue
            S2[:, :] = 0.0; l = 0.0
            for i in range(m):
                p = s[i] / Z; yi = y[o + i]
                if yi > 0:
                    l += yi * np.log(p)
                    for k in range(K): G[f, k] += yi * X[o + i, k]
                for k in range(K):
                    xk = p * X[o + i, k]
                    for q in range(k, K): S2[k, q] += xk * X[o + i, q]
            ll[f] = l
            for k in range(K):
                G[f, k] -= npos * mu[k]
                for q in range(k, K): Hc[c, k, q] += npos * (S2[k, q] - mu[k] * mu[q])
    H = Hc.sum(0)
    for k in range(K):
        for q in range(k): H[k, q] = H[q, k]
    return ll, G, H, p_row, MU


def fit_big(D, X, use=None, theta0=None, iters=40, ridge=1e-6):
    K = X.shape[1]; use = np.ones(D["n"], np.bool_) if use is None else use
    th = np.zeros(K) if theta0 is None else np.asarray(theta0, np.float64).copy()
    if theta0 is None: th[0] = 1.0
    r = clogit_big(X, D["y"], D["off"], D["M"], th, use, 24); L = r[0].sum()
    for it in range(iters):
        g = r[1].sum(0) - ridge * th * 0; Hs = r[2] + ridge * np.eye(K) * np.trace(r[2]) / K
        step = np.linalg.solve(Hs, g); t = 1.0
        while True:
            th2 = th + t * step; r2 = clogit_big(X, D["y"], D["off"], D["M"], th2, use, 24); L2 = r2[0].sum()
            if L2 >= L - 1e-9 or t < 1e-4: break
            t *= 0.5
        done = abs(L2 - L) < 1e-8 * max(1.0, abs(L)); th, r, L = th2, r2, L2
        if done: break
    return dict(theta=th, L=L, ll=r[0], G=r[1], H=r[2], p=r[3], MU=r[4])


def control_block(D, exclude=()):
    names, cols = [], []
    for n in POP8 + CTX + PRE2 + INMODEL:
        if n in exclude: continue
        F = forms(cs.load(STORE[n][sp], D), D)
        if F["varshare"] < 0.01: continue
        cand = [(n + ":z", F["z"])] + ([(n + ":rk", F["rk"])] if n in RK else [])
        for nm, c in cand:
            dup = False
            for c2 in cols[-12:]:
                if abs(np.corrcoef(c, c2)[0, 1]) > 0.999: dup = True; break
            if not dup: names.append(nm); cols.append(c)
    return names, cols


class Null:
    """null model = pool + control block; score test for one more column"""
    def __init__(self, D, names, cols):
        self.D = D; self.names = ["pool"] + names
        self.X = np.ascontiguousarray(np.stack([D["base"]] + cols, 1).astype(np.float32))
        f0 = fit(D, []); self.L_pool = f0["L"]
        self.r = fit_big(D, self.X, theta0=[f0["theta"][0]] + [0.0] * len(cols))
        self.fid = np.repeat(np.arange(D["n"]), D["M"]); self.npos = np.bincount(self.fid, D["y"], D["n"])
        self.rw = (self.npos[self.fid] * self.r["p"]).astype(np.float32)           # row weight npos_f * p_i
        self.Hi = np.linalg.inv(self.r["H"] + 1e-6 * np.eye(self.X.shape[1]) * np.trace(self.r["H"]) / self.X.shape[1])
        self.GA = self.r["G"] @ self.Hi                                                # per-feed score of the null model times I_cc^-1
        self.nMU = self.npos[:, None] * self.r["MU"].astype(np.float64)

    def score(self, x):
        D = self.D; x = np.asarray(x, np.float32); p = self.r["p"]
        mx = np.bincount(self.fid, p * x, D["n"])
        g = np.bincount(self.fid, D["y"] * x, D["n"]) - self.npos * mx
        Ixx = float((self.npos * (np.bincount(self.fid, p * x * x, D["n"]) - mx * mx)).sum())
        Ixc = (self.rw * x) @ self.X - mx @ self.nMU
        Uf = g - self.GA @ Ixc; U = float(Uf.sum())
        V = Ixx - float(Ixc @ self.Hi @ Ixc)
        Uc = np.bincount(D["clus"], Uf); Vr = float((Uc ** 2).sum())
        if V <= 1e-9 * max(Ixx, 1e-12) or Vr <= 0:
            return dict(cz=0.0, cz_model=0.0, b1=0.0, dnats=0.0, r2=1.0)
        return dict(cz=U / np.sqrt(Vr), cz_model=U / np.sqrt(V), b1=U / V, dnats=0.5 * U * U / V / D["n"], r2=1.0 - V / max(Ixx, 1e-12))

    def nested_block(self):
        """what re-weighting the ranker's own inputs is worth in NDCG (5 user-grouped folds), against the pool alone"""
        D = self.D; d = np.zeros(D["n"])
        for k in range(5):
            tr = D["fold"] != k; r = fit_big(D, self.X, use=tr, theta0=self.r["theta"], iters=6)
            s = (self.X @ r["theta"].astype(np.float32)).astype(np.float32); e = ndcg(s, D); d[~tr] = e[~tr] - D["ndcg0"][~tr]
        return d


def read_cand(N, x):
    F = forms(x, N.D); out = dict(nanshare=F["nanshare"], varshare=F["varshare"])
    if F["varshare"] < 0.01: out["skip"] = "constant within feeds"; return out
    for k in ("z", "rk", "nan"):
        if k in F:
            for a, b in N.score(F[k]).items(): out[f"{a}_{k}"] = float(b)
    return out


@nb.njit(cache=True)
def perm_within(x, off, M, seed):
    np.random.seed(seed); out = x.copy()
    for f in range(len(M)):
        o = off[f]; m = M[f]
        for i in range(m - 1, 0, -1):
            j = np.random.randint(0, i + 1); t = out[o + i]; out[o + i] = out[o + j]; out[o + j] = t
    return out

