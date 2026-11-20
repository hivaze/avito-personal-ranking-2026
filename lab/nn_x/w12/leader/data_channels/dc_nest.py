"""Nested read of a channel on top of the best file's validation score.

clogit: per-feed conditional logit (listwise softmax, multiple positives) s' = b*s + sum_j w_j x_j, fitted on one half of
the users and scored on the other half (both directions), for several random user halvings (user-0 feeds are singleton
groups). The control is the same fit with x empty (b*s alone, which leaves the order of s unchanged). Reported:
  dNDCG   = mean over feeds of [NDCG(arm) - NDCG(control)] averaged over halvings, SE = paired per-feed SE;
  dnats   = held-out log-likelihood gain per click, sum over feeds / number of positives (SE from per-feed values);
  pmix    = the same NDCG delta re-weighted to the public / private mix (lab/src/pmix.py);
  insample= NDCG gain of a fit on all feeds (ceiling, not a result).
lgbr: LightGBM lambdarank residual that starts from b*s (init_score), same halvings, fixed small trees; dNDCG and dnats
      against the control b*s (information only; the clogit is the main read).
"""
import numpy as np
import numba as nb
from scipy.optimize import minimize
from dc_lib import ndcg_per_feed, user_halves, wmean_se, pmix_weights


@nb.njit(cache=True)
def _feed_ll_grad(beta, X, y, o, m, g):
    k = X.shape[1]
    s = np.empty(m)
    mx = -1e300
    for i in range(m):
        v = 0.0
        for j in range(k):
            v += X[o + i, j] * beta[j]
        s[i] = v
        if v > mx:
            mx = v
    z = 0.0
    for i in range(m):
        s[i] = np.exp(s[i] - mx)
        z += s[i]
    npos = 0.0
    for i in range(m):
        if y[o + i] > 0:
            npos += 1.0
    ll = 0.0
    for i in range(m):
        p = s[i] / z
        c = npos * p - y[o + i]
        for j in range(k):
            g[j] += c * X[o + i, j]
        if y[o + i] > 0:
            ll += np.log(p)
    return ll


@nb.njit(parallel=True, cache=True)
def _ll_grad(beta, X, y, off, M, feeds, lam_mask, lam):
    k = X.shape[1]
    nf = len(feeds)
    G = np.zeros((nf, k))
    L = np.zeros(nf)
    for q in nb.prange(nf):
        f = feeds[q]
        L[q] = _feed_ll_grad(beta, X, y, off[f], M[f], G[q])
    loss = -L.sum() + 0.5 * lam * np.sum(lam_mask * beta * beta)
    g = G.sum(axis=0) + lam * lam_mask * beta
    return loss, g


@nb.njit(cache=True)
def _one_ll(score, y, o, m):
    mx = -1e300
    for i in range(m):
        if score[o + i] > mx:
            mx = score[o + i]
    z = 0.0
    for i in range(m):
        z += np.exp(score[o + i] - mx)
    lse = mx + np.log(z)
    ll = 0.0
    for i in range(m):
        if y[o + i] > 0:
            ll += score[o + i] - lse
    return ll


@nb.njit(parallel=True, cache=True)
def _ll_feed(score, y, off, M):
    n = len(M)
    out = np.zeros(n)
    for f in nb.prange(n):
        out[f] = _one_ll(score, y, off[f], M[f])
    return out


def fit_clogit(X, y, off, M, feeds, lam=1.0):
    k = X.shape[1]
    mask = np.ones(k); mask[0] = 0.0                    # base column not penalised
    b0 = np.zeros(k); b0[0] = 1.0
    fun = lambda b: _ll_grad(b, X, y, off, M, feeds, mask, lam)
    r = minimize(fun, b0, jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return r.x


def prep_columns(cols, off, M):
    """cols: list of 1-D arrays (raw features). NaN -> 0. Each scaled by its global sd (label-free)."""
    out = []
    for c in cols:
        c = np.nan_to_num(np.asarray(c, np.float64))
        sd = c.std()
        out.append(c / sd if sd > 0 else c)
    return np.stack(out, 1) if out else np.zeros((len(cols), 0))


class Nested:
    def __init__(self, sp, base, seeds=(0, 1, 2, 3, 4), lam=1.0):
        self.sp = sp; self.s = np.asarray(base, np.float64); self.seeds = seeds; self.lam = lam
        self.y = sp.y; self.off = sp.off; self.M = sp.M
        self.npos = np.bincount(sp.fid, weights=sp.y, minlength=sp.n)
        self.halves = [user_halves(sp.user, sd) for sd in seeds]
        self.wpub, self.wpriv = pmix_weights()
        self._ctrl = None

    ALPHAS = (0.0, 0.03, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)

    def _run(self, X):
        """Held-out per-feed NDCG (blend scale chosen by NDCG on the training half) and held-out LL of the pure clogit.
        X is one matrix, or a callable X(k) giving the (cross-fitted) matrix of halving k."""
        P = np.zeros(self.sp.n); L = np.zeros(self.sp.n); betas = []; alphas = []
        for kh, h in enumerate(self.halves):
            Xk = X(kh) if callable(X) else X
            Xf = np.concatenate([self.s[:, None], Xk], 1) if Xk.shape[1] else self.s[:, None]
            for a in (True, False):
                tr = np.flatnonzero(h == a); te = h != a
                b = fit_clogit(Xf, self.y, self.off, self.M, tr, self.lam)
                betas.append(b)
                sc = Xf @ b
                ll = _ll_feed(sc, self.y, self.off, self.M)
                L[te] += ll[te]
                if Xk.shape[1]:
                    comp = (Xk @ b[1:]) / b[0]
                    nds = [ndcg_per_feed(self.s + al * comp, self.y, self.off, self.M) for al in self.ALPHAS]
                    k = int(np.argmax([n[tr].mean() for n in nds])); alphas.append(self.ALPHAS[k])
                    P[te] += nds[k][te]
                else:
                    P[te] += ndcg_per_feed(self.s, self.y, self.off, self.M)[te]
        P /= len(self.halves); L /= len(self.halves)
        self._alphas = alphas
        return P, L, np.array(betas)

    def control(self):
        if self._ctrl is None:
            self._ctrl = self._run(np.zeros((self.sp.n, 0)))
        return self._ctrl

    def read(self, name, cols, insample=True, mask_feeds=None):
        """cols: list of columns, or a callable cols(k) -> list of columns for halving k (cross-fitted features)."""
        if callable(cols):
            cache = {}
            def X(k):
                if k not in cache:
                    cache.clear(); cache[k] = prep_columns(cols(k), self.off, self.M)
                return cache[k]
            insample = False
            kdim = X(0).shape[1]
        else:
            X = prep_columns(cols, self.off, self.M); kdim = X.shape[1]
        P0, L0, _ = self.control()
        P1, L1, B = self._run(X)
        d = P1 - P0; dl = L1 - L0
        m, se = wmean_se(d)
        mp, sep = wmean_se(d, self.wpub); mq, seq = wmean_se(d, self.wpriv)
        tot = self.npos.sum()
        nats = dl.sum() / tot; nats_se = dl.std() * np.sqrt(len(dl)) / tot
        res = dict(name=name, k=kdim, dndcg=m, se=se, z=m / se if se > 0 else 0.0, pub=mp, pub_se=sep, priv=mq, priv_se=seq,
                   dnats=nats, dnats_se=nats_se, beta_mean=B.mean(0).round(4).tolist(), alphas=list(self._alphas))
        if insample:
            Xf = np.concatenate([self.s[:, None], X], 1)
            b = fit_clogit(Xf, self.y, self.off, self.M, np.arange(self.sp.n), self.lam)
            comp = (X @ b[1:]) / b[0]
            p0 = ndcg_per_feed(self.s, self.y, self.off, self.M)
            res["insample"] = float(max((ndcg_per_feed(self.s + al * comp, self.y, self.off, self.M) - p0).mean() for al in self.ALPHAS))
        if mask_feeds is not None:
            mm = mask_feeds
            res["sub_n"] = int(mm.sum()); res["sub_dndcg"] = float(d[mm].mean()); res["sub_se"] = float(d[mm].std() / np.sqrt(mm.sum()))
            res["sub_dnats"] = float(dl[mm].sum() / self.npos[mm].sum())
        self.last = (d, dl)
        return res


def fmt(r):
    s = (f"{r['name']:34s} k={r['k']:2d} dNDCG {r['dndcg']:+.5f} ± {r['se']:.5f} (z {r['z']:+.2f}) | pub {r['pub']:+.5f} ± {r['pub_se']:.5f}"
         f" priv {r['priv']:+.5f} ± {r['priv_se']:.5f} | dnats/click {r['dnats']:+.5f} ± {r['dnats_se']:.5f}")
    if "insample" in r:
        s += f" | in-sample {r['insample']:+.5f}"
    if "sub_n" in r:
        s += f" | covered feeds n={r['sub_n']} {r['sub_dndcg']:+.4f} ± {r['sub_se']:.4f} nats {r['sub_dnats']:+.4f}"
    if "alphas" in r:
        s += f" | alpha {np.mean(r['alphas']):.2f}"
    return s


def _perm_within_feed(x, off, M, rng):
    o = np.lexsort((rng.random(len(x)), np.repeat(np.arange(len(M)), M)))
    return x[o]


def lgbr(nest, cols, name, rounds=150, leaves=15, lr=0.05, seeds=(0, 1, 2)):
    """LightGBM lambdarank residual from b*s on [s, X] against the same fit on [s, X permuted within each feed]
    (placebo with the same column distributions per feed). Held-out halves; information only."""
    import lightgbm as lgb
    sp = nest.sp
    Xr = [np.nan_to_num(np.asarray(c, np.float64)) for c in cols]
    rng = np.random.default_rng(123)
    perm = _perm_within_feed(np.arange(sp.N), sp.off, sp.M, rng)
    XA = np.stack([nest.s] + Xr, 1); XP = np.stack([nest.s] + [c[perm] for c in Xr], 1)
    out = {}
    for tag, X in (("real", XA), ("placebo", XP)):
        P = np.zeros(sp.n); L = np.zeros(sp.n)
        for si, sd in enumerate(seeds):
            h = nest.halves[si]
            for a in (True, False):
                trf = np.flatnonzero(h == a)
                b = fit_clogit(nest.s[:, None], sp.y, sp.off, sp.M, trf, 1.0)[0]
                rows_tr = np.isin(sp.fid, trf)
                ds = lgb.Dataset(X[rows_tr], label=sp.y[rows_tr], group=sp.M[h == a], init_score=b * nest.s[rows_tr], free_raw_data=False)
                prm = dict(objective="lambdarank", num_leaves=leaves, learning_rate=lr, min_data_in_leaf=200, lambda_l2=10.0,
                           feature_fraction=1.0, verbose=-1, num_threads=12, deterministic=True, seed=sd,
                           lambdarank_truncation_level=300, label_gain=[0, 1])
                bst = lgb.train(prm, ds, num_boost_round=rounds)
                sc = b * nest.s + bst.predict(X, raw_score=True)
                te = h != a
                P[te] += ndcg_per_feed(sc, sp.y, sp.off, sp.M)[te]; L[te] += _ll_feed(sc, sp.y, sp.off, sp.M)[te]
        out[tag] = (P / len(seeds), L / len(seeds))
    d = out["real"][0] - out["placebo"][0]; dl = out["real"][1] - out["placebo"][1]
    m, se = wmean_se(d); tot = nest.npos.sum()
    base = ndcg_per_feed(nest.s, sp.y, sp.off, sp.M)
    return dict(name="lgbr:" + name, dndcg_vs_placebo=m, se=se, dnats_vs_placebo=dl.sum() / tot,
                dnats_se=dl.std() * np.sqrt(len(dl)) / tot, real_vs_base=float((out["real"][0] - base).mean()),
                placebo_vs_base=float((out["placebo"][0] - base).mean()))
