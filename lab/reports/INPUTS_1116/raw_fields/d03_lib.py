"""INVENTORY D step 3 library: conditional-logit likelihood-ratio lens with a within-feed-shuffle placebo (CPU only, read-only).

Model per feed:  P(positive p | feed) = softmax_j(u_j)[p],  u = C @ a (+ beta * x);  every positive of a feed contributes one term.
Control C (4 columns): s, s^2, s^3, s*log(M/50)  with s = within-feed z of the pool score (so the score enters flexibly).
LR(x) = 2 (ll[C, x] - ll[C]), 1 dof.  Placebo = the same with x shuffled within feed (seeded).  Rule (written before any run,
d03_rule.txt): "interesting" <=> LR >= 10.83 and placebo LR < 3.84.
Secondary read (scale only): 5-fold user-grouped cross-fitted NDCG delta of adding beta*x to the control utility.
Val control = preds/ens_FD16P20kd_to_val.npy (train-only branch of the pick-1 pool; no val label inside), second control
= final FD16P20 score 0.5 z(FT oof) + 0.5 z(TO).  Holdout control = 0.8 z(mean z of the 16 held-out wide students) + 0.2 z(mean z of
the 8 held-out pop1 students) on the 30,000 holdout feeds (W16P20x8 composition, nn_x/w12/kd/build_r4.py; loaders as in
nn_x/w12/recheck/pool_composition/pc_common.py)."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
sys.path.insert(0, '<repo>/src')
import numpy as np, numba as nb, polars as pl
from pathlib import Path
R = Path('<repo>'); F = R / 'data/flat'; PR = R / 'preds'; OUT = R / 'reports/INPUTS_1116/raw_fields'
W = R / 'nn_x/w5/preds'; AB = R / 'nn_x/abl/preds'


@nb.njit(parallel=True, cache=True)
def _z(off, M, x):
    out = np.empty(len(x), np.float64)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]; v = x[o:o + m].astype(np.float64); mu = v.mean(); sd = v.std() + 1e-9
        out[o:o + m] = (v - mu) / sd
    return out


@nb.njit(parallel=True, cache=True)
def _ll_grad_hess(off, M, y, X, th):
    n = len(M); k = X.shape[1]
    ll = np.zeros(n); G = np.zeros((n, k)); Hs = np.zeros((n, k, k))
    for f in nb.prange(n):
        o, m = off[f], M[f]
        npos = 0.0
        u = np.zeros(m)
        for i in range(m):
            for c in range(k): u[i] += X[o + i, c] * th[c]
            npos += y[o + i]
        if npos == 0: continue
        mx = u.max(); p = np.exp(u - mx); Z = p.sum(); p /= Z; lse = mx + np.log(Z)
        xb = np.zeros(k)
        for i in range(m):
            for c in range(k): xb[c] += p[i] * X[o + i, c]
        s = 0.0
        for i in range(m):
            if y[o + i] > 0:
                s += u[i] - lse
                for c in range(k): G[f, c] += X[o + i, c] - xb[c]
        ll[f] = s
        for i in range(m):
            for c in range(k):
                dc = X[o + i, c] - xb[c]
                for d in range(c, k):
                    Hs[f, c, d] += npos * p[i] * dc * (X[o + i, d] - xb[d])
    return ll, G, Hs


def fit(off, M, y, X, feeds=None, th0=None, iters=30):
    """Newton fit of the conditional logit; returns (theta, loglik). feeds = optional boolean mask of feeds used."""
    k = X.shape[1]; th = np.zeros(k) if th0 is None else th0.copy(); last = -np.inf
    for it in range(iters):
        ll, G, Hs = _ll_grad_hess(off, M, y, X, th)
        if feeds is not None: ll, G, Hs = ll[feeds], G[feeds], Hs[feeds]
        L = ll.sum(); g = G.sum(0); H = Hs.sum(0); H = H + np.triu(H, 1).T + 1e-8 * np.eye(k)
        if abs(L - last) < 1e-7 * max(1.0, abs(L)) and it > 2: break
        last = L
        step = np.linalg.solve(H, g)
        # damped step
        t = 1.0
        while t > 1e-4:
            ll2, _, _ = _ll_grad_hess(off, M, y, X, th + t * step)
            L2 = ll2[feeds].sum() if feeds is not None else ll2.sum()
            if L2 >= L - 1e-9: break
            t /= 2
        th = th + t * step
    ll, _, _ = _ll_grad_hess(off, M, y, X, th)
    return th, (ll[feeds].sum() if feeds is not None else ll.sum())


@nb.njit(parallel=True, cache=True)
def _shuffle_within(off, M, x, seed):
    out = np.empty_like(x)
    for f in nb.prange(len(M)):
        np.random.seed(seed + f)
        o, m = off[f], M[f]
        out[o:o + m] = np.random.permutation(x[o:o + m])
    return out


@nb.njit(parallel=True, cache=True)
def ndcg_pf(s, y, off, M):
    n = len(M); out = np.ones(n, np.float64)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        sv = s[o:o + m]; yv = y[o:o + m]
        npos = 0
        for k in range(m):
            if yv[k] > 0: npos += 1
        if npos == 0: continue
        order = np.argsort(-sv, kind='mergesort')
        dcg = 0.0; i = 0
        while i < m:
            j = i
            while j + 1 < m and sv[order[j + 1]] == sv[order[i]]: j += 1
            if j == i:
                if yv[order[i]] > 0: dcg += 1.0 / np.log2(i + 2.0)
            else:
                cp = 0
                for k in range(i, j + 1):
                    if yv[order[k]] > 0: cp += 1
                for k in range(j - cp + 1, j + 1): dcg += 1.0 / np.log2(k + 2.0)
            i = j + 1
        idcg = 0.0
        for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


class Split:
    """off, M, y, user (per feed), ctrl dict name -> control matrix C [n,4], score dict."""
    def controls(self, s):
        z = _z(self.off, self.M, np.asarray(s, np.float64)); lm = np.log(np.repeat(self.M, self.M) / 50.0)
        return np.stack([z, z * z, z ** 3, z * lm], 1)


def load_val():
    S = Split(); f = pl.read_parquet(F / 'val/feeds.parquet'); S.feeds = f
    S.M = f['M'].to_numpy().astype(np.int64); S.off = f['cand_off'].to_numpy().astype(np.int64); S.user = f['user_id'].to_numpy()
    S.y = np.load(F / 'val/cand_target.npy').astype(np.float64); S.fi = np.repeat(np.arange(len(S.M)), S.M)
    to = np.load(PR / 'ens_FD16P20kd_to_val.npy'); ft = np.load(PR / 'ens_FD16P20kd_ft_val.npy')
    fin = 0.5 * _z(S.off, S.M, ft.astype(np.float64)) + 0.5 * _z(S.off, S.M, to.astype(np.float64))
    S.score = {'TO': to.astype(np.float64), 'FIN': fin}; S.ctrl = {k: S.controls(v) for k, v in S.score.items()}
    S.name = 'val'
    return S


def holdout_feeds():
    """Feed indices (into train) of the 30,000 holdout feeds: same draw as nn_x/w5/train.py --holdout 30000."""
    f = pl.read_parquet(F / 'train/feeds.parquet'); M = f['M'].to_numpy().astype(np.int64)
    pr = np.load(R / 'eda/labels_position/out/train_prandom.npz'); assert (pr['fidx'] == np.arange(len(M))).all()
    p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    return np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False)), f


def load_hold():
    S = Split(); hold, f = holdout_feeds(); S.feeds = f[hold]; S.hold = hold
    Mt = f['M'].to_numpy().astype(np.int64); offt = f['cand_off'].to_numpy().astype(np.int64)
    S.M = Mt[hold]; S.off = np.r_[0, np.cumsum(S.M)[:-1]].astype(np.int64); S.user = f['user_id'].to_numpy()[hold]
    S.rows = np.concatenate([np.arange(o, o + m) for o, m in zip(offt[hold], Mt[hold])])   # rows into train cand arrays
    S.y = np.load(W / 'holdout/holdout_y.npy').astype(np.float64)
    Mh, oh = np.load(W / 'holdout/holdout_meta.npy'); assert (Mh == S.M).all() and (oh == S.off).all()
    yt = np.load(F / 'train/cand_target.npy', mmap_mode='r'); assert (np.asarray(yt[S.rows]) == S.y).all()
    S.fi = np.repeat(np.arange(len(S.M)), S.M)
    wide = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(951, 959)]
    pop = [f'abl_e10_pop1kd05t25_s{s}' for s in range(951, 955)] + [f'abl_e10_pop1kd05w192_s{s}' for s in range(951, 955)]
    zw = np.mean([_z(S.off, S.M, np.load(W / f'holdout/nn_{v}_holdout.npy').astype(np.float64)) for v in wide], 0)
    zp = np.mean([_z(S.off, S.M, np.load(AB / f'holdout/nn_{v}_holdout.npy').astype(np.float64)) for v in pop], 0)
    sc = 0.8 * _z(S.off, S.M, zw) + 0.2 * _z(S.off, S.M, zp)
    S.score = {'H': sc}; S.ctrl = {'H': S.controls(sc)}; S.name = 'hold'
    return S


def lr_test(S, x, ctrl, n_placebo=3, cv=True):
    """Returns dict: LR, beta, sd-within-feed of x, placebo LRs, cross-fitted NDCG delta (mean, se)."""
    C = S.ctrl[ctrl]; x = np.nan_to_num(np.asarray(x, np.float64))
    th0, ll0 = S._base[ctrl] if hasattr(S, '_base') and ctrl in S._base else fit(S.off, S.M, S.y, C)
    if not hasattr(S, '_base'): S._base = {}
    S._base[ctrl] = (th0, ll0)
    X = np.column_stack([C, x]); th, ll1 = fit(S.off, S.M, S.y, X, th0=np.r_[th0, 0.0])
    xc = x - np.repeat(np.add.reduceat(x, S.off) / S.M, S.M); sdw = float(np.sqrt((xc ** 2).mean()))
    pl_ = []
    for k in range(n_placebo):
        xs = _shuffle_within(S.off, S.M, x, 1000 * (k + 1))
        _, llp = fit(S.off, S.M, S.y, np.column_stack([C, xs]), th0=np.r_[th0, 0.0]); pl_.append(2 * (llp - ll0))
    out = dict(LR=2 * (ll1 - ll0), beta=float(th[-1]), sd_within=sdw, effect=float(th[-1]) * sdw, placebo=pl_)
    if cv:
        # 5 user-grouped folds (user 0 feeds spread by feed index)
        u = S.user.astype(np.int64).copy(); z0 = u == 0; u[z0] = -(np.arange(len(u))[z0] + 1)
        fold = (np.abs(u * 2654435761) % 1000003) % 5
        base = ndcg_pf(C @ th0, S.y, S.off, S.M); sc = np.empty(len(x))
        scb = np.empty(len(x))
        for k in range(5):
            trm = fold != k
            tb, _ = fit(S.off, S.M, S.y, C, feeds=trm, th0=th0)
            tk, _ = fit(S.off, S.M, S.y, X, feeds=trm, th0=th)
            te_rows = np.repeat(~trm, S.M)
            sc[te_rows] = (X @ tk)[te_rows]; scb[te_rows] = (C @ tb)[te_rows]
        d = ndcg_pf(sc, S.y, S.off, S.M) - ndcg_pf(scb, S.y, S.off, S.M)
        out['cv_dndcg'] = float(d.mean()); out['cv_se'] = float(d.std() / np.sqrt(len(d)))
        raw = ndcg_pf(S.score[ctrl], S.y, S.off, S.M); out['ndcg_raw'] = float(raw.mean()); out['ndcg_ctrl_insample'] = float(base.mean())
    return out


def fmt(name, sp, ctrl, r, cover=None):
    pmax = max(r['placebo'])
    flag = 'INTERESTING' if (r['LR'] >= 10.83 and r['placebo'][0] < 3.84) else ('lr-pass/placebo-fail' if r['LR'] >= 10.83 else 'null')
    s = (f'{name:34s} {sp:5s} ctrl {ctrl:3s} LR {r["LR"]:8.2f} beta {r["beta"]:+.4f} effect(beta*sd_w) {r["effect"]:+.4f} placebo {r["placebo"][0]:5.2f} '
         f'(max of {len(r["placebo"])}: {pmax:5.2f})')
    if 'cv_dndcg' in r: s += f' | cross-fit dNDCG {r["cv_dndcg"]:+.5f}+-{r["cv_se"]:.5f}'
    if cover is not None: s += f' | coverage {cover:.4f}'
    return s + f' | {flag}'
