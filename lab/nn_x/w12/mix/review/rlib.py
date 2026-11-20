"""Reviewer library (own code path; imports nothing from build_r4 / rulesA / gateB).
Own loader, own per-feed z (float64), own NDCG (stable sort, negatives first among ties), own user-grouped folds, grid tensor, small nested search.
CPU only; writes only under nn_x/w12/mix/review."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.dont_write_bytecode = True
import numpy as np, numba as nb, polars as pl, scipy.sparse as sps
from pathlib import Path

ROOT = Path('<repo>'); P = ROOT / 'preds'; HERE = ROOT / 'nn_x/w12/mix/review'; CACHE = HERE / 'cache'
CACHE.mkdir(exist_ok=True)
BOUND = 1790497920
WP = np.round(np.arange(0, 0.8001, 0.05), 2); WF = np.round(np.arange(0.2, 0.8001, 0.05), 2); NP_, NF_ = len(WP), len(WF)
ip = lambda w: int(round(w / 0.05)); jf = lambda w: int(round((w - 0.2) / 0.05)); comb = lambda w, c: ip(w) * NF_ + jf(c)
PLAIN = comb(0.2, 0.5); C04 = comb(0.4, 0.5)
S8 = list(range(951, 959)); S4 = list(range(951, 955)); S4b = list(range(955, 959)); ARCH = ('kd05w192', 'kd05t25w192')
MEM = {
    'FD16': [f'w5_fd10_{a}_s{s}' for a in ARCH for s in S8], 'FD8': [f'w5_fd10_{a}_s{s}' for a in ARCH for s in S4], 'FD8b': [f'w5_fd10_{a}_s{s}' for a in ARCH for s in S4b],
    'W16': [f'w5_e10_{a}_s{s}' for a in ARCH for s in S8], 'W8': [f'w5_e10_{a}_s{s}' for a in ARCH for s in S4], 'W8b': [f'w5_e10_{a}_s{s}' for a in ARCH for s in S4b],
    'FDP': [f'abl_fd10_pop1kd05w192_s{s}' for s in S4], 'FDPa': [f'abl_fd10_pop1kd05w192_s{s}' for s in (951, 952)], 'FDPb': [f'abl_fd10_pop1kd05w192_s{s}' for s in (953, 954)],
    'P1w': [f'abl_e10_pop1kd05w192_s{s}' for s in S4], 'P1wa': [f'abl_e10_pop1kd05w192_s{s}' for s in (951, 952)], 'P1wb': [f'abl_e10_pop1kd05w192_s{s}' for s in (953, 954)],
    'FDPd': [f'abl_fd10_pop1kd05t25_s{s}' for s in S4], 'P1d': [f'abl_e10_pop1kd05t25_s{s}' for s in S4],
}
COMPS = {'C1': ('FD16', 'FDP'), 'C2': ('FD8', 'FDP'), 'C3': ('W16', 'P1w'), 'C4': ('W8', 'P1w'),
         'HA': ('FD8', 'FDPa'), 'HB': ('FD8b', 'FDPb'), 'WA': ('W8', 'P1wa'), 'WB': ('W8b', 'P1wb')}
SUF = {'val': ('n2cpv3s3_val_oof', 'trainonly_val'), 'test': ('n2cpv3s3_test', 'trainonly_test')}
_META = {}


def meta(sp):
    if sp not in _META:
        f = pl.read_parquet(ROOT / f'data/flat/{sp}/feeds.parquet')
        M = f['M'].to_numpy().astype(np.int64); off = f['cand_off'].to_numpy().astype(np.int64)
        assert (off == np.concatenate([[0], np.cumsum(M)[:-1]])).all()
        _META[sp] = dict(M=M, off=off, uid=f['user_id'].to_numpy().astype(np.int64), H=f['H'].to_numpy().astype(np.int64), ts=f['timestamp'].to_numpy().astype(np.int64), n=len(M), rows=int(M.sum()))
    return _META[sp]


def labels():
    y = np.load(ROOT / 'data/flat/val/cand_target.npy'); assert len(y) == meta('val')['rows']; return np.asarray(y, np.float64)


@nb.njit(parallel=True, cache=False)
def zfeed(x, off, M):
    out = np.empty(len(x), np.float64)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]; mu = 0.0
        for k in range(m): mu += x[o + k]
        mu /= m; s = 0.0
        for k in range(m): s += (x[o + k] - mu) ** 2
        sd = np.sqrt(s / m) + 1e-9
        for k in range(m): out[o + k] = (x[o + k] - mu) / sd
    return out


def Z(sp, x):
    m = meta(sp); return zfeed(np.asarray(x, np.float64), m['off'], m['M'])


def pool(names, sp):
    """member-equal mean of per-feed z -> (FT pool, TO pool), float64."""
    ft = None; to = None
    for nme in names:
        a = Z(sp, np.load(P / f'nn_{nme}_{SUF[sp][0]}.npy')); b = Z(sp, np.load(P / f'nn_{nme}_{SUF[sp][1]}.npy'))
        ft = a if ft is None else ft + a; to = b if to is None else to + b
    return ft / len(names), to / len(names)


def pools4(key, sp='val'):
    fn = CACHE / f'pools_{key}_{sp}.npy'
    if fn.exists(): return np.load(fn)
    b, p = COMPS[key]; bF, bT = pool(MEM[b], sp); pF, pT = pool(MEM[p], sp)
    q = np.stack([Z(sp, bF), Z(sp, bT), Z(sp, pF), Z(sp, pT)]); np.save(fn, q); return q


def final(sp, q, w_feed, c_feed=0.5):
    """build_r4 form with per-feed (or scalar) weights; float64."""
    m = meta(sp); r = lambda v: np.repeat(np.asarray(v, np.float64), m['M']) if np.ndim(v) else float(v)
    w = r(w_feed); c = r(c_feed)
    return c * Z(sp, (1 - w) * q[0] + w * q[2]) + (1 - c) * Z(sp, (1 - w) * q[1] + w * q[3])


@nb.njit(inline='always')
def _ndcg_one(s, y, m, idx0, tmp):
    # idx0: feed-local indices with negatives first; stable sort of -s over that order => among ties negatives stay first (pessimistic)
    npos = 0
    for k in range(m):
        if y[k] > 0: npos += 1
    if npos == 0: return 1.0
    for k in range(m): tmp[k] = -s[idx0[k]]
    order = np.argsort(tmp[:m], kind='mergesort')
    dcg = 0.0
    for r in range(m):
        if y[idx0[order[r]]] > 0: dcg += 1.0 / np.log2(r + 2.0)
    idcg = 0.0
    for r in range(npos): idcg += 1.0 / np.log2(r + 2.0)
    return dcg / idcg


@nb.njit(parallel=True, cache=False)
def ndcg_feed(s, y, off, M):
    n = len(M); out = np.ones(n)
    for f in nb.prange(n):
        o = off[f]; m = M[f]; idx0 = np.empty(m, np.int64); j = 0
        for k in range(m):
            if y[o + k] <= 0: idx0[j] = k; j += 1
        for k in range(m):
            if y[o + k] > 0: idx0[j] = k; j += 1
        tmp = np.empty(m)
        out[f] = _ndcg_one(s[o:o + m], y[o:o + m], m, idx0, tmp)
    return out


@nb.njit(parallel=True, cache=False)
def tensor_k(q, y, off, M, WP, WF):
    """E[f, i, j]: NDCG of feed f at (w_pop = WP[i] in both branches, w_ft = WF[j]). q = 4 standardised pool arrays (bF, bT, pF, pT)."""
    n = len(M); nP = len(WP); nF = len(WF); out = np.ones((n, nP * nF))
    for f in nb.prange(n):
        o = off[f]; m = M[f]; npos = 0
        for k in range(m):
            if y[o + k] > 0: npos += 1
        if npos == 0: continue
        idx0 = np.empty(m, np.int64); j = 0
        for k in range(m):
            if y[o + k] <= 0: idx0[j] = k; j += 1
        for k in range(m):
            if y[o + k] > 0: idx0[j] = k; j += 1
        yy = y[o:o + m]; tmp = np.empty(m); s = np.empty(m); brF = np.empty(m); brT = np.empty(m)
        for i in range(nP):
            w = WP[i]
            for br in range(2):
                mu = 0.0
                for k in range(m):
                    v = (1 - w) * q[br, o + k] + w * q[2 + br, o + k]; mu += v
                    if br == 0: brF[k] = v
                    else: brT[k] = v
                mu /= m; sd = 0.0
                for k in range(m):
                    v = brF[k] if br == 0 else brT[k]; sd += (v - mu) ** 2
                sd = np.sqrt(sd / m) + 1e-9
                for k in range(m):
                    if br == 0: brF[k] = (brF[k] - mu) / sd
                    else: brT[k] = (brT[k] - mu) / sd
            for jj in range(nF):
                c = WF[jj]
                for k in range(m): s[k] = c * brF[k] + (1 - c) * brT[k]
                out[f, i * nF + jj] = _ndcg_one(s, yy, m, idx0, tmp)
    return out


def tensor(key, y=None):
    fn = CACHE / f'E_{key}.npy'
    if fn.exists(): return np.load(fn)
    m = meta('val'); y = labels() if y is None else y
    E = tensor_k(pools4(key), y, m['off'], m['M'], WP, WF); np.save(fn, E); return E


# ---------------------------------------------------------------- groups, folds, SE
def groups():
    u = meta('val')['uid'].copy(); z = u == 0; u[z] = -1 - np.arange(z.sum())
    return np.unique(u, return_inverse=True)[1]


def folds(g, seed, K=5):
    ng = g.max() + 1; rng = np.random.default_rng(seed); perm = rng.permutation(ng); fo = np.empty(ng, np.int64); fo[perm] = np.arange(ng) % K
    return fo[g]


def cse(d, g):
    d = np.asarray(d, np.float64); mu = d.mean(); s = np.bincount(g, d - mu); return mu, np.sqrt((s ** 2).sum()) / len(d)


def pse(d):
    d = np.asarray(d, np.float64); return d.mean(), d.std() / np.sqrt(len(d))


def fm(d, g):
    a, b = cse(d, g); return f'{a:+.5f}+-{b:.5f}'


def wse(d, w):
    m = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum(); return m, np.sqrt(np.average((d - m) ** 2, weights=w) / neff)


# ---------------------------------------------------------------- label-free feed properties
def cold_pmix(sp):
    t = pl.read_parquet(ROOT / ('eda/errors/segments/feed_table.parquet' if sp == 'val' else 'eda/errors/segments/test_segments.parquet'))
    m = meta(sp); a = np.full(m['n'], np.nan); a[t['fidx'].to_numpy()] = t['cold_share'].to_numpy(); assert not np.isnan(a).any() and (t['M'].to_numpy() == m['M'][t['fidx'].to_numpy()]).all()
    return a


def cold_luo(sp):
    m = meta(sp); x = np.load(ROOT / f'nn/feat_cache/{sp}/it_hist_all_luo.npy', mmap_mode='r'); assert len(x) == m['rows']
    return np.bincount(np.repeat(np.arange(m['n']), m['M']), (np.asarray(x) == 0).astype(np.float64), m['n']) / m['M']


# ---------------------------------------------------------------- families on the grid tensor
AP = [ip(w) for w in (0, .1, .2, .3, .4, .5, .6)]; AC = [jf(c) for c in (.3, .4, .5, .6, .7, .8)]; J5 = jf(0.5)


def fam_defs(pr, hq=None):
    """list of (name, [sub-families]); a sub-family = (cell id per feed, ncell, R[n_rules, ncell] of combo indices). pr = dict of label-free props (M, H). hq = H quantile cuts for HSTUMP."""
    M, H = pr['M'], pr['H']; one = np.zeros(len(M), np.int64); g = lambda *a: [x.ravel() for x in np.meshgrid(*a, indexing='ij')]
    c2 = lambda i, j: np.asarray(i) * NF_ + np.asarray(j)
    F = []
    a, c = g(AP, AC); F.append(('A', [(one, 1, c2(a, c)[:, None])]))
    a0, a1 = g(AP, AP); RC = np.stack([c2(a0, J5), c2(a1, J5)], 1)
    F.append(('C(M50)', [((M >= 50).astype(np.int64), 2, RC)])); F.append(('C(M70)', [((M >= 70).astype(np.int64), 2, RC)]))
    c0, c1, a = g(AC, AC, AP); F.append(('F(Hq)', [((H >= 190).astype(np.int64), 2, np.stack([c2(a, c0), c2(a, c1)], 1))]))
    a0, a1, a2 = g(AP, AP, AP); F.append(('G(M3q)', [((M >= 39).astype(np.int64) + (M >= 117), 3, np.stack([c2(a0, J5), c2(a1, J5), c2(a2, J5)], 1))]))
    q = g(AP, AP, AP, AP); F.append(('K(M70,Hq)', [(2 * (M >= 70).astype(np.int64) + (H >= 190), 4, np.stack([c2(x, J5) for x in q], 1))]))
    # B's grid
    allp = np.arange(NP_); allf = np.arange(NF_)
    F.append(('CONSTp', [(one, 1, c2(allp, J5)[:, None])]))
    a, c = g(allp, allf); F.append(('CONST2', [(one, 1, c2(a, c)[:, None])]))
    a0, a1 = g(allp, allp); RB = np.stack([c2(a0, J5), c2(a1, J5)], 1)
    F.append(('MSTEP50', [((M >= 50).astype(np.int64), 2, RB)]))
    F.append(('MSTEP(25,50,100)', [((M >= cc).astype(np.int64), 2, RB) for cc in (25, 50, 100)]))
    F.append(('MSTEPc(20..150)', [((M >= cc).astype(np.int64), 2, RB) for cc in (20, 30, 50, 70, 100, 150)]))
    if hq is not None:
        F.append(('HSTUMP', [((H > hc).astype(np.int64), 2, None) for hc in hq]))      # R None = every combo free in every cell (separable)
    return F


def block_sums(D, fold, cell, ncell, K=5):
    n = len(fold); gidx = fold * ncell + cell
    Mx = sps.csr_matrix((np.ones(n), (gidx, np.arange(n))), shape=(K * ncell, n))
    return np.asarray(Mx @ D).reshape(K, ncell, D.shape[1])


def best_rule(T, R):
    """T[ncell, comb] training sums -> (best training sum, combo index per cell)."""
    nc = T.shape[0]
    if R is None:
        r = T.argmax(1); return T[np.arange(nc), r].sum(), r
    sc = T[np.arange(nc)[None, :], R].sum(1); i = int(sc.argmax()); return sc[i], R[i]


def nested(D, fams, fold_list, K=5, want_vec=True, want_pick=False):
    """D[n, comb] = contrast to PLAIN (or any per-feed table). Returns per family: in-sample best mean, nested mean per seed, per-feed nested vector (mean over seeds), picks."""
    n = D.shape[0]; out = {}
    cache = {}
    for name, subs in fams:
        res = dict(ins=None, seed=[], vec=np.zeros(n) if want_vec else None, picks=[])
        # in-sample
        bi = (-np.inf, None, None)
        for si, (cell, nc, R) in enumerate(subs):
            T = block_sums(D, np.zeros(n, np.int64), cell, nc, 1)[0]; sc, r = best_rule(T, R)
            if sc > bi[0] + 1e-12: bi = (sc, si, r)
        res['ins'] = bi[0] / n; res['ins_rule'] = (bi[1], bi[2])
        for fi, fold in enumerate(fold_list):
            S = []
            for si, (cell, nc, R) in enumerate(subs):
                key = (fi, id(cell))
                if key not in cache: cache[key] = block_sums(D, fold, cell, nc, K)
                S.append(cache[key])
            tot = 0.0
            for k in range(K):
                bk = (-np.inf, None, None)
                for si, (cell, nc, R) in enumerate(subs):
                    sc, r = best_rule(S[si].sum(0) - S[si][k], R)
                    if sc > bk[0] + 1e-12: bk = (sc, si, r)
                si, r = bk[1], bk[2]; cell, nc, R = subs[si]
                tot += S[si][k][np.arange(nc), r].sum()
                if want_pick: res['picks'].append((si, tuple(int(x) for x in r)))
                if want_vec:
                    mk = fold == k; res['vec'][mk] += D[mk][np.arange(mk.sum()), r[cell[mk]]] / len(fold_list)
            res['seed'].append(tot / n)
        res['nested'] = float(np.mean(res['seed'])); out[name] = res
    return out


def dec(c):
    c = int(c); return (float(WP[c // NF_]), float(WF[c % NF_]))
