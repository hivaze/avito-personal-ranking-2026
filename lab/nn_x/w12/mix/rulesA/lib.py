"""rulesA library: own loader (member files -> pools), numba kernel for the per-feed NDCG table over the weight grid, gates, rule families, nested search.
CPU only. Never writes outside nn_x/w12/mix/rulesA. See DECLARATION.txt for every definition used here."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/src')
import numpy as np, numba as nb, polars as pl, scipy.sparse as sps
from pathlib import Path
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path('<repo>'); P = ROOT / 'preds'; HERE = ROOT / 'nn_x/w12/mix/rulesA'
BOUND = 1790497920
AG = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]); CG = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]); NA, NC = len(AG), len(CG); NCOMB = NA * NA * NC
A02, A04, C05 = 2, 4, 2          # grid indices of the plain / M-rule values
S8 = range(951, 959); S4 = range(951, 955)
MEMBERS = {
    'FD16': [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8],
    'FD8': [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S4],
    'W16': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8],
    'W8': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S4],
    'FDP': [f'abl_fd10_pop1kd05w192_s{s}' for s in S4],
    'FDPd': [f'abl_fd10_pop1kd05t25_s{s}' for s in S4],
    'FDPw8': [f'abl_fd10_pop1kd05w192_s{s}' for s in range(951, 959)],
    'P1w': [f'abl_e10_pop1kd05w192_s{s}' for s in S4],
    'P1d': [f'abl_e10_pop1kd05t25_s{s}' for s in S4],
}
COMPS = {'C1': ('FD16', 'FDP'), 'C2': ('FD8', 'FDP'), 'C3': ('W16', 'P1w'), 'C4': ('W8', 'P1w')}
SUF = {'val': ('n2cpv3s3_val_oof', 'trainonly_val'), 'test': ('n2cpv3s3_test', 'trainonly_test')}
_META = {}


def meta(sp):
    if sp not in _META:
        f, M, off = feed_meta(sp); _META[sp] = (f, M, np.asarray(off, np.int64))
    return _META[sp]


def labels():
    return np.load(FLAT / 'val/cand_target.npy')


def member_z(m, sp):
    """(z of the fine-tuned file, z of the train-only file), float32, per-feed standardised exactly as build_r4.py (stack_val.feed_rank_z)."""
    f, M, off = meta(sp)
    return tuple(feed_rank_z(off, M, np.asarray(np.load(P / f'nn_{m}_{s}.npy'), np.float32))[:, 1] for s in SUF[sp])


def pool(names, sp, cache=None):
    """member-equal mean of per-feed z -> (FT pool, TO pool), float32."""
    zs = [cache[(m, sp)] if cache is not None and (m, sp) in cache else member_z(m, sp) for m in names]
    return np.mean([z[0] for z in zs], 0), np.mean([z[1] for z in zs], 0)


def ref_final(sp, bF, bT, pF, pT, a_ft, a_to, c):
    """build_r4.py's path with the project functions (float32). a_ft / a_to / c may be scalars or per-row arrays."""
    f, M, off = meta(sp); Z = lambda x: feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]
    a_ft = np.float32(a_ft) if np.isscalar(a_ft) else np.asarray(a_ft, np.float32); a_to = np.float32(a_to) if np.isscalar(a_to) else np.asarray(a_to, np.float32)
    c = np.float32(c) if np.isscalar(c) else np.asarray(c, np.float32)
    brF = (1 - a_ft) * Z(bF) + a_ft * Z(pF); brT = (1 - a_to) * Z(bT) + a_to * Z(pT)
    return c * Z(brF) + (1 - c) * Z(brT)


@nb.njit(inline='always')
def _z(v, out):
    m = len(v); mu = 0.0
    for k in range(m): mu += v[k]
    mu /= m; s = 0.0
    for k in range(m): s += (v[k] - mu) ** 2
    sd = np.sqrt(s / m) + 1e-9
    for k in range(m): out[k] = (v[k] - mu) / sd


@nb.njit(parallel=True, cache=False)
def ndcg_table(off, M, y, bF, bT, pF, pT, AG, CG):
    """N[f, ia_ft, ia_to, ic] = NDCG of feed f under final = c z((1-a_ft) z(bF) + a_ft z(pF)) + (1-c) z((1-a_to) z(bT) + a_to z(pT)); binary gain, pessimistic ties, no positives -> 1."""
    nf = len(M); nA = len(AG); nC = len(CG)
    out = np.ones((nf, nA, nA, nC))
    for f in nb.prange(nf):
        o = off[f]; m = M[f]
        npos = 0
        for k in range(m):
            if y[o + k] > 0: npos += 1
        if npos == 0: continue
        pos = np.empty(npos, np.int64); j = 0
        for k in range(m):
            if y[o + k] > 0: pos[j] = k; j += 1
        idcg = 0.0
        for r in range(npos): idcg += 1.0 / np.log2(r + 2.0)
        zbF = np.empty(m); zbT = np.empty(m); zpF = np.empty(m); zpT = np.empty(m)
        _z(bF[o:o + m].astype(np.float64), zbF); _z(bT[o:o + m].astype(np.float64), zbT); _z(pF[o:o + m].astype(np.float64), zpF); _z(pT[o:o + m].astype(np.float64), zpT)
        ZF = np.empty((nA, m)); ZT = np.empty((nA, m)); tmp = np.empty(m); s = np.empty(m)
        for ia in range(nA):
            a = AG[ia]
            for k in range(m): tmp[k] = (1 - a) * zbF[k] + a * zpF[k]
            _z(tmp, ZF[ia])
            for k in range(m): tmp[k] = (1 - a) * zbT[k] + a * zpT[k]
            _z(tmp, ZT[ia])
        for ia in range(nA):
            for ib in range(nA):
                for ic in range(nC):
                    c = CG[ic]
                    for k in range(m): s[k] = c * ZF[ia, k] + (1 - c) * ZT[ib, k]
                    dcg = 0.0
                    for q in range(npos):
                        i = pos[q]; si = s[i]; r = 0
                        for k in range(m):
                            if s[k] > si: r += 1
                            elif s[k] == si and k != i and (y[o + k] <= 0 or k < i): r += 1
                        dcg += 1.0 / np.log2(r + 2.0)
                    out[f, ia, ib, ic] = dcg / idcg
    return out


def table(comp_pools, y=None):
    f, M, off = meta('val'); y = labels() if y is None else y
    bF, bT, pF, pT = comp_pools
    return ndcg_table(off, M, np.asarray(y, np.float64), bF, bT, pF, pT, AG, CG).reshape(len(M), NCOMB)


def cidx(ia_ft, ia_to, ic):
    return (ia_ft * NA + ia_to) * NC + ic


PLAIN = cidx(A02, A02, C05)


# ---------------------------------------------------------------- gates (label-free)
def cold_luo(sp):
    f, M, off = meta(sp); x = np.asarray(np.load(ROOT / f'nn/feat_cache/{sp}/it_hist_all_luo.npy', mmap_mode='r'))
    return np.bincount(np.repeat(np.arange(len(M)), M), (x == 0).astype(np.float64), len(M)) / M


def cold_seg(sp):
    f, M, off = meta(sp)
    t = pl.read_parquet(ROOT / ('eda/errors/segments/feed_table.parquet' if sp == 'val' else 'eda/errors/segments/test_segments.parquet'))
    a = np.full(len(M), np.nan); a[t['fidx'].to_numpy()] = t['cold_share'].to_numpy(); assert not np.isnan(a).any() and (t['M'].to_numpy() == M[t['fidx'].to_numpy()]).all()
    return a


def props(sp):
    f, M, off = meta(sp); ts = f['timestamp'].to_numpy().astype(np.int64)
    return dict(M=M, H=f['H'].to_numpy().astype(np.int64), U0=(f['user_id'].to_numpy() == 0), CL=cold_luo(sp), CS=cold_seg(sp), ts=ts,
                DAY=((ts + 3 * 3600) // 86400 - 20718 >= 2) if sp == 'val' else np.ones(len(M), bool), uid=f['user_id'].to_numpy())


G2 = ['M50', 'Mq', 'CS45', 'CLq', 'Hq', 'U0', 'DAY']; G2E = G2[:-1]; G3 = ['M3q', 'M3r', 'CL3q', 'H3q']


def gates(pr):
    """dict gate name -> int cell id per feed."""
    g = {'M50': pr['M'] >= 50, 'Mq': pr['M'] >= 70, 'CS45': pr['CS'] >= 0.45, 'CLq': pr['CL'] >= 0.4225, 'Hq': pr['H'] >= 190, 'U0': pr['U0'], 'DAY': pr['DAY']}
    g = {k: v.astype(np.int64) for k, v in g.items()}
    g['M3q'] = (pr['M'] >= 39).astype(np.int64) + (pr['M'] >= 117); g['M3r'] = (pr['M'] >= 30).astype(np.int64) + (pr['M'] >= 100)
    g['CL3q'] = (pr['CL'] >= 0.3737).astype(np.int64) + (pr['CL'] >= 0.4744); g['H3q'] = (pr['H'] >= 141).astype(np.int64) + (pr['H'] >= 241)
    return g


# ---------------------------------------------------------------- families: (name, partition key, ncell, R[n_rules, ncell] of combo indices, n_par, eligible, form)
def families():
    fam = []; ar = np.arange
    a1, c1 = np.meshgrid(ar(NA), ar(NC), indexing='ij')
    fam.append(('A', ('one',), 1, cidx(a1, a1, c1).reshape(-1, 1), 2, True, 'A'))
    aF, aT, c = np.meshgrid(ar(NA), ar(NA), ar(NC), indexing='ij')
    fam.append(('B', ('one',), 1, cidx(aF, aT, c).reshape(-1, 1), 3, True, 'B'))
    for g in G2:
        el = g != 'DAY'
        a0, a1_ = [x.ravel() for x in np.meshgrid(ar(NA), ar(NA), indexing='ij')]
        fam.append((f'C({g})', (g,), 2, np.stack([cidx(a0, a0, C05), cidx(a1_, a1_, C05)], 1), 2, el, 'C'))
        a0, a1_, c = [x.ravel() for x in np.meshgrid(ar(NA), ar(NA), ar(NC), indexing='ij')]
        fam.append((f'D({g})', (g,), 2, np.stack([cidx(a0, a0, c), cidx(a1_, a1_, c)], 1), 3, el, 'D'))
        c0, c1_ = [x.ravel() for x in np.meshgrid(ar(NC), ar(NC), indexing='ij')]
        fam.append((f'E({g})', (g,), 2, np.stack([cidx(A02, A02, c0), cidx(A02, A02, c1_)], 1), 2, el, 'E'))
        c0, c1_, a = [x.ravel() for x in np.meshgrid(ar(NC), ar(NC), ar(NA), indexing='ij')]
        fam.append((f'F({g})', (g,), 2, np.stack([cidx(a, a, c0), cidx(a, a, c1_)], 1), 3, el, 'F'))
        f0, f1, t0, t1 = [x.ravel() for x in np.meshgrid(ar(NA), ar(NA), ar(NA), ar(NA), indexing='ij')]
        fam.append((f'I({g})', (g,), 2, np.stack([cidx(f0, t0, C05), cidx(f1, t1, C05)], 1), 4, el, 'I'))
    for t in G3:
        a0, a1_, a2 = [x.ravel() for x in np.meshgrid(ar(NA), ar(NA), ar(NA), indexing='ij')]
        fam.append((f'G({t})', (t,), 3, np.stack([cidx(a0, a0, C05), cidx(a1_, a1_, C05), cidx(a2, a2, C05)], 1), 3, True, 'G'))
        a0, a1_, a2, c = [x.ravel() for x in np.meshgrid(ar(NA), ar(NA), ar(NA), ar(NC), indexing='ij')]
        fam.append((f'H({t})', (t,), 3, np.stack([cidx(a0, a0, c), cidx(a1_, a1_, c), cidx(a2, a2, c)], 1), 4, True, 'H'))
        c0, c1_, c2 = [x.ravel() for x in np.meshgrid(ar(NC), ar(NC), ar(NC), indexing='ij')]
        fam.append((f'L({t})', (t,), 3, np.stack([cidx(A02, A02, c0), cidx(A02, A02, c1_), cidx(A02, A02, c2)], 1), 3, True, 'L'))
    a0, a1_, c0, c1_ = [x.ravel() for x in np.meshgrid(ar(NA), ar(NA), ar(NC), ar(NC), indexing='ij')]
    for g in G2E:                     # J: a by g, c by g2
        for g2 in G2E:
            if g == g2:
                fam.append((f'J({g},{g2})', (g,), 2, np.stack([cidx(a0, a0, c0), cidx(a1_, a1_, c1_)], 1), 4, True, 'J'))
            else:                     # partition key = sorted pair (by G2E order); cell = 2 * cell(first) + cell(second)
                i, j = G2E.index(g), G2E.index(g2); key = (g, g2) if i < j else (g2, g)
                aa = (a0, a1_); cc = (c0, c1_); cols = []
                for u in range(2):
                    for v in range(2):
                        ga, gc = (u, v) if i < j else (v, u)        # cell of g (drives a) and of g2 (drives c)
                        cols.append(cidx(aa[ga], aa[ga], cc[gc]))
                fam.append((f'J({g},{g2})', key, 4, np.stack(cols, 1), 4, True, 'J'))
    q = [x.ravel() for x in np.meshgrid(ar(NA), ar(NA), ar(NA), ar(NA), indexing='ij')]
    for i, g in enumerate(G2E):
        for g2 in G2E[i + 1:]:
            fam.append((f'K({g},{g2})', (g, g2), 4, np.stack([cidx(x, x, C05) for x in q], 1), 4, True, 'K'))
    return fam


def partitions(gt):
    """dict partition key -> (cell id per feed, ncell)."""
    out = {('one',): (np.zeros(len(gt['M50']), np.int64), 1)}
    for g in G2: out[(g,)] = (gt[g], 2)
    for t in G3: out[(t,)] = (gt[t], 3)
    for i, g in enumerate(G2E):
        for g2 in G2E[i + 1:]: out[(g, g2)] = (2 * gt[g] + gt[g2], 4)
    return out


def folds(uid, seed, K=5):
    """user-grouped folds; each user_id 0 feed is its own group."""
    grp = uid.astype(np.int64).copy(); z = grp == 0; grp[z] = -1 - np.arange(z.sum())
    u, inv = np.unique(grp, return_inverse=True); rng = np.random.default_rng(seed); perm = rng.permutation(len(u))
    fu = np.empty(len(u), np.int64); fu[perm] = np.arange(len(u)) % K
    return fu[inv]


SEEDS = [11, 22, 33, 44, 55]


def block_sums(N, fold, cell, ncell, K=5):
    """S[fold, cell, comb] = sum of N over the feeds of (fold, cell)."""
    n = len(fold); g = fold * ncell + cell
    Mx = sps.csr_matrix((np.ones(n), (g, np.arange(n))), shape=(K * ncell, n))
    return np.asarray(Mx @ N).reshape(K, ncell, N.shape[1])


def best(T, R):
    """T[ncell, comb] sums over a training set; R[n_rules, ncell] -> (index of the best rule, its training sum)."""
    sc = T[np.arange(R.shape[1])[None, :], R].sum(1); i = int(np.argmax(sc)); return i, sc[i]


def search(N, parts, fams, fold_list, want_vec=True, K=5):
    """The whole declared search on one per-feed table N[n, comb].
    Returns dict with: ins[fam] (in-sample best mean), ins_rule[fam] (row of R), nested_sum[fam] (mean over seeds of the nested SUM), nested vec per family (mean over seeds) if want_vec,
    variants 'B-all', 'B-3', 'G' (same quantities), and the families chosen by the variants."""
    n = N.shape[0]; nf = len(fams); names = [f[0] for f in fams]
    elig = np.array([f[5] for f in fams]); le3 = np.array([f[4] <= 3 for f in fams]) & elig
    tot = {}
    res = {'ins': np.zeros(nf), 'ins_rule': [None] * nf, 'nested': np.zeros(nf), 'nested_seed': np.zeros((len(fold_list), nf))}
    var = {v: {'nested': 0.0, 'nested_seed': np.zeros(len(fold_list)), 'chosen': []} for v in ('B-all', 'B-3', 'G')}
    if want_vec:
        res['vec'] = np.zeros((nf, n)); [var[v].__setitem__('vec', np.zeros(n)) for v in var]
    plain_mean = N[:, PLAIN].mean()
    for si, fold in enumerate(fold_list):
        S = {k: block_sums(N, fold, c, nc, K) for k, (c, nc) in parts.items()}
        nfold = np.bincount(fold, minlength=K)
        inner = np.zeros((K, nf)); outer_tr = np.zeros((K, nf)); held = np.zeros((K, nf)); rule_o = np.zeros((K, nf), np.int64)
        for fi, (name, key, ncell, R, npar, el, form) in enumerate(fams):
            Sk = S[key]; Tall = Sk.sum(0); cells = np.arange(ncell)
            if si == 0:
                i, sc = best(Tall, R); res['ins'][fi] = sc / n; res['ins_rule'][fi] = R[i]
            # outer fits (train = all but k) and the fits without two folds (inner CV)
            for k in range(K):
                Tk = Tall - Sk[k]; i, sc = best(Tk, R); rule_o[k, fi] = i; outer_tr[k, fi] = sc / (n - nfold[k]); held[k, fi] = Sk[k][cells, R[i]].sum()
            for k in range(K):
                for j in range(k + 1, K):
                    i, _ = best(Tall - Sk[k] - Sk[j], R); r = R[i]
                    inner[k, fi] += Sk[j][cells, r].sum(); inner[j, fi] += Sk[k][cells, r].sum()      # read on j counts for outer k; read on k counts for outer j
        res['nested_seed'][si] = held.sum(0) / n
        # variants
        pick = {'B-all': np.where(elig, inner, -np.inf).argmax(1), 'B-3': np.where(le3, inner, -np.inf).argmax(1), 'G': np.where(elig, outer_tr, -np.inf).argmax(1)}
        for v in var:
            var[v]['nested_seed'][si] = sum(held[k, pick[v][k]] for k in range(K)) / n; var[v]['chosen'].append([names[i] for i in pick[v]])
        if want_vec:
            for fi, (name, key, ncell, R, npar, el, form) in enumerate(fams):
                cell = parts[key][0]; comb = np.empty(n, np.int64)
                for k in range(K):
                    mk = fold == k; comb[mk] = R[rule_o[k, fi]][cell[mk]]
                v_ = N[np.arange(n), comb]; res['vec'][fi] += v_ / len(fold_list)
                for v in var:
                    mk = np.zeros(n, bool)
                    for k in range(K):
                        if pick[v][k] == fi: mk |= fold == k
                    var[v]['vec'][mk] += v_[mk] / len(fold_list)
    res['nested'] = res['nested_seed'].mean(0)
    for v in var: var[v]['nested'] = var[v]['nested_seed'].mean()
    res['var'] = var; res['plain'] = plain_mean; res['names'] = names
    return res


def rule_vec(N, cell, r):
    """per-feed NDCG of a rule: r[ncell] combo indices."""
    return N[np.arange(N.shape[0]), np.asarray(r)[cell]]


def se(d, w=None):
    if w is None: return d.mean(), d.std() / np.sqrt(len(d))
    m = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum(); return m, np.sqrt(np.average((d - m) ** 2, weights=w) / neff)


def fmt(d, w=None):
    m, s = se(d, w); return f'{m:+.5f}+-{s:.5f}'


def decode(c):
    c = int(c); ic = c % NC; ib = (c // NC) % NA; ia = c // (NC * NA); return (float(AG[ia]), float(AG[ib]), float(CG[ic]))
