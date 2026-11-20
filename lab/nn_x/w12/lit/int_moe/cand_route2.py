"""int_moe / cand_route2.py (block 2 of DECLARATION.txt): three-class candidate gates; shared part copied from cand_route.py. Original docstring: candidate-level cold / warm gate between two existing pools, nested by user, with placebos,
plus in-sample per-feed oracles with a same-recipe null. CPU only, stored val predictions only. Rules: DECLARATION.txt block 1.
Reads: preds/nn_*_{n2cpv3s3_val_oof,trainonly_val}.npy, data flat val, nn/feat_cache/val/it_hist_all_luo.npy. Writes: this directory only."""
import os, sys, json, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/src')
import numpy as np, numba as nb
from pathlib import Path
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path('<repo>'); P = ROOT / 'preds'; HERE = Path(__file__).resolve().parent
T0 = time.time()
LOG = open(HERE / 'cand_route2.log', 'w')


def say(*a):
    s = f'[{time.time() - T0:6.1f}s] ' + ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


S8 = range(951, 959); S4 = range(951, 955)
MEM = {
    'FD16': [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8],
    'FDP': [f'abl_fd10_pop1kd05w192_s{s}' for s in S4],
    'W16': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8],
    'P1w': [f'abl_e10_pop1kd05w192_s{s}' for s in S4],
}
SUF = ('n2cpv3s3_val_oof', 'trainonly_val')
feeds, M, off = feed_meta('val'); M = M.astype(np.int64); off = off.astype(np.int64); NF = len(M)
y = np.load(FLAT / 'val/cand_target.npy')
fid = np.repeat(np.arange(NF), M)
luo = np.asarray(np.load(ROOT / 'nn/feat_cache/val/it_hist_all_luo.npy', mmap_mode='r'), np.float32)
assert len(luo) == M.sum() == len(y)
COLD = (luo == 0)
Mrow = np.repeat(M, M)


@nb.njit(parallel=True, cache=False)
def ndcg_fast(s, y, off, M):
    n = len(M); out = np.ones(n)
    for f in nb.prange(n):
        o = off[f]; m = M[f]; npos = 0
        for k in range(m):
            if y[o + k] > 0: npos += 1
        if npos == 0: continue
        idcg = 0.0
        for r in range(npos): idcg += 1.0 / np.log2(r + 2.0)
        dcg = 0.0
        for k in range(m):
            if y[o + k] > 0:
                sk = s[o + k]; r = 0
                for j in range(m):
                    sj = s[o + j]
                    if sj > sk: r += 1
                    elif sj == sk and j != k and (y[o + j] <= 0 or j < k): r += 1
                dcg += 1.0 / np.log2(r + 2.0)
        out[f] = dcg / idcg
    return out


@nb.njit(parallel=True, cache=False)
def zfeed(x, off, M):
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]; v = x[o:o + m]; mu = v.mean(); sd = v.std() + 1e-9
        for k in range(m): out[o + k] = (v[k] - mu) / sd
    return out


def Zp(x):
    return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]


_MZ = {}


def member_z(name, br):
    k = (name, br)
    if k not in _MZ:
        _MZ[k] = Zp(np.load(P / f'nn_{name}_{SUF[br]}.npy'))
    return _MZ[k]


def pools(base, pop):
    out = {}
    for br, tag in ((0, 'ft'), (1, 'to')):
        out['B' + tag] = Zp(np.mean([member_z(m, br) for m in base], 0))
        out['P' + tag] = Zp(np.mean([member_z(m, br) for m in pop], 0))
    return out


def final(q, wrow):
    wrow = np.float32(wrow) if np.ndim(wrow) == 0 else wrow.astype(np.float32)
    bf = zfeed(((1 - wrow) * q['Bft'] + wrow * q['Pft']).astype(np.float32), off, M)
    bt = zfeed(((1 - wrow) * q['Bto'] + wrow * q['Pto']).astype(np.float32), off, M)
    return (0.5 * bf + 0.5 * bt).astype(np.float32)


def ev(s):
    return ndcg_fast(np.asarray(s, np.float32), y, off, M)


# user-grouped folds, user 0 = singleton groups
u = feeds['user_id'].to_numpy().astype(np.int64); g = u.copy(); z0 = u == 0; g[z0] = -(np.arange(z0.sum()) + 1)
GRP = np.unique(g, return_inverse=True)[1]; NG = GRP.max() + 1


def cse(d):
    d = np.asarray(d, np.float64); m = d.mean(); s = np.bincount(GRP, d - m, NG)
    return m, np.sqrt((s ** 2).sum()) / len(d)


def fmt(d):
    m, s = cse(d); return f'{m:+.5f} +- {s:.5f} (z {m / s:+.2f})' if s > 0 else f'{m:+.5f}'


def folds(seed, k=5):
    r = np.random.default_rng(seed); gf = r.integers(0, k, NG)
    return gf[GRP]



G4 = np.array([0, .2, .4, .6], np.float32)
RULES = [(a, b, c) for a in G4 for b in G4 for c in G4]
CONST_IDX = np.array([i for i, r in enumerate(RULES) if r[0] == r[1] == r[2]])


def tensor3(q, cls):
    E = np.empty((len(RULES), NF))
    for i, r in enumerate(RULES):
        E[i] = ev(final(q, np.asarray(r, np.float32)[cls]))
    return E


def nested(E, idx=None, seeds=range(5)):
    idx = np.arange(E.shape[0]) if idx is None else np.asarray(idx)
    out = np.zeros(NF); picks = {}
    for sd in seeds:
        fo = folds(1000 + sd)
        for k in range(5):
            te = fo == k
            r = idx[np.argmax(E[idx][:, ~te].mean(1))]
            out[te] += E[r, te]; picks[int(r)] = picks.get(int(r), 0) + 1
    return out / len(list(seeds)), picks


def perm_within_feed(lab, seed):
    r = np.random.default_rng(seed); key = r.random(len(lab))
    return lab[np.lexsort((key, fid))]


def main():
    res = {}
    f4 = np.asarray(np.load(FLAT / 'val/cand_f4.npy', mmap_mode='r'), np.float32)
    warm = ~COLD
    med_f4 = float(np.median(f4[warm])); med_luo = float(np.median(luo[warm]))
    CLS = {'A3': np.where(COLD, 0, np.where(f4 < med_f4, 1, 2)).astype(np.int64),
           'L3': np.where(COLD, 0, np.where(luo < med_luo, 1, 2)).astype(np.int64)}
    say(f'warm rows {warm.mean():.4f}; median f4 of warm rows {med_f4:.4f}; median luo count of warm rows {med_luo:.1f}')
    for k, c in CLS.items():
        say(f'{k}: class shares of rows {np.bincount(c, minlength=3) / len(c)}; target rate by class {[float(y[c == j].mean()) for j in range(3)]}')
    Q = {'C1': pools(MEM['FD16'], MEM['FDP']), 'C3': pools(MEM['W16'], MEM['P1w'])}
    wm = np.where(Mrow >= 50, 0.4, 0.2).astype(np.float32)
    PL = {c: ev(final(Q[c], 0.2)) for c in Q}; MR = {c: ev(final(Q[c], wm)) for c in Q}
    for c in Q:
        say(f'{c}: PLAIN {PL[c].mean():.6f} | M-RULE {MR[c].mean():.6f}')
    assert abs(PL['C1'].mean() - 0.402714) < 1e-5 and abs(PL['C3'].mean() - 0.402141) < 1e-5
    import pmix
    W = pmix.weights()
    for fam, cls in CLS.items():
        row = {}
        for c in Q:
            E = tensor3(Q[c], cls); nC, pC = nested(E, CONST_IDX); nW, pW = nested(E)
            G = nW - nC; m, s = cse(G)
            b = RULES[int(np.argmax(E.mean(1)))]
            gpriv, spriv = pmix.wmean_se(G.astype(np.float64), W['private']); gpub, spub = pmix.wmean_se(G.astype(np.float64), W['public'])
            say(f'{fam} {c}: in-sample best - PLAIN {E.mean(1).max() - PL[c].mean():+.5f} at (cold, warm-low, warm-high) = {tuple(round(float(x), 1) for x in b)}; in-sample best CONST - PLAIN {E[CONST_IDX].mean(1).max() - PL[c].mean():+.5f}')
            say(f'{fam} {c}: nested family - PLAIN {fmt(nW - PL[c])}; nested - M-RULE {fmt(nW - MR[c])}; G = nested family - nested CONST {fmt(G)}; public mix {gpub:+.5f} +- {spub:.5f}; private mix {gpriv:+.5f} +- {spriv:.5f}')
            say(f'{fam} {c}: picks {[(tuple(round(float(x), 1) for x in RULES[k]), v) for k, v in sorted(pW.items(), key=lambda t: -t[1])][:5]}')
            # fixed diagnostic contrasts vs PLAIN (not selected)
            for r in ((.2, .4, .2), (.2, .2, .4), (.2, .0, .2), (.2, .2, .0)):
                i = RULES.index(tuple(np.float32(x) for x in r)); say(f'   fixed {r} - PLAIN {fmt(E[i] - PL[c])}')
            row[c] = (m, s, gpriv)
        Gp = []
        for sd in range(5):
            cp = perm_within_feed(cls, 700 + sd)
            Ep = tensor3(Q['C1'], cp); nC, _ = nested(Ep, CONST_IDX); nW, _ = nested(Ep); Gp.append(cse(nW - nC)[0])
            say(f'{fam} placebo {sd}: G {fmt(nW - nC)}')
        g1, s1, gp1 = row['C1']; g3, s3, _ = row['C3']
        a = g1 >= 2.24 * s1; b_ = g3 >= s3; c_ = g1 > max(Gp); d = gp1 >= 0
        say(f'{fam} CRITERION: G(C1) {g1:+.5f} +- {s1:.5f} >= +2.24 SE: {a} | G(C3) {g3:+.5f} +- {s3:.5f} >= +1 SE: {b_} | > max placebo ({max(Gp):+.5f}): {c_} | private-mix G(C1) {gp1:+.5f} >= 0: {d} -> ' + ('SIGNAL' if (a and b_ and c_ and d) else 'NO SIGNAL'))
        res[fam] = {'G_C1': [float(g1), float(s1)], 'G_C3': [float(g3), float(s3)], 'G_C1_private_mix': float(gp1), 'placebo': [float(x) for x in Gp], 'verdict': 'SIGNAL' if (a and b_ and c_ and d) else 'NO SIGNAL'}
    json.dump(res, open(HERE / 'cand_route2.json', 'w'), indent=1)
    say('done')


if __name__ == '__main__':
    main()
