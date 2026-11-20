"""ADDENDUM 2: L1 label-permutation null of the whole nested search (C1 pools), L2 constant-weight curves of the W16 placebo."""
import sys, time; t0 = time.time()
from lib import *
f, M, off = meta('val'); y = np.asarray(labels(), np.float64); pr = props('val'); gt = gates(pr); parts = partitions(gt); fams = families(); FL = [folds(pr['uid'], s) for s in SEEDS]; n = len(M)
elig = np.array([x[5] for x in fams]); names = [x[0] for x in fams]
@nb.njit
def perm_within(off, M, y, seed):
    np.random.seed(seed); out = y.copy()
    for f in range(len(M)):
        o = off[f]; m = M[f]; out[o:o + m] = np.random.permutation(y[o:o + m])
    return out
what = sys.argv[1]
if what in ('L1', 'L1b'):
    pools = np.load(HERE / 'pools_C1_val.npy'); nd = 60 if what == 'L1' else 240; base_seed = 1000 if what == 'L1' else 2000; nes = np.zeros((nd, len(fams))); ins = np.zeros((nd, len(fams))); var = {v: np.zeros(nd) for v in ('B-all', 'B-3', 'G')}
    for d in range(nd):
        yp = perm_within(off, M, y, base_seed + d); N = ndcg_table(off, M, yp, pools[0], pools[1], pools[2], pools[3], AG, CG).reshape(n, NCOMB); r = search(N, parts, fams, FL, want_vec=False)
        nes[d] = r['nested'] - r['plain']; ins[d] = r['ins'] - r['plain']
        for v in var: var[v][d] = r['var'][v]['nested'] - r['plain']
        if d % 10 == 0: print('perm', d, f'plain {r["plain"]:.5f}', time.time() - t0, flush=True)
    np.savez(HERE / f'leak_{what}.npz', nested=nes, ins=ins, **{'var_' + v: var[v] for v in var})
    pm = nes[:, elig].mean(1); se_ = pm.std() / np.sqrt(nd)
    print(f'L1 label-permutation null, {nd} permutations: pooled nested gain mean {pm.mean():+.6f} SE {se_:.6f} (z {pm.mean() / se_:+.2f}) | per-family sd {np.median(nes[:, elig].std(0)):.5f} | in-sample optimism mean {ins[:, elig].mean():+.5f}')
    ok = abs(pm.mean()) <= 2.5 * se_
    for v in var:
        s_ = var[v].std() / np.sqrt(nd); print(f'  variant {v}: mean {var[v].mean():+.6f} SE {s_:.6f} (z {var[v].mean() / s_:+.2f})'); ok &= abs(var[v].mean()) <= 2.5 * s_
    zf = nes[:, elig].mean(0) / (nes[:, elig].std(0) / np.sqrt(nd)); print(f'  per-family z of the mean: min {zf.min():+.2f} max {zf.max():+.2f} mean {zf.mean():+.2f}; share |z| > 2: {(np.abs(zf) > 2).mean():.3f}')
    print('L1', 'PASS (no leak)' if ok else 'FAIL')
else:
    for key in ('W16', 'FD16'):
        mem = MEMBERS[key]; zc = {m: member_z(m, 'val') for m in mem}; rng = np.random.default_rng(424242); nd = 40; ca = np.zeros((nd, NA)); cc = np.zeros((nd, NC))
        for d in range(nd):
            perm = rng.permutation(16); pp, bb = perm[:4], perm[4:]; sw = set(rng.choice(bb, 6, replace=False).tolist()) | set(rng.choice(pp, 2, replace=False).tolist())
            def pl_pool(idx):
                return np.mean([zc[mem[i]][1] if i in sw else zc[mem[i]][0] for i in idx], 0), np.mean([zc[mem[i]][0] if i in sw else zc[mem[i]][1] for i in idx], 0)
            bF, bT = pl_pool(bb); pF, pT = pl_pool(pp); N = ndcg_table(off, M, y, bF, bT, pF, pT, AG, CG).reshape(n, NCOMB); m_ = N.mean(0)
            ca[d] = [m_[cidx(a, a, C05)] - m_[PLAIN] for a in range(NA)]; cc[d] = [m_[cidx(A02, A02, c)] - m_[PLAIN] for c in range(NC)]
        print(f'L2 {key} placebo, {nd} draws: const a (c = .5) minus PLAIN, mean over draws [share of draws > 0]: ' + ' | '.join(f'a {AG[a]:.1f}: {ca[:, a].mean():+.5f} [{(ca[:, a] > 0).mean():.2f}]' for a in range(NA)))
        print(f'L2 {key} placebo: const c (a = .2) minus PLAIN: ' + ' | '.join(f'c {CG[c]:.1f}: {cc[:, c].mean():+.5f} [{(cc[:, c] > 0).mean():.2f}]' for c in range(NC)), flush=True)
