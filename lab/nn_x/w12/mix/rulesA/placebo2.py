"""ADDENDUM 3: matched two-member-set placebo PL-A2 (same member indices in FD16 and W16)."""
import sys, time; t0 = time.time()
from lib import *
nd = int(sys.argv[1])
f, M, off = meta('val'); y = np.asarray(labels(), np.float64); pr = props('val'); gt = gates(pr); parts = partitions(gt); fams = families(); FL = [folds(pr['uid'], s) for s in SEEDS]
nf = len(fams); n = len(M); rng = np.random.default_rng(31337); keys = ('FD16', 'W16')
zc = {k: [member_z(m, 'val') for m in MEMBERS[k]] for k in keys}; print('cached', time.time() - t0, flush=True)
nes = np.zeros((2, nd, nf)); ins = np.zeros((2, nd, nf)); mr = np.zeros((2, nd)); var = {v: np.zeros((2, nd)) for v in ('B-all', 'B-3', 'G')}
for d in range(nd):
    perm = rng.permutation(16); pp, bb = perm[:4], perm[4:]; sw = set(rng.choice(bb, 6, replace=False).tolist()) | set(rng.choice(pp, 2, replace=False).tolist())
    for ki, k in enumerate(keys):
        z = zc[k]
        def pl_pool(idx):
            return np.mean([z[i][1] if i in sw else z[i][0] for i in idx], 0), np.mean([z[i][0] if i in sw else z[i][1] for i in idx], 0)
        bF, bT = pl_pool(bb); pF, pT = pl_pool(pp); N = ndcg_table(off, M, y, bF, bT, pF, pT, AG, CG).reshape(n, NCOMB); r = search(N, parts, fams, FL, want_vec=False)
        nes[ki, d] = r['nested'] - r['plain']; ins[ki, d] = r['ins'] - r['plain']; mr[ki, d] = (np.where(M >= 50, N[:, cidx(A04, A04, C05)], N[:, PLAIN]) - N[:, PLAIN]).mean()
        for v in var: var[v][ki, d] = r['var'][v]['nested'] - r['plain']
    if d % 10 == 0 or d == nd - 1:
        print('draw', d, time.time() - t0, flush=True); np.savez(HERE / 'pl_PLA2.npz', nested=nes[:, :d + 1], ins=ins[:, :d + 1], mrule=mr[:, :d + 1], **{'var_' + v: var[v][:, :d + 1] for v in var})
print('done', time.time() - t0)
