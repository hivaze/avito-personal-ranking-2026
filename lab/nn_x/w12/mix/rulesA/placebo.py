"""Step 4 placebo (DECLARATION block 5).  placebo.py PLA <FD16|W16> <ndraw>   |   placebo.py PLG <C1|C2|C3|C4> <nperm>
PLA: 4 random members of the 16 wide students = pseudo popularity-light pool, the other 12 = base; a random half of each (6 / 2) swaps its fine-tuned / train-only files. Identical nested search.
PLG: real pools of the composition; the feed-property vector (all gates jointly) permuted across feeds. Identical nested search.
Output: pl_<mode>_<key>.npz  (ins[d, fam], nested[d, fam] = gains over the draw's own PLAIN; var_<v>[d]; mrule[d] = fixed M-rule minus plain; greedy in-sample max)."""
import sys, time; t0 = time.time()
from lib import *
mode, key, nd = sys.argv[1], sys.argv[2], int(sys.argv[3])
f, M, off = meta('val'); y = np.asarray(labels(), np.float64); pr = props('val'); gt = gates(pr); parts = partitions(gt); fams = families(); FL = [folds(pr['uid'], s) for s in SEEDS]
nf = len(fams); elig = np.array([x[5] for x in fams]); n = len(M)
rng = np.random.default_rng({'PLA': 7001, 'PLG': 9001}[mode] + sum(map(ord, key)))
ins = np.zeros((nd, nf)); nes = np.zeros((nd, nf)); var = {v: np.zeros(nd) for v in ('B-all', 'B-3', 'G')}; mr = np.zeros(nd); plain = np.zeros(nd); draws = []
if mode == 'PLA':
    mem = MEMBERS[key]; assert len(mem) == 16
    zc = {m: member_z(m, 'val') for m in mem}; print('member z cached', time.time() - t0, flush=True)
else:
    N0 = np.load(HERE / f'N_{key}.npy')
for d in range(nd):
    if mode == 'PLA':
        perm = rng.permutation(16); pp, bb = perm[:4], perm[4:]; sw = set(rng.choice(bb, 6, replace=False).tolist()) | set(rng.choice(pp, 2, replace=False).tolist())
        def pl_pool(idx):
            F = np.mean([zc[mem[i]][1] if i in sw else zc[mem[i]][0] for i in idx], 0); T = np.mean([zc[mem[i]][0] if i in sw else zc[mem[i]][1] for i in idx], 0); return F, T
        bF, bT = pl_pool(bb); pF, pT = pl_pool(pp)
        N = ndcg_table(off, M, y, bF, bT, pF, pT, AG, CG).reshape(n, NCOMB); prt = parts; draws.append((perm.tolist(), sorted(sw)))
    else:
        N = N0; p = rng.permutation(n); prt = partitions({k: v[p] for k, v in gt.items()})
    r = search(N, prt, fams, FL, want_vec=False)
    ins[d] = r['ins'] - r['plain']; nes[d] = r['nested'] - r['plain']; plain[d] = r['plain']
    for v in var: var[v][d] = r['var'][v]['nested'] - r['plain']
    cm = prt[('M50',)][0]; mr[d] = (np.where(cm == 1, N[:, cidx(A04, A04, C05)], N[:, PLAIN]) - N[:, PLAIN]).mean()
    if d % 10 == 0 or d == nd - 1:
        print(f'draw {d} plain {plain[d]:.5f} | C(M50) ins {ins[d, [x[0] for x in fams].index("C(M50)")]:+.5f} nested {nes[d, [x[0] for x in fams].index("C(M50)")]:+.5f} | B-all {var["B-all"][d]:+.5f} | {time.time() - t0:.0f}s', flush=True)
        np.savez(HERE / f'pl_{mode}_{key}.npz', ins=ins[:d + 1], nested=nes[:d + 1], mrule=mr[:d + 1], plain=plain[:d + 1], names=np.array([x[0] for x in fams]), elig=elig, **{'var_' + v: var[v][:d + 1] for v in var})
print('done', time.time() - t0)
