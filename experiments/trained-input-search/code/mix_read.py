# Is a member with the new input a different view for the pool? It compares pair correlations of members with
# and without the input, and scores 200 random mixed 8-member pools against the two pure pools.
# CPU only, no arguments; it reuses the loader of nx3_read_a.py.
# Output: ../results/walk_input_mix_read.txt. Cross correlation 0.9487 against 0.9490 and 0.9495 inside each
# arm on val; size-matched mix gain -0.00017 +- 0.00019.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/nx3/mix_read.py
"""MIX READ (ledger 2026-11-16T08:3xZ, declared before it was computed): descriptive, CPU only, on the stage-A hard-label members
(arm abl_e10_nx3_s951-958, controls w5_e10_base_s951-956 + abl_e10_base_s957-958; selected checkpoints; predictions already opened by the stage-A read).
(i) mean member pair correlation (Pearson of the within-feed z scores over rows) nx3-nx3, base-base, nx3-base on val and on the lag-view holdout;
(ii) size-matched pools: mean per-feed NDCG of 200 random mixed 8-pools (4 base + 4 nx3; seed 20261116) minus the mean of the two pure 8-pools, paired per feed.
Reuses Reader of nx3_read_a.py (same loading, same NDCG). Writes nn_x/w12/nx3/out/mix_read.txt."""
import sys, itertools
sys.path.insert(0, '<repo>/nn_x/w12/nx3')
import numpy as np
import nx3_read_a as ra

R = ra.Reader('<repo>/nn_x/w12/nx3/out/mix_read.txt'); P = R.P
arm = ra.members('abl_e10_nx3:951-958'); ctl = ra.members('w5_e10_base:951-956+abl_e10_base:957-958')
A = [R.load(v, '') for v in arm]; C = [R.load(v, '') for v in ctl]
m = R.m
P('=== nx3 MIX READ (descriptive; stage-A hard-label members, selected checkpoints) ===')
rng = np.random.default_rng(20261116)
draws = [(rng.choice(8, 4, replace=False), rng.choice(8, 4, replace=False)) for _ in range(200)]
for tag, iz, sp, mask_feed in (('val', 0, 'val', None), ('lag-view holdout', 1, 'hold', R.sub['lagview']), ('holdout, all 30,000', 1, 'hold', None)):
    ZA = np.stack([a[iz] for a in A]).astype(np.float32); ZC = np.stack([c[iz] for c in C]).astype(np.float32)
    off, M = m.off[sp], m.M[sp]
    if mask_feed is not None:
        rows = np.zeros(ZA.shape[1], bool)
        for f in np.flatnonzero(mask_feed): rows[off[f]:off[f] + M[f]] = True
    else:
        rows = np.ones(ZA.shape[1], bool)
    def cc(X, Y, same):
        out = []
        for i in range(len(X)):
            for j in range(len(Y)):
                if same and j <= i: continue
                out.append(np.corrcoef(X[i][rows], Y[j][rows])[0, 1])
        return float(np.mean(out))
    P(f'-- {tag}: mean pair correlation  nx3-nx3 {cc(ZA, ZA, True):.4f} | base-base {cc(ZC, ZC, True):.4f} | nx3-base {cc(ZA, ZC, False):.4f}')
    fm = np.ones(len(M), bool) if mask_feed is None else mask_feed
    nA = R.nd(sp, ZA.mean(0)); nC = R.nd(sp, ZC.mean(0))
    mix = np.zeros(len(M), np.float64)
    for ia, ic in draws:
        mix += R.nd(sp, (ZA[ia].sum(0) + ZC[ic].sum(0)) / 8.0)
    mix /= len(draws)
    # size-matched pure 4-pools for scale: mean of 200 random pure 4+4 from within one arm is the 8-pool itself, so compare with the two pure 8-pools
    d = (mix - 0.5 * (nA + nC))[fm]; dA = (nA - nC)[fm]; dm = (mix - nC)[fm]
    se = lambda x: x.std() / np.sqrt(len(x))
    P(f'   pure 8-pools: nx3 {nA[fm].mean():.5f} | base {nC[fm].mean():.5f} | nx3 - base {dA.mean():+.5f} +- {se(dA):.5f}')
    P(f'   mixed 8-pool (4 + 4, mean of 200 draws) {mix[fm].mean():.5f} | MIX GAIN = mixed - mean of the two pure pools {d.mean():+.5f} +- {se(d):.5f} (z {d.mean() / se(d):+.2f}) | mixed - base {dm.mean():+.5f} +- {se(dm):.5f}')
    full = R.nd(sp, (ZA.sum(0) + ZC.sum(0)) / 16.0)
    d16 = (full - nC)[fm]; d16a = (full - nA)[fm]
    P(f'   all 16 (8 + 8) {full[fm].mean():.5f} | minus base 8-pool {d16.mean():+.5f} +- {se(d16):.5f} | minus nx3 8-pool {d16a.mean():+.5f} +- {se(d16a):.5f}   (16 against 8: includes the pool-size effect)')
P('NOTE: the SE of the mix gain is the paired per-feed SE of a mean over 200 draws; the draws share members, so it understates the seed part.')
