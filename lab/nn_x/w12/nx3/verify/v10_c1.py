"""nx3 verifier v10: MY walk code against the STORED array of record cache/c09_hold.npz (written 2026-11-13 by c09_build.py, before this test existed):
all 30,000 holdout feeds, c09's holdout convention = dense train sequences of the days before the row's own day, no sparse members, deg = dense postings.
Ties the three objects together: array of record == my independent code == (v02 / v05) the installed files.  Label-free."""
from v_common import *
P = Log(f'{OUT}/v10_c1.txt')
TR = Split('train'); maxid = 111319574
pr = np.load(ROOT + '/eda/labels_position/out/train_prandom.npz'); p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[TR.M < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((p > 0.9) & (TR.M >= 50))[0], 30000, replace=False))
c9 = np.load(ROOT + '/nn_x/insights/ccb_1113/scratch/tester3/cache/c09_hold.npz'); Mh = TR.M[hold]; offh = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64)
assert len(c9['rp3b05']) == int(Mh.sum())
mine = np.zeros((int(Mh.sum()), 2)); tot_bad = 0
for s in np.unique(TR.slot[hold]):
    qi = np.nonzero(TR.slot[hold] == s)[0]
    if s == 0:
        P(f'slot 0: no earlier day -> zeros ({len(qi)} feeds)'); continue
    dn = dense_members(TR, int(s)); U = Universe([(TR, dn, True)], maxid); Q = Queries(TR, hold[qi], np.full(len(qi), s), maxid); r = run(U, Q)
    lr = np.repeat(offh[qi] - Q.roff, Q.M) + np.arange(Q.nrow)
    mine[lr, 0] = r['rp3'][0]; mine[lr, 1] = r['p3a'][0]
    a, b = r['rp3'][0], c9['rp3b05'][lr]; a2, b2 = r['p3a'][0], c9['p3a'][lr]
    bad = int((a != b).sum() + (a2 != b2).sum()); tot_bad += bad
    P(f'slot {s:2d}: {len(qi)} feeds, {Q.nrow} rows, dense seqs {len(dn)}; non-zero share {np.mean(a2 > 0):.4f}; max abs diff rp3b05 {np.abs(a - b).max():.3e}, p3a {np.abs(a2 - b2).max():.3e}; rows not bit-equal {bad}')
    del U, Q, r
P(f'ALL 30,000 holdout feeds, {len(mine)} rows: max abs diff rp3b05 {np.abs(mine[:, 0] - c9["rp3b05"]).max():.3e}, p3a {np.abs(mine[:, 1] - c9["p3a"]).max():.3e}; zero-pattern mismatches {int(((mine[:, 1] > 0) != (c9["p3a"] > 0)).sum())}; rows not bit-equal {tot_bad}; non-zero share mine {np.mean(mine[:, 1] > 0):.6f} stored {np.mean(c9["p3a"] > 0):.6f}')
P('DONE')
