# The null used by the independent check of both searches: one random sign per user, applied to
# every per-feed contrast against the plain weights, so that the noise level of the real pools is
# kept and any true effect is removed. The whole nested search is repeated on each draw.
# How it was run: placebo.py <first draw> <n draws> <out tag> from lab/nn_x/w12/mix/review, in two
# processes (logs log_placebo_a.txt and log_placebo_b.txt), 400 draws in total.
# What it produced: null distributions read by placebo_read.py; copy of the read-out in
# results/mix_check_sign_flip_null.txt. Under this null the best family of the first search has
# a family-wise p of 0.16 or more.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/mix/review/placebo.py
"""CHECK 3: sign-flip placebo (reviewer's design). usage: placebo.py <first draw> <n draws> <out tag>; draw -1 = the real data (all signs +1), always computed first.
One Rademacher sign per user group; every contrast to PLAIN flipped jointly on all compositions. Fold seed 7001 inside the placebo."""
import sys, time
sys.path.insert(0, '<repo>/nn_x/w12/mix/review')
from rlib import *
d0, nd, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
t0 = time.time(); mv = meta('val'); g = groups(); M = mv['M']; H = mv['H']; big = M >= 50; n = mv['n']
FL = [folds(g, 7001)]; hq = np.quantile(H, [.2, .35, .5, .65, .8]); fams = fam_defs(dict(M=M, H=H), hq); fd = dict(fams); names = [f[0] for f in fams]
KEYS = ('C1', 'C2', 'C3', 'C4'); HALV = ('HA', 'HB', 'WA', 'WB'); TW = {'C1': 'C3', 'C3': 'C1', 'C2': 'C4', 'C4': 'C2'}
D0 = {k: tensor(k) - tensor(k)[:, [PLAIN]] for k in KEYS + HALV}
# origin scan: 3 gates x 25 weight pairs from {0, .1, .2, .3, .4} at w_ft 0.5
cs = cold_pmix('val'); GATES = [big.astype(np.int64), (cs >= 0.30).astype(np.int64), (cs >= 0.45).astype(np.int64)]
W5 = [comb(w, 0.5) for w in (0, .1, .2, .3, .4)]; R75 = np.array([[a, b] for a in W5 for b in W5])
one = np.zeros(n, np.int64)


def scan75(Dsel):
    """in-sample best of the 75 rules on Dsel -> (gate index, combo per cell, in-sample mean)."""
    bestv = (-np.inf, None, None)
    for gi, cell in enumerate(GATES):
        T = block_sums(Dsel, one, cell, 2, 1)[0]; sc, r = best_rule(T, R75)
        if sc > bestv[0] + 1e-12: bestv = (sc, gi, r)
    return bestv[1], bestv[2], bestv[0] / n


def read(Dt, gi, r):
    return Dt[np.arange(n), r[GATES[gi]]].mean()


def cross(Dsrc, Dtgt, subs, fold, K=5):
    S = [block_sums(Dsrc, fold, cell, nc, K) for (cell, nc, R) in subs]; tot = 0.0
    for k in range(K):
        bk = (-np.inf, None, None)
        for si, (cell, nc, R) in enumerate(subs):
            sc, r = best_rule(S[si].sum(0) - S[si][k], R)
            if sc > bk[0] + 1e-12: bk = (sc, si, r)
        cell = subs[bk[1]][0]; mk = fold == k; tot += Dtgt[mk][np.arange(mk.sum()), bk[2][cell[mk]]].sum()
    return tot / n


def stats(sg):
    s = sg[g][:, None]; D = {k: s * D0[k] for k in D0}; out = {}
    for k in KEYS:
        r = nested(D[k], fams, FL, want_vec=False)
        out[f'nest|{k}'] = np.array([r[nm]['nested'] for nm in names]); out[f'ins|{k}'] = np.array([r[nm]['ins'] for nm in names])
        out[f'fix|{k}'] = np.array([np.where(big, D[k][:, C04], 0.0).mean()])
    for k in KEYS:
        out[f'H3|{k}'] = np.array([cross(D[TW[k]], D[k], fd['MSTEP50'], FL[0])]); out[f'H3a|{k}'] = np.array([cross(D[TW[k]], D[k], fd['C(M50)'], FL[0])])
    for k in HALV: out[f'fix|{k}'] = np.array([np.where(big, D[k][:, C04], 0.0).mean()])
    # P4 origin emulation
    gi, r, v = scan75(D['C3']); out['P4|C3sel'] = np.array([v, read(D['C1'], gi, r), read(D['C2'], gi, r), read(D['C4'], gi, r), gi, r[0], r[1]])
    gi, r, v = scan75(D['C1']); out['P4|C1sel'] = np.array([v, read(D['C3'], gi, r), read(D['C4'], gi, r), read(D['C2'], gi, r), gi, r[0], r[1]])
    gi, r, v = scan75(0.5 * (D['C3'] + D['C4'])); out['P5|C34sel'] = np.array([v, read(0.5 * (D['C1'] + D['C2']), gi, r), gi, r[0], r[1]])
    gi, r, v = scan75(0.5 * (D['C1'] + D['C2'])); out['P5|C12sel'] = np.array([v, read(0.5 * (D['C3'] + D['C4']), gi, r), gi, r[0], r[1]])
    return out


ng = g.max() + 1; real = stats(np.ones(ng)); acc = {k: [v] for k, v in real.items()}
print('real done', f'{time.time() - t0:.0f}s', flush=True)
for d in range(d0, d0 + nd):
    sg = np.random.default_rng([20261115, d]).integers(0, 2, ng) * 2.0 - 1.0; o = stats(sg)
    for k in o: acc[k].append(o[k])
    if (d - d0) % 20 == 19: print(f'draw {d} done {time.time() - t0:.0f}s', flush=True)
np.savez(HERE / f'placebo_{tag}.npz', names=np.array(names), **{k: np.array(v) for k, v in acc.items()})
print('saved', f'{time.time() - t0:.0f}s')
