"""verify_mech v1 (CPU, read-only on stored predictions).
 (1) M5 null: the 'systematic delta^2' estimator of mech_sysdiff.py applied to every 4-vs-4 split of the 8 reference seeds 951-958
     (35 splits, all cross pairs are different seeds) -> null spread; per-member seed-noise sd over all 70 ref 4-subsets.
 (2) member and 4-pool NDCG deltas arm - ref (same seeds 951-954, paired per feed) at the SELECTED and the @last checkpoints, val and
     holdout; null = ref{951-954} pool vs ref{955-958} pool and spread over the 35 splits.
 (3) M6: per-seed (seed-level) SE of the train-fit deltas from mech/out/trainfit2_ndcg.npz; share of feeds with no positive (NDCG 1).
"""
import itertools, os, sys
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
import numpy as np
from pathlib import Path
from pc_common import Meta, _ndcg

ROOT = Path('<repo>')
MOE, W5 = ROOT / 'lab/nn_x/w12/moe/out/preds', ROOT / 'nn_x/w5/preds'
ARMS = dict(ref='kd05w192', A='kd05w192moeffn', B='kd05w192moevo', C='kd05w192b3ff1')
m = Meta()
nd = lambda sp, s: _ndcg(np.ascontiguousarray(s, np.float32), m.y[sp], m.off[sp], m.M[sp])
se = lambda d: float(np.std(d) / np.sqrt(len(d)))


def path(g, s, sp, suf=''):
    v = f'w5_e10_{ARMS[g]}_s{s}'
    d = W5 if g == 'ref' else MOE
    return d / (f'nn_{v}{suf}_trainonly_val.npy' if sp == 'val' else f'holdout/nn_{v}{suf}_holdout.npy')


members = [('ref', s) for s in range(951, 959)] + [(g, s) for g in 'ABC' for s in range(951, 955)]
for sp in ('val', 'hold'):
    M = m.M[sp]; fid = np.repeat(np.arange(len(M)), M)
    def zc(x):
        x = x.astype(np.float64); c = x - (np.bincount(fid, x, len(M)) / M)[fid]; return c / np.sqrt((c ** 2).mean())
    Z = {k: zc(np.load(path(*k, sp))) for k in members}
    D = {}
    keys = list(Z)
    for i, p in enumerate(keys):
        for q in keys[i + 1:]:
            D[p, q] = D[q, p] = float(((Z[p] - Z[q]) ** 2).mean())
    del Z
    s2 = lambda G: np.mean([D[p, q] for p, q in itertools.combinations(G, 2)]) / 2
    def delta2(X, R, excl_same=True):
        cross = np.mean([D[p, q] for p in X for q in R if not (excl_same and p[1] == q[1])])
        return cross - s2(X) - s2(R)
    refs = [('ref', s) for s in range(951, 959)]
    null = []
    for G1 in itertools.combinations(refs, 4):
        if ('ref', 951) not in G1:
            continue
        G2 = [r for r in refs if r not in G1]
        null.append(delta2(list(G1), G2))
    null = np.array(null)
    print(f'== {sp} (1) systematic delta^2 (units: within-feed logit sd^2)')
    print(f'   NULL ref 4v4 splits of seeds 951-958 (n={len(null)}): mean {null.mean():+.5f} sd {null.std():.5f} min {null.min():+.5f} max {null.max():+.5f}'
          f'  [mech used one split: 951,952 vs 953,954 (2v2)]')
    R4 = [('ref', s) for s in range(951, 955)]
    for g in 'ABC':
        X = [(g, s) for s in range(951, 955)]
        d_same4 = delta2(X, R4)
        d_all8 = delta2(X, refs)
        print(f'   {g}: delta^2 vs ref951-954 {d_same4:+.5f} (mech) | vs all 8 ref seeds {d_all8:+.5f} | z vs null {(d_all8 - null.mean()) / null.std():+.1f} | '
              f'share of null >= value {(null >= d_all8).mean():.2f}')
    sn = np.array([np.sqrt(s2(list(G))) for G in itertools.combinations(refs, 4)])
    print(f'   per-member seed-noise sd: ref 4-subsets (n=70) mean {sn.mean():.4f} range [{sn.min():.4f}, {sn.max():.4f}]; ' +
          ' '.join(f'{g} {np.sqrt(s2([(g, s) for s in range(951, 955)])):.4f}' for g in 'ABC'))

print()
print('== (2) NDCG deltas arm - ref, same seeds 951-954, paired per feed (member = per-feed mean of the 4 seed deltas; pool = mean rank-z)')
for sp in ('val', 'hold'):
    for suf in ('', '@last'):
        base = {s: nd(sp, np.load(path('ref', s, sp, suf))) for s in range(951, 959)}
        zr = {s: m.z(sp, np.load(path('ref', s, sp, suf))) for s in range(951, 959)}
        pool_ref = nd(sp, np.mean([zr[s] for s in range(951, 955)], 0))
        pool_ref2 = nd(sp, np.mean([zr[s] for s in range(955, 959)], 0))
        mem_ref = np.mean([base[s] for s in range(951, 955)], 0)
        lab = 'selected' if suf == '' else '@last'
        nullp = []
        for G1 in itertools.combinations(range(951, 959), 4):
            if 951 not in G1:
                continue
            G2 = [s for s in range(951, 959) if s not in G1]
            nullp.append(nd(sp, np.mean([zr[s] for s in G1], 0)).mean() - nd(sp, np.mean([zr[s] for s in G2], 0)).mean())
        nullp = np.array(nullp)
        print(f'-- {sp} {lab}: ref member {mem_ref.mean():.5f}, ref pool {pool_ref.mean():.5f} (gain {pool_ref.mean() - mem_ref.mean():+.5f}); '
              f'NULL ref pool(951-954) - pool(955-958) {pool_ref.mean() - pool_ref2.mean():+.5f}; 35-split null sd of pool diff {nullp.std():.5f} (|max| {np.abs(nullp).max():.5f})')
        for g in 'ABC':
            ms = {s: nd(sp, np.load(path(g, s, sp, suf))) for s in range(951, 955)}
            dm = np.mean([ms[s] - base[s] for s in range(951, 955)], 0)
            per = [float((ms[s] - base[s]).mean()) for s in range(951, 955)]
            zp = np.mean([m.z(sp, np.load(path(g, s, sp, suf))) for s in range(951, 955)], 0)
            pa = nd(sp, zp)
            dp = pa - pool_ref
            print(f'   {g}: member {dm.mean():+.5f}+-{se(dm):.5f} (per seed {" ".join(f"{x:+.5f}" for x in per)}; seed-level SE {np.std(per, ddof=1) / 2:.5f}) | '
                  f'pool {dp.mean():+.5f}+-{se(dp):.5f} | pool gain over member {pa.mean() - np.mean([ms[s] for s in range(951, 955)], 0).mean():+.5f}')

print()
print('== (3) M6 train-fit: seed-level view (mech/out/trainfit2_ndcg.npz)')
z = np.load(ROOT / 'lab/nn_x/w12/moe/why/mech/out/trainfit2_ndcg.npz')
for suf in ('@70000', '@last'):
    for g in 'ABC':
        for part in ('train', 'hold'):
            per = np.array([float((z[f'{g}.{s}{suf}.{part}'].astype(np.float64) - z[f'ref.{s}{suf}.{part}']).mean()) for s in range(951, 955)])
            print(f'   {suf} {g} {part:5s} per-seed {" ".join(f"{x:+.5f}" for x in per)} | mean {per.mean():+.5f} seed-level SE {per.std(ddof=1) / 2:.5f}')
for part in ('train', 'hold'):
    x = z[f'ref.951@last.{part}']
    print(f'   share of {part} feeds with NDCG exactly 1 (no positive or perfect) {np.mean(x == 1.0):.4f}')
for g in ('ref', 'A', 'B', 'C'):
    per_t = [float((z[f'{g}.{s}@last.train'].astype(np.float64) - z[f'{g}.{s}@70000.train']).mean()) for s in range(951, 955)]
    per_h = [float((z[f'{g}.{s}@last.hold'].astype(np.float64) - z[f'{g}.{s}@70000.hold']).mean()) for s in range(951, 955)]
    print(f'   cooldown {g}: train per seed {" ".join(f"{x:+.4f}" for x in per_t)} | hold {" ".join(f"{x:+.4f}" for x in per_h)}')
