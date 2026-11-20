"""Variance vs bias split: pool NDCG as a function of member count (random subsets), extrapolated to infinitely many members;
per-segment version (cold/warm positives); holdout analogue."""
from lib import *

V = Val('val')
y, M, off, fid, n = V.y, V.M, V.off, V.fid, V.n
T = pl.read_parquet(OUT / 'out/feed_table.parquet')
g = lambda c: T[c].to_numpy()
keep = ~g('leak')
D = np.load(OUT / 'out/member_ndcg.npz'); mem = list(D['members']); nG3 = int(D['G3n'])
cache = {}
for m in mem:
    cache[(m, 'to')] = load_member(V, m, 'to'); cache[(m, 'ft')] = load_member(V, m, 'ft')
log('loaded members')
Mh, oh = np.load(PH / 'holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(PH / 'holdout_y.npy')
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1].astype(np.float64)
HO = {m: zh(np.load(PH / f'nn_{m}_holdout.npy')) for m in mem}
cp, wp = g('coldpos'), g('warmpos')
rng = np.random.default_rng(0)


def curve(pool_members, label, reps=8):
    print(f'\n=== {label}: pool NDCG vs number of members (random subsets, {reps} reps; train-only half and final) ===')
    print(f'{"k":>3s} {"TO val":>9s} {"final val":>9s} {"TO coldpos":>10s} {"TO warmpos":>10s} {"holdout":>9s}')
    rows = []
    ks = [1, 2, 3, 4, 6, 8, 12, 16, 20, 25, 30]
    for k in [k for k in ks if k <= len(pool_members)]:
        acc = []
        for _ in range(reps if k < len(pool_members) else 1):
            ms = list(rng.choice(pool_members, k, replace=False))
            to = V.z(np.mean([cache[(m, 'to')] for m in ms], 0)); ft = V.z(np.mean([cache[(m, 'ft')] for m in ms], 0))
            a = V.ndcg(to); b = V.ndcg(0.5 * to + 0.5 * ft)
            h = ndcg_per_feed(zh(np.mean([HO[m] for m in ms], 0)), yh, oh, Mh).mean()
            acc.append([a[keep].mean(), b[keep].mean(), a[cp & keep].mean(), a[wp & keep].mean(), h])
        acc = np.mean(acc, 0); rows.append([k] + list(acc))
        print(f'{k:3d} ' + ' '.join(f'{x:9.5f}' for x in acc))
    R = np.array(rows)
    # fit NDCG(k) = a - b/k on k >= 2
    for j, nm in enumerate(['TO val', 'final val', 'TO coldpos', 'TO warmpos', 'holdout']):
        kk = R[:, 0]; yy = R[:, j + 1]; mk = kk >= 2
        A = np.column_stack([np.ones(mk.sum()), 1 / kk[mk]]); coef, *_ = np.linalg.lstsq(A, yy[mk], rcond=None)
        print(f'  fit {nm:10s}: k->inf {coef[0]:.5f}; at k={int(kk[-1])} {coef[0] + coef[1]/kk[-1]:.5f}; remaining variance headroom {-coef[1]/kk[-1]:+.5f}; '
              f'single {yy[0]:.5f}')
    return R


curve(mem[:nG3], 'G3 members (20)')
curve(mem, 'G3+NEW members (30)', reps=6)
curve([m for m in mem if m.startswith('r6_cd_m02_r')] + [m for m in mem if m.startswith('r6_cd_m02_p70k')], 'M2 = F7x+F7x70 (14, same 7 trunks re-cooled twice)', reps=6)
