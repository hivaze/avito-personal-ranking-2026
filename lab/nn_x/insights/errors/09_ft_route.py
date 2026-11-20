"""Nested (user-grouped 5-fold) test: does routing the FT weight w_ft by an OBSERVABLE feed segment beat the fixed 0.5?
Pre-declared here (2 segmentations only): M<50 vs M>=50; feed cold share (LUO) <30% / 30-45% / >=45%.
Grid w_ft in {0, 0.25, 0.5, 0.75, 1.0}, chosen per segment on 4 folds, applied to the 5th. Pools: G3 and N1."""
from lib import *
from sklearn.model_selection import GroupKFold

V = Val('val')
T = pl.read_parquet(OUT / 'out/feed_table.parquet')
g = lambda c: T[c].to_numpy()
S = dict(np.load(OUT / 'out/cand_scores.npz'))
M = V.M
users = g('user_id').copy(); u0 = users == 0
users[u0] = -np.arange(u0.sum()) - 1  # user 0 as singletons
grid = [0.0, 0.25, 0.5, 0.75, 1.0]
segs = {'M<50|>=50': np.where(M < 50, 0, 1),
        'coldshare <30|30-45|>=45': np.select([g('cold_share_luo') < 0.3, g('cold_share_luo') < 0.45], [0, 1], 2)}
for pool in ['G3', 'N1']:
    ft, to = S[pool + '_ft'].astype(np.float64), S[pool + '_to'].astype(np.float64)
    ND = np.stack([V.ndcg(w * ft + (1 - w) * to) for w in grid])  # (grid, feeds)
    base = ND[2]
    for sname, sg in segs.items():
        out = np.zeros(V.n); chosen = []
        for tr, te in GroupKFold(5).split(np.zeros(V.n), groups=users):
            for s in np.unique(sg):
                mtr = tr[sg[tr] == s]; mte = te[sg[te] == s]
                k = int(np.argmax(ND[:, mtr].mean(1)))
                out[mte] = ND[k, mte]; chosen.append((s, grid[k]))
        d = out - base
        print(f'{pool} route w_ft by {sname:26s}: nested {d.mean():+.5f} ± {se(d):.5f}  chosen per fold {chosen}')
    # descriptive (in-sample) per-segment optimum
    for sname, sg in segs.items():
        for s in np.unique(sg):
            m = sg == s
            print(f'   in-sample {sname} seg {s}: ' + ' '.join(f'w{w}:{ND[i, m].mean():.5f}' for i, w in enumerate(grid)))
