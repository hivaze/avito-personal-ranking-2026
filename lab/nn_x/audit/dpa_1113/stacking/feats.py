"""Second-level input matrices. form 'full' = ft+to inputs (val/test), 'to' = train-only inputs (val/holdout/test)."""
from lib import *

def feed_feats(sp):
    f = np.stack([np.log(sp.M), np.log1p(sp.H), sp.u0.astype(np.float64), sp.cshare], 1).astype(np.float32)
    return f[sp.fid]

def cand_feats(sp):
    return np.stack([sp.cold.astype(np.float32), feed_pct(sp.off, sp.M, np.ascontiguousarray(sp.f1, dtype=np.float32))], 1)

def pool_score(sp, form):
    if form == 'full':
        return np.load(CACHE / f'{sp.name}_g3.npy')
    if sp.name == 'hold':
        return np.load(CACHE / 'hold_pool_to.npy')
    if sp.name == 'val' and form == 'to':
        return np.load(CACHE / 'val_pool_to.npy')
    if sp.name == 'val' and form == 'ftas':
        return np.load(CACHE / 'val_pool_ft.npy')
    if form == 'ftas':
        return sp.z(np.asarray(sp.Z('ft')).mean(0).astype(np.float32))
    return sp.z(np.asarray(sp.Z('to')).mean(0).astype(np.float32))

def build(sp, arm, form):
    """arm: B0 [pool]; B1 [pool, pct, feed, cand]; B2 [pool, pct, members, fam, sd, feed]; B3 = B2 + cand."""
    pool = pool_score(sp, form).astype(np.float32)
    cols = [pool[:, None]]; names = ['pool']
    if arm == 'B0':
        return np.hstack(cols), names, pool
    cols.append(feed_pct(sp.off, sp.M, pool)[:, None]); names.append('pool_pct')
    if arm in ('B2', 'B3'):
        if form == 'full':
            Zm = 0.5 * np.asarray(sp.Z('ft')) + 0.5 * np.asarray(sp.Z('to'))
            fam = [np.asarray(sp.Z(h))[FAMIDX == i].mean(0) for h in ('ft', 'to') for i in range(3)]
            fnames = [f'fam{i}_{h}' for h in ('ft', 'to') for i in range(3)]
        else:
            Zm = np.asarray(sp.Z('ft' if form == 'ftas' else 'to'))
            fam = [Zm[FAMIDX == i].mean(0) for i in range(3)]
            fnames = [f'fam{i}_to' for i in range(3)]
        cols.append(Zm.T.astype(np.float32)); names += [f'm{j}' for j in range(Zm.shape[0])]
        cols.append(np.stack(fam, 1).astype(np.float32)); names += fnames
        cols.append(Zm.std(0)[:, None].astype(np.float32)); names.append('msd')
    cols.append(feed_feats(sp)); names += ['logM', 'logH', 'u0', 'cshare']
    if arm in ('B1', 'B3'):
        cols.append(cand_feats(sp)); names += ['cold', 'f1pct']
    return np.hstack(cols).astype(np.float32), names, pool
