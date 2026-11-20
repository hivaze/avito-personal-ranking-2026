"""nx3 verifier v02 (a) LEAK: the INSTALLED arrays (as the trainer reads them: nn/feat_cache/<split>/nx_rp3_{z,rk}.npy; twins nn_x/w12/nx3/staleval/)
against my first-principles walk (out/v01_*.npz).  For every recomputed feed the entered forms are rebuilt from my raw values (my_forms, own code)
with MY K (1 / median non-zero rp3b05 of all val rows, from my val recomputation) and compared row by row.
Leak counts per split (must all be 0):
  NOT_ADMISSIBLE  rows whose installed value differs from the admissible-only recomputation V0 (tolerance 2e-5 on z, 1e-6 on rank)
  of them explained by  V1 (own user's sequences admitted) / V2 (sparse source at or after the row's time admitted) / V3 (user 0 linked)
Sensitivity (the check can fail): rows whose V1 / V2 / V3 forms differ from the V0 forms.  usage: v02_compare.py [suffix of the train npz]"""
from v_common import *
TAG = sys.argv[1] if len(sys.argv) > 1 else ''
P = Log(f'{OUT}/v02_compare{TAG}.txt')
FC = ROOT + '/nn/feat_cache'; DF = ROOT + '/data/feats'; TW = ROOT + '/nn_x/w12/nx3/staleval'; RAW = ROOT + '/nn_x/w12/nx3/out/raw'
TZ, TR_ = 2e-5, 1e-6
V = np.load(f'{OUT}/v01_natural_val.npz')
assert (V['rows'] == np.arange(NROWS['val'])).all()
x = V['rp3'][0]; K = 1.0 / np.median(x[x > 0]); Kb = np.load(RAW + '/K.npy')
P(f'K: mine (1 / median non-zero V0 rp3b05 of all {len(x)} val rows) = {K:.6e}; builder out/raw/K.npy = {Kb[0]:.6e}; relative difference {abs(K - Kb[0]) / Kb[0]:.2e}')


def offs(M):
    return np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)


def cmp(tag, npz, zpath, rpath, braw=None, groups=None, aligned=False):
    d = np.load(npz); rows = d['rows']; M = d['M']; off = offs(M); q = d['q']
    zi = np.load(zpath, mmap_mode='r'); ri = np.load(rpath, mmap_mode='r')
    o = np.argsort(rows, kind='stable'); zin = np.empty(len(rows), np.float32); rin = np.empty(len(rows), np.float32)
    zin[o] = np.asarray(zi[rows[o]]); rin[o] = np.asarray(ri[rows[o]])
    P(f'=== {tag}: {len(M)} feeds, {len(rows)} rows; installed files {zpath.replace(ROOT + "/", "")} ({zi.shape[0]} rows, {zi.dtype}), {rpath.replace(ROOT + "/", "")}')
    P(f'    NaN among the recomputed rows: z {int(np.isnan(zin).sum())}, rank {int(np.isnan(rin).sum())}')
    F = [my_forms(d['rp3'][v], off, M, K) for v in range(NV)]
    z0, r0 = F[0]
    bz = np.abs(zin.astype(np.float64) - z0) > TZ; br = np.abs(rin.astype(np.float64) - r0) > TR_; bad = bz | br | np.isnan(zin) | np.isnan(rin)
    fb = np.bincount(q[bad], minlength=len(M)) > 0
    P(f'    INSTALLED vs V0 (admissible sources only): max |dz| {np.nanmax(np.abs(zin - z0)):.2e}, max |drank| {np.nanmax(np.abs(rin - r0)):.2e}; rows differing: z {int(bz.sum())}, rank {int(br.sum())}; '
      f'NOT_ADMISSIBLE rows {int(bad.sum())} in {int(fb.sum())} feeds')
    nzi = zin > np.minimum.reduceat(zin, off)[q]          # installed non-zero pattern: value above the feed minimum
    nz0 = d['p3a'][0] > 0
    P(f'    zero pattern: installed "above the feed minimum" vs my walk > 0: mismatches {int((nzi != nz0).sum())} (installed non-zero {nzi.mean():.4f}, mine {nz0.mean():.4f})')
    names = ['', 'V1 own user admitted', 'V2 sparse at/after own time', 'V3 user 0 linked']
    for v in (1, 2, 3):
        zv, rv = F[v]
        dv = (np.abs(zv - z0) > TZ) | (np.abs(rv - r0) > TR_); expl = bad & (np.abs(zin - zv) <= TZ) & (np.abs(rin - rv) <= TR_)
        eqv = (np.abs(zin - zv) <= TZ) & (np.abs(rin - rv) <= TR_) & dv
        P(f'    {names[v]}: its forms differ from V0 on {int(dv.sum())} rows / {int((np.bincount(q[dv], minlength=len(M)) > 0).sum())} feeds (raw walk differs on {int((d["p3a"][v] != d["p3a"][0]).sum())} rows) [sensitivity]; '
          f'installed equals this variant where it differs from V0: {int(eqv.sum())} rows; NOT_ADMISSIBLE rows explained by it: {int(expl.sum())}')
    if braw is not None:
        b = braw if aligned else braw[rows]
        for j, k in ((0, 'rp3'), (1, 'p3a')):
            a = d[k][0]; bb = b[:, j]
            rel = np.abs(a - bb) / np.maximum(np.abs(bb), 1e-300); rel[(a == 0) & (bb == 0)] = 0
            P(f'    (information) my V0 {k} vs the builder\'s raw array: max relative difference {rel.max():.3e}; bit-equal rows {np.mean(a == bb):.6f}; zero-pattern mismatches {int(((a > 0) != (bb > 0)).sum())}')
    if groups is not None:
        for gn, gm in groups(d).items():
            rm = gm[q]
            P(f'    by group {gn}: feeds {int(gm.sum())}, rows {int(rm.sum())}, NOT_ADMISSIBLE rows {int(bad[rm].sum())}, non-zero share {nz0[rm].mean():.4f}')
    return int(bad.sum()), len(rows), len(M)


RES = {}
RES['val'] = cmp('VAL natural (all feeds)', f'{OUT}/v01_natural_val.npz', FC + '/val/nx_rp3_z.npy', FC + '/val/nx_rp3_rk.npy', np.load(RAW + '/val_nat.npy'))
pub = None


def tgroups(d):
    return {'public': d['q_ts'] < PUB_TS, 'private': d['q_ts'] >= PUB_TS, 'user 0': d['q_user'] == 0}


RES['test'] = cmp('TEST natural (all feeds)', f'{OUT}/v01_natural_test.npz', FC + '/test/nx_rp3_z.npy', FC + '/test/nx_rp3_rk.npy', np.load(RAW + '/test_nat.npy'), tgroups)
if os.path.exists(f'{OUT}/v01_twin_twin.npz'):
    RES['twin'] = cmp('VAL stale-val twin (all feeds)', f'{OUT}/v01_twin_twin.npz', TW + '/nx_rp3_z.npy', TW + '/nx_rp3_rk.npy', np.load(RAW + '/val_twin.npy'))


def trgroups(d):
    g = {}
    for s in np.unique(d['slot']):
        g[f'slot {s} (Sep {6 + s})'] = d['slot'] == s
    for c in np.unique(d['q_cut']):
        g[f'cut {c}'] = d['q_cut'] == c
    g['holdout feeds'] = d['hold']; g['not holdout'] = ~d['hold']; g['user 0'] = d['q_user'] == 0
    g['pool-like (p_random > 0.5, M >= 50, not holdout)'] = (d['prand'] > 0.5) & (d['M'] >= 50) & ~d['hold']
    return g


# builder's raw train array is stored for the has = 1 rows only, in flat order: map my rows to it
trn = f'{OUT}/v01_train_train{TAG}.npz'
d = np.load(trn); TRf = Split('train', hist=False); has = TRf.slot >= 6; hasrow_off = np.concatenate([[0], np.cumsum(np.where(has, TRf.M, 0))[:-1]])
bidx = np.repeat(hasrow_off[d['qf']] - offs(d['M']), d['M']) + np.arange(len(d['rows']))
braw = np.load(RAW + '/train_lag_hasrows.npy', mmap_mode='r'); o = np.argsort(bidx, kind='stable'); bsel = np.empty((len(bidx), 2)); bsel[o] = np.asarray(braw[bidx[o]])
RES['train'] = cmp('TRAIN lag view (sample)', trn, FC + '/train/nx_rp3_z.npy', FC + '/train/nx_rp3_rk.npy', bsel, trgroups, aligned=True)
P('=== SUMMARY (a): split | feeds | rows | NOT_ADMISSIBLE rows (installed value not reproduced from admissible sources alone)')
for k, (b, r, f) in RES.items():
    P(f'  {k} | {f} | {r} | {b}')
P('  audit counts of my own V0 pairs [admitted, dense, sparse, dense slot >= cut, sparse ts >= own, own non-zero user, own feed, any ts >= own]:')
for k, fn in (('val', 'v01_natural_val'), ('test', 'v01_natural_test'), ('twin', 'v01_twin_twin'), ('train', f'v01_train_train{TAG}')):
    if os.path.exists(f'{OUT}/{fn}.npz'):
        P(f'    {k}: {np.load(f"{OUT}/{fn}.npz")["cnt"].tolist()}')
P('DONE')
