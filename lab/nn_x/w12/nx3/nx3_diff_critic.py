"""nx3 STEP 0: the difference between the BUILD OF RECORD (BUILD_SPEC 2: sparse members admitted by TIMESTAMP, deg = DENSE posting count with floor 1)
and the critic's advance build (reports/INPUTS_1116/critic/out/c01_walk.npz: sparse members admitted by DAY, deg counts the admitted sparse members, as c09_build).
Label-free. On the 15,225 lag-view holdout feeds the difference is split in two steps with THIS builder's driver:
   (crit) day admission, deg with sparse   -> must reproduce the critic's array exactly (driver cross-check)
   (mid)  day admission, deg dense floor 1 -> effect of the degree rule
   (rec)  timestamp admission, deg dense floor 1 = the build of record -> effect of the admission rule
For the natural val / test rows and the stale-val twin: record against critic, no extra build. Output out/diff_critic.txt (exclusive create)."""
from nx3_common import *
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows')
_argv = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
from lens import feed_z, feed_rk
sys.argv = _argv
P = Log(OUT / f'diff_critic{TAG}.txt')
W = np.load('<repo>/reports/INPUTS_1116/critic/out/c01_walk.npz')
D = Data(P); ft = D.F['train']; At, Av, Ae = D.A['train'], D.A['val'], D.A['test']
Kr, Kp = np.load(RAW_DIR / 'K.npy')
def offs(M): return np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
def cmp_(tag, mine, crit, M, extra=''):
    """mine, crit: raw rp3b05 arrays on the same rows (feeds with sizes M)"""
    o = offs(M); zm, rm = forms_of(mine, o, M, Kr, feed_z, feed_rk); zc, rc = forms_of(crit, o, M, Kr, feed_z, feed_rk)
    eq = np.abs(mine - crit) <= 1e-12 * np.maximum(np.abs(crit), 1e-300)
    P(f'{tag}: rows {len(mine)}; non-zero share {np.mean(mine > 0):.4f} vs critic {np.mean(crit > 0):.4f}; non-zero in mine only {int(((mine > 0) & (crit == 0)).sum())}, in critic only {int(((mine == 0) & (crit > 0)).sum())}; '
      f'rows with equal raw value {eq.mean():.4f}; max abs diff raw {np.abs(mine - crit).max():.3e}; entered forms: corr nx_rp3_z {np.corrcoef(zm, zc)[0, 1]:.5f}, corr nx_rp3_rk {np.corrcoef(rm, rc)[0, 1]:.5f}, '
      f'mean |d z| {np.abs(zm - zc).mean():.4f}, rows with |d z| > 0.1: {np.mean(np.abs(zm - zc) > 0.1):.4f}, mean |d rk| {np.abs(rm - rc).mean():.5f} {extra}')
# ---------------- natural val / test and twin: record vs critic
VN = np.load(RAW_DIR / 'val_nat.npy'); EN = np.load(RAW_DIR / 'test_nat.npy'); VT = np.load(RAW_DIR / 'val_twin.npy')
cmp_('natural VAL rows (critic = stored c09_val: days, deg with earlier val feeds, no test feeds)', VN[:, 0], W['val_nat'][:, 0], Av['M'])
cmp_('natural TEST rows (critic: days, deg with earlier val / test feeds)', EN[:, 0], W['test_nat'][:, 0], Ae['M'])
cmp_('STALE-VAL TWIN (critic: days, deg with the admitted sparse members)', VT[:, 0], W['val_twin'][:, 0], Av['M'])
for sp, mine, crit, a in (('val', VN, W['val_nat'], Av), ('test', EN, W['test_nat'], Ae), ('twin', VT, W['val_twin'], Av)):
    srow = np.repeat(a['slot'], a['M'])
    P(f'  {sp} non-zero share by slot, record / critic: ' + ', '.join(f'slot {s}: {np.mean(mine[srow == s, 0] > 0):.4f} / {np.mean(crit[srow == s, 0] > 0):.4f}' for s in np.unique(a['slot']) if (a['slot'] == s).sum() > 100))
# ---------------- holdout lag view
hold, _p = hc.holdout_feeds(ft); Mh = At['M'][hold]; hoff = offs(Mh); nrow_h = int(Mh.sum())
L = np.load(hc.HERE / 'feats_L/train/hz2_lag.npy').astype(np.int64); tslot = At['slot']; hslot = tslot[hold]; hcut = np.maximum(hslot - L[hold], 5); hhas = hslot >= 6
has = np.load(RAW_DIR / 'train_has_feed.npy'); hasrow = np.repeat(has, At['M']); hrows = np.nonzero(hasrow)[0]; raw_tr = np.load(RAW_DIR / 'train_lag_hasrows.npy')
rows_h = np.repeat(At['co'][hold] - hoff, Mh) + np.arange(nrow_h); lv = np.repeat(hhas, Mh)
rec = np.full(nrow_h, np.nan); rec[lv] = raw_tr[np.searchsorted(hrows, rows_h[lv]), 0]
spf = hc.sparse_flag(At['fid']); crit_my = np.full(nrow_h, np.nan); mid = np.full(nrow_h, np.nan)
def put(dst, qi, o):
    Mq = Mh[qi]; offq = offs(Mq)
    for j, qpos in enumerate(qi): dst[hoff[qpos]:hoff[qpos] + Mq[j]] = o[offq[j]:offq[j] + Mq[j], 0]
for c in (5, 6, 7):
    qi = np.nonzero(hhas & (hcut == c))[0]; dense = c9.source_fidx(ft, int(hc.DAY0 + c)); sp_ = np.nonzero(spf & (tslot >= c))[0]
    S = Src(D, [('train', dense, np.full(len(dense), -1, np.int64), True), ('train', sp_, At['day'][sp_], False)])      # kernel time of sparse members = DAY
    dq = np.unique(At['day'][hold[qi]]); nde = int(S.idx.lens[:S.n_dense].sum()); sde = np.repeat(S.idx.days[S.n_dense:], S.idx.lens[S.n_dense:]); evs = S.idx.ev[nde:]
    deg_c = np.stack([S.deg_dense + np.bincount(evs[sde < d_], minlength=D.G) for d_ in dq])
    o, *_ = run_kernel(D, 'train', S.idx, deg_c, hold[qi], np.searchsorted(dq, At['day'][hold[qi]]), At['day'][hold[qi]]); put(crit_my, qi, o)
    o, *_ = run_kernel(D, 'train', S.idx, S.deg_floor1(), hold[qi], np.zeros(len(qi), np.int64), At['day'][hold[qi]]); put(mid, qi, o)
    del S, deg_c
cw = W['hold_lag'][:, 0]
assert (np.isnan(cw) == ~lv).all() and (np.isnan(crit_my) == ~lv).all()
P(f'driver cross-check: this driver in the critic\'s convention (day admission, deg with sparse) against critic/out/c01_walk.npz hold_lag on {int(lv.sum())} rows: max abs diff {np.abs(crit_my[lv] - cw[lv]).max():.3e}, zero-pattern mismatches {int(((crit_my[lv] > 0) != (cw[lv] > 0)).sum())}')
Ml = Mh[hhas]
cmp_('holdout lag view, step 1 (degree rule): day admission + deg dense floor 1  against the critic', mid[lv], cw[lv], Ml)
cmp_('holdout lag view, step 2 (admission rule): BUILD OF RECORD (timestamp) against day admission + deg dense floor 1', rec[lv], mid[lv], Ml)
cmp_('holdout lag view, total: BUILD OF RECORD against the critic', rec[lv], cw[lv], Ml)
srow = np.repeat(hslot, Mh)
P('  holdout lag view non-zero share by slot, record / day + dense deg / critic: ' + ', '.join(f'slot {s}: {np.mean(rec[srow == s] > 0):.4f} / {np.mean(mid[srow == s] > 0):.4f} / {np.mean(cw[srow == s] > 0):.4f}' for s in (6, 7, 8, 11, 12, 13)))
P('DONE')
