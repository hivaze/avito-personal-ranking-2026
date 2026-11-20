"""CRITIC c03: the STEP 0 gates of RULE_LEDGER.txt read IN ADVANCE on the critic's builds (c01_walk.npz, c02_bt.npz), for the walk (W) and the
click strength (T).  Part 1 (label-free): G-parity = C3 coverage gap by listing-age bucket (dZ, largest bucket; hz2 gate.py definition),
C5 non-default share by test day, C4 PSI of the entered forms.  Part 2: G-info = the calibrated lens of inventory B (condlib.Null: score test given
pool + 84 own inputs) on the lag-built holdout rows and on the stale-val twin, next to the fresh / natural column on the SAME feeds, with placebo.
'Test-like train rows' here = the holdout feeds of slots 11-13 (p_random > 0.9, M >= 50) instead of the p_random > 0.5 pool of BUILD_SPEC C3.
CPU only, read-only; writes critic/out/c03_gates.txt."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
CR = '<repo>/reports/INPUTS_1116/critic'
os.environ['NUMBA_CACHE_DIR'] = CR + '/out/numba_cache'
sys.dont_write_bytecode = True
import numba  # noqa
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows'); sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
sys.argv = [sys.argv[0], 'val', 'TO16']
from lens import *
import condlib
import hz2_common as hc
C9 = ROOT / 'nn_x/insights/ccb_1113/scratch/tester3/cache'
out = open(CR + '/out/c03_gates.txt', 'w')
def Pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.write(s + '\n'); out.flush()
W = np.load(CR + '/out/c01_walk.npz'); B = np.load(CR + '/out/c02_bt.npz'); c9h = np.load(C9 / 'c09_hold.npz'); c6h = np.load(C9 / 'c06_hold.npz')
kn = hc.knots()
# ------------------------------------------------------------------ Part 1: label-free parity
ft, hold = hold_feeds(); Mh = ft['M'].to_numpy().astype(np.int64)[hold]; co = ft['cand_off'].to_numpy().astype(np.int64)[hold]
offh = np.concatenate([[0], np.cumsum(Mh)[:-1]]); rows_h = np.repeat(co - offh, Mh) + np.arange(int(Mh.sum()))
slot_h = W['hold_slot'].astype(np.int64); f4h = np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')[rows_h]); bh = hc.age_bucket(hc.age_hours(f4h, kn))
fv = pl.read_parquet(FLAT / 'val/feeds.parquet'); fe = pl.read_parquet(FLAT / 'test/feeds.parquet')
Mv = fv['M'].to_numpy().astype(np.int64); Me = fe['M'].to_numpy().astype(np.int64)
vslot = np.repeat(hc.day_of(fv['timestamp'].to_numpy()) - hc.DAY0, Mv); eslot = np.repeat(hc.day_of(fe['timestamp'].to_numpy()) - hc.DAY0, Me)
epub = np.repeat(fe['timestamp'].to_numpy().astype(np.int64) < hc.PUB_TS, Me); ebig = np.repeat(Me >= 50, Me); vbig = np.repeat(Mv >= 50, Mv)
bv = hc.age_bucket(hc.age_hours(np.load(FLAT / 'val/cand_f4.npy'), kn)); be = hc.age_bucket(hc.age_hours(np.load(FLAT / 'test/cand_f4.npy'), kn))
def gap(view_nz, view_b, ref_nz, ref_b):
    w = np.bincount(view_b, minlength=7) / len(view_b)
    zv = np.array([view_nz[view_b == k].mean() if (view_b == k).any() else 0.0 for k in range(7)]); zr = np.array([ref_nz[ref_b == k].mean() if (ref_b == k).any() else 0.0 for k in range(7)])
    g = np.abs(zv - zr); big = np.where(w >= 0.05, g, 0.0)
    return float((w * g).sum()), float(big.max()), hc.AGE_NAMES[int(big.argmax())], zv, zr, w
Pr('=== PART 1. G-parity (label-free). C3: non-default share by listing-age bucket; bars dZ <= 0.05 and largest bucket <= 0.10 ===')
Pr('age buckets:', hc.AGE_NAMES)
cols = {'W non-zero (rp3b05 > 0)': (W['hold_lag'][:, 1] > 0, W['val_nat'][:, 1] > 0, W['test_nat'][:, 1] > 0, c9h['p3a'] > 0, W['val_twin'][:, 1] > 0),
        'T seen': (B['hold_lag_seen'] > 0, B['val_nat_seen'] > 0, B['test_nat_seen'] > 0, c6h['seen_l3'] > 0, B['val_twin_seen'] > 0)}
for nm, (hl, vn, en, hf, vt) in cols.items():
    Pr(f'--- {nm}')
    tl = slot_h >= 11; vl = (slot_h >= 6) & (slot_h <= 8)
    for lab, vmask, ref_nz, ref_b in (('test-like train rows (holdout slots 11-13, lag view) vs PRIVATE rows (M >= 50)', tl, en[~epub & ebig], be[~epub & ebig]),
                                      ('test-like train rows (holdout slots 11-13, lag view) vs PUBLIC rows (M >= 50)', tl, en[epub & ebig], be[epub & ebig]),
                                      ('val-like train rows (holdout slots 6-8, lag view) vs VAL rows (M >= 50)', vl, vn[vbig], bv[vbig])):
        dZ, big, bn, zv, zr, w = gap(hl[vmask], bh[vmask], ref_nz, ref_b)
        Pr(f'  {lab}: share {hl[vmask].mean():.4f} vs {ref_nz.mean():.4f}; dZ {dZ:.4f} ({"ok" if dZ <= 0.05 else "FAIL"}), largest bucket {big:.4f} at {bn} ({"ok" if big <= 0.10 else "FAIL"})')
        Pr('     view by bucket ' + ' '.join(f'{x:.3f}' for x in zv) + ' | ref ' + ' '.join(f'{x:.3f}' for x in zr) + ' | view row share ' + ' '.join(f'{x:.3f}' for x in w))
    # for scale: the FRESH (cut = own day) column of the same holdout rows against the same references, and the stale-val twin against private
    dZ, big, bn, zv, zr, w = gap(hf[tl], bh[tl], en[~epub & ebig], be[~epub & ebig]); Pr(f'  for scale, FRESH column (cut = own day) on holdout slots 11-13 vs PRIVATE: share {hf[tl].mean():.4f}; dZ {dZ:.4f}, largest bucket {big:.4f}')
    dZ, big, bn, zv, zr, w = gap(vt[vbig], bv[vbig], en[~epub & ebig], be[~epub & ebig]); Pr(f'  stale-val twin (val M >= 50) vs PRIVATE: share {vt[vbig].mean():.4f} vs {en[~epub & ebig].mean():.4f}; dZ {dZ:.4f}, largest bucket {big:.4f} at {bn}')
    Pr('  C5 non-default share by day: val slot 16 / 18: ' + ' / '.join(f'{vn[vslot == s].mean():.4f}' for s in (16, 18)) + ' | test slot 20 / 21 / 22 / 23: '
       + ' / '.join(f'{en[eslot == s].mean():.4f}' for s in (20, 21, 22, 23)) + f' | public {en[epub].mean():.4f} private {en[~epub].mean():.4f}'
       + ' | holdout lag view slot 6 / 7 / 8 / 11 / 12 / 13: ' + ' / '.join(f'{hl[slot_h == s].mean():.4f}' for s in (6, 7, 8, 11, 12, 13)))
# entered forms and PSI (C4): nx_rp3_z = within-feed z of log1p(K * rp3b05), zeros are values, clip 5; nx_rp3_rk = within-feed mid-rank - 0.5
def fz(x, M):
    off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64); return feed_z(np.asarray(x, np.float64), off, M, 5.0), feed_rk(np.asarray(x, np.float64), off, M)
Kr = 1.0 / np.median(W['val_nat'][:, 0][W['val_nat'][:, 0] > 0]); Kp = 1.0 / np.median(W['val_nat'][:, 1][W['val_nat'][:, 1] > 0])
Pr(f'K (rp3b05) {Kr:.4e}, K (p3a) {Kp:.4e}')
hasrow = slot_h >= 6
def psi(a, b, edges):
    pa = np.bincount(np.searchsorted(edges, a, side='right'), minlength=len(edges) + 1) / len(a); pb = np.bincount(np.searchsorted(edges, b, side='right'), minlength=len(edges) + 1) / len(b)
    pa = np.maximum(pa, 1e-6); pb = np.maximum(pb, 1e-6); return float(((pa - pb) * np.log(pa / pb)).sum())
forms_ = {}
for nm, hl, vn, en, vt, K in (('nx_rp3', W['hold_lag'][:, 0], W['val_nat'][:, 0], W['test_nat'][:, 0], W['val_twin'][:, 0], Kr), ('nx_p3a', W['hold_lag'][:, 1], W['val_nat'][:, 1], W['test_nat'][:, 1], W['val_twin'][:, 1], Kp),
                              ('nx_bt', B['hold_lag_theta'], B['val_nat_theta'], B['test_nat_theta'], B['val_twin_theta'], None)):
    tr = (lambda x: np.log1p(K * x)) if K else (lambda x: x)
    zh, rh = fz(tr(np.where(hasrow, hl, 0.0)), Mh); zv, rv = fz(tr(vn), Mv); ze, re_ = fz(tr(en), Me); zt, rt = fz(tr(vt), Mv)
    forms_[nm] = dict(zh=zh, rh=rh, zv=zv, rv=rv, zt=zt, rt=rt)
    for fn, a_h, a_v, a_e in (('_z', zh, zv, ze), ('_rk', rh, rv, re_)):
        ed = np.unique(np.quantile(a_v, np.linspace(0.1, 0.9, 9)))
        Pr(f'  C4 PSI {nm}{fn} (bar 0.25): train has = 1 (holdout lag view) vs val {psi(a_h[hasrow], a_v, ed):.4f}, vs public {psi(a_h[hasrow], a_e[epub], ed):.4f}, vs private {psi(a_h[hasrow], a_e[~epub], ed):.4f}; '
           f'test-like slots 11-13 vs private {psi(a_h[slot_h >= 11], a_e[~epub], ed):.4f}; val vs private {psi(a_v, a_e[~epub], ed):.4f}')
out.flush()
# ------------------------------------------------------------------ Part 2: G-info with the calibrated lens
Pr('\n=== PART 2. G-info: calibrated lens (score test given pool + own inputs), z form / rank form; bar |cz| >= 2.5 with the sign on record ===')
def nan_out(x, keep_rows):
    y = np.array(x, np.float64); y[~keep_rows] = np.nan; return y
def sc(N, x, D, log=False):
    F = forms(np.asarray(x, np.float64), D, log=log)
    if F['varshare'] < 0.001: return None
    a = N.score(F['z']); b = N.score(F['rk']); return a['cz'], b['cz'], a['dnats'], a['r2'], F
# holdout
D = load_split('hold', 'TO16'); condlib.sp = 'hold'; nm_, cl = condlib.control_block(D); N = condlib.Null(D, nm_, cl)
assert (D['rows'] == rows_h).all()
reg = {'all lag-view feeds (slots 6-13; 15,225)': slot_h >= 6, 'val-like (slots 6-8; 6,946)': (slot_h >= 6) & (slot_h <= 8), 'test-like (slots 11-13; 8,279)': slot_h >= 11}
Pr('[holdout, pool = 16 held-out wide students] column | feeds | FRESH (cut = own day) cz z / rank (dnats) | LAG VIEW cz z / rank (dnats) | kept share of cz (z / rank)')
pz = []
for cn, fresh, lagc in (('rp3b05 (raw, as inventory B)', c9h['rp3b05'], W['hold_lag'][:, 0]), ('p3a (raw)', c9h['p3a'], W['hold_lag'][:, 1]),
                        ('nx_rp3 = log1p(K rp3b05) (entered form)', np.log1p(Kr * c9h['rp3b05']), np.log1p(Kr * np.nan_to_num(W['hold_lag'][:, 0]))),
                        ('bt theta', c6h['theta_l3'], B['hold_lag_theta']), ('bt seen', c6h['seen_l3'], B['hold_lag_seen'])):
    for rn, rm in reg.items():
        a = sc(N, nan_out(fresh, rm), D); b = sc(N, nan_out(lagc, rm), D)
        Pr(f'  {cn:40s} | {rn:40s} | {a[0]:+6.2f} / {a[1]:+6.2f} ({a[2]:.5f}) | {b[0]:+6.2f} / {b[1]:+6.2f} ({b[2]:.5f}) | {b[0] / a[0]:.2f} / {b[1] / a[1]:.2f}')
        if cn.startswith('rp3b05') and rn.startswith('all'):
            for s in range(20): pz.append(N.score(condlib.perm_within(b[4]['z'], D['off'], D['M'], 3000 + s))['cz'])
a = sc(N, c9h['rp3b05'], D); Pr(f'  check: rp3b05 fresh on ALL 30,000 holdout feeds: {a[0]:+.2f} / {a[1]:+.2f} (record +8.20 / +9.75)')
pz = np.array(pz); Pr(f'  placebo: 20 within-feed permutations of the lag-view rp3b05 z column: sd {pz.std():.2f}, max |cz| {np.abs(pz).max():.2f}')
# what the lag view and the natural test-row build share: on test-like holdout rows, agreement of lag-view and fresh within-feed ranks
tl = slot_h >= 11; fr = forms(nan_out(c9h['rp3b05'], tl), D, log=False)['rk'][tl]; lr = forms(nan_out(W['hold_lag'][:, 0], tl), D, log=False)['rk'][tl]
Pr(f'  test-like holdout rows: correlation of within-feed rank, lag view vs fresh: {np.corrcoef(fr, lr)[0, 1]:.3f}')
del N, D
# val
for base in ('TO16', 'FILE'):
    D = load_split('val', base); condlib.sp = 'val'; nm_, cl = condlib.control_block(D); N = condlib.Null(D, nm_, cl)
    Pr(f'[val, base {base}] column | NATURAL (all train + earlier val) cz z / rank (dnats) | STALE-VAL TWIN (cut = slot - L) cz z / rank (dnats) | kept share')
    allv = np.ones(len(D['y']), bool)
    for cn, nat, tw in (('rp3b05 (raw)', W['val_nat'][:, 0], W['val_twin'][:, 0]), ('p3a (raw)', W['val_nat'][:, 1], W['val_twin'][:, 1]),
                        ('nx_rp3 = log1p(K rp3b05)', np.log1p(Kr * W['val_nat'][:, 0]), np.log1p(Kr * W['val_twin'][:, 0])),
                        ('bt theta', B['val_nat_theta'], B['val_twin_theta']), ('bt seen', B['val_nat_seen'], B['val_twin_seen'])):
        a = sc(N, nat, D); b = sc(N, tw, D)
        Pr(f'  {cn:30s} | {a[0]:+6.2f} / {a[1]:+6.2f} ({a[2]:.5f}) | {b[0]:+6.2f} / {b[1]:+6.2f} ({b[2]:.5f}) | {b[0] / a[0]:.2f} / {b[1] / a[1]:.2f}')
        if base == 'TO16' and cn.startswith('rp3b05'):
            for vs in (16, 18):
                mk = vslot == vs; a2 = sc(N, nan_out(nat, mk), D); b2 = sc(N, nan_out(tw, mk), D)
                Pr(f'      val slot {vs}: natural {a2[0]:+.2f} / {a2[1]:+.2f} | twin {b2[0]:+.2f} / {b2[1]:+.2f}')
            pz = np.array([N.score(condlib.perm_within(b[4]['z'], D['off'], D['M'], 4000 + s))['cz'] for s in range(20)])
            Pr(f'      placebo: 20 within-feed permutations of the twin z column: sd {pz.std():.2f}, max |cz| {np.abs(pz).max():.2f}')
    del N, D
out.close()
