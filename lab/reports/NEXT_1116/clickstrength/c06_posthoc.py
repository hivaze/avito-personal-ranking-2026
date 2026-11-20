"""Lane A c06: POST HOC information, written and run AFTER the gates were read (c02-c05). Nothing here can change the verdict or promote anything.
 A (label-free) how the new item statistics relate to the record's D5 columns on the same val / test rows (rescue/out/arr: bt5, coec5).
 B (holdout labels) instrument checks of my scoring harness on H: (i) the record's bt5:rk array must read the record's +4.17;
   (ii) positive control of the STATISTIC: the same two entered values built from the three dense days up to and INCLUDING the row's own day (hold__*_LEAK; the holdout
   feed itself is never a source, but same-day feeds are, later ones included: NOT admissible as an input, never built for training). It shows what the exposure-free
   click part reads when it is fresh, i.e. whether the weak reads of c03 come from the statistic or from the lag.
usage: c06_posthoc.py   output out/c06_posthoc.txt"""
from cs_common import *
lens, condlib = load_lens()
P = Log(OUT / 'c06_posthoc.txt')
V = Views(None)
RARR = ROOT / 'reports/INPUTS_1116/round2/rescue/out/arr'
P('=== A. label-free: Pearson correlation of the new item statistics with the D5 columns of record on the same rows ===')
for q in ('val_nat', 'test_nat'):
    bt = np.load(RARR / f'{q}__bt5.npy').astype(np.float64); co = np.log(np.load(RARR / f'{q}__coec5.npy').astype(np.float64)); seen5 = np.load(RARR / f'{q}__bt5_seen.npy') > 0; imp5 = np.load(RARR / f'{q}__imp5.npy') > 0
    xx = get(q, 'cs3x_x'); xa = get(q, 'cs3a_x'); sx = get(q, 'cs3x_e') > 0; sa = get(q, 'cs3a_e') > 0
    P(f'  {q}: corr(cs3x_x, bt5 theta) all rows {np.corrcoef(xx, bt)[0, 1]:.3f}, rows seen by both {np.corrcoef(xx[sx & seen5], bt[sx & seen5])[0, 1]:.3f} | corr(cs3a_x, log coec5) all rows {np.corrcoef(xa, co)[0, 1]:.3f}, rows seen by both {np.corrcoef(xa[sa & imp5], co[sa & imp5])[0, 1]:.3f} | '
      f'seen share: exploration 3-day {sx.mean():.4f} against 5-day {seen5.mean():.4f}; all feeds 3-day {sa.mean():.4f} against 5-day {imp5.mean():.4f} | corr(cs3x_x, cs3a_x) {np.corrcoef(xx, xa)[0, 1]:.3f}')
P('=== B. holdout: instrument checks (null model = pool + control block + nx3, as c03) ===')
D = lens.load_split('hold', 'TO16'); condlib.sp = 'hold'; cn, cl = condlib.control_block(D); off, M = D['off'], D['M']; rows = V.rows('hold')[0]
assert (D['rows'] == rows).all()
nz = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0); nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
N = condlib.Null(D, cn + ['nx_rp3_z', 'nx_rp3_rk'], cl + [nz, nr])
b = np.load(RARR / 'hold__bt5.npy').astype(np.float64)
P(f'  (i) the record\'s D5 array bt5 in my harness: rank form cz {N.score(lens.feed_rk(b, off, M))["cz"]:+.2f} (record +4.17), z form {N.score(lens.feed_z(b, off, M, 5.0))["cz"]:+.2f} (record +0.98)')
slot = V.row_slot('hold')
for col, sc in (('cs3a_c', 25.0), ('cs3x_c', 300.0)):
    rec = (np.nan_to_num(get('hold', col)) * sc).astype(np.float32); x = (np.nan_to_num(get('hold', col + '_LEAK')) * sc).astype(np.float32)
    r = N.score(x); p0 = np.array([float(N.score(condlib.perm_within(x, off, M, 9000 + s))['cz']) for s in range(20)])
    P(f'  (ii) {col}: build of record (lag 3-10 d) cz {N.score(rec)["cz"]:+.2f} | SAME statistic from the three dense days up to and including the row\'s own day (not admissible): cz {r["cz"]:+.2f}, nats per feed {r["dnats"]:.6f}; '
      f'slots 6-8 {N.score(np.where(slot <= 8, x, 0).astype(np.float32))["cz"]:+.2f}, slots 11-13 {N.score(np.where(slot >= 11, x, 0).astype(np.float32))["cz"]:+.2f}; 20 within-feed permutations sd {p0.std():.2f}, max {np.abs(p0).max():.2f}')
P('DONE')
