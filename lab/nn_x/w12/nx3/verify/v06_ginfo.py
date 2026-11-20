"""nx3 verifier v06 (e) G-info on the INSTALLED files, with the lens of record (reports/INPUTS_1116/residual_rows/{lens,condlib}.py imported unchanged:
score test of one more column given pool + the control block of the ranker's own inputs).  Orchestration is my own.
  holdout rows (lens row order D['rows'] = flat train rows of the 30,000 holdout feeds): nn/feat_cache/train/nx_rp3_{z,rk}.npy, NaN -> 0 on the has = 0 feeds
  val rows: the stale-val twins nn_x/w12/nx3/staleval/nx_rp3_{z,rk}.npy (gate) and the installed natural val files (information)
Two readings of "the family's z or rank form":  (i) the two files scored as entered;  (ii) BUILD_SPEC C7 "for each column", usage example p05: lens.forms(column) -> z form and rank form.
Control of the instrument: the stored FRESH c09_hold rp3b05 on all 30,000 holdout feeds must read the value on record (+8.20 / +9.75).
C6: 20 within-feed permutations of the z file (seeds 3000-3019 holdout, 4000-4019 twin: the seeds of the critic's advance read and of the builder)."""
from v_common import *
P = Log(f'{OUT}/v06_ginfo.txt')
sys.path.insert(0, ROOT + '/reports/INPUTS_1116/residual_rows')
_a = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
import lens, condlib
sys.argv = _a
import hashlib
P(f'lens md5 {hashlib.md5(open(ROOT + "/reports/INPUTS_1116/residual_rows/lens.py", "rb").read()).hexdigest()}, condlib md5 {hashlib.md5(open(ROOT + "/reports/INPUTS_1116/residual_rows/condlib.py", "rb").read()).hexdigest()}; numba cache dir {nb.config.CACHE_DIR}')
FC = ROOT + '/nn/feat_cache'; TW = ROOT + '/nn_x/w12/nx3/staleval'; C9 = ROOT + '/nn_x/insights/ccb_1113/scratch/tester3/cache'
TRf = Split('train', hist=False)
def both(N, D, col, keep=None):
    """reading (ii): forms of the column (NaN = absent); reading (i): the column as entered with NaN -> 0"""
    x = np.array(col, np.float64)
    if keep is not None: x[~keep] = np.nan
    F = lens.forms(x, D, log=False); a = N.score(F['z']); b = N.score(F['rk']); e = N.score(np.nan_to_num(x, nan=0.0).astype(np.float32))
    return a, b, e
# ------------------------------------------------------------------ holdout
D = lens.load_split('hold', 'TO16'); condlib.sp = 'hold'; nm_, cl = condlib.control_block(D); N = condlib.Null(D, nm_, cl)
rows_h = D['rows']; M = D['M']
# my own derivation of the holdout rows: must be the lens's rows
pr = np.load(ROOT + '/eda/labels_position/out/train_prandom.npz'); p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[TRf.M < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((p > 0.9) & (TRf.M >= 50))[0], 30000, replace=False))
myrows = np.repeat(TRf.co[hold] - np.concatenate([[0], np.cumsum(TRf.M[hold])[:-1]]), TRf.M[hold]) + np.arange(int(TRf.M[hold].sum()))
assert (myrows == rows_h).all() and (M == TRf.M[hold]).all()
slot_h = np.repeat(TRf.slot[hold], M)
zt = np.load(f'{FC}/train/nx_rp3_z.npy', mmap_mode='r'); rt = np.load(f'{FC}/train/nx_rp3_rk.npy', mmap_mode='r')
zh = np.asarray(zt[rows_h]).astype(np.float64); rh = np.asarray(rt[rows_h]).astype(np.float64)
P(f'[holdout] pool = 16 held-out wide students; control block {len(nm_)} columns; feeds {D["n"]}, rows {len(rows_h)}; lag-view feeds (slot >= 6) {int((TRf.slot[hold] >= 6).sum())}; NaN rows of the installed column {int(np.isnan(zh).sum())} = rows of slots 0-5 {int((slot_h < 6).sum())}: {bool((np.isnan(zh) == (slot_h < 6)).all())}')
c9h = np.load(C9 + '/c09_hold.npz'); a, b, _ = both(N, D, c9h['rp3b05'])
P(f'  instrument control: stored FRESH rp3b05 on all 30,000 holdout feeds: cz {a["cz"]:+.2f} / {b["cz"]:+.2f} (z / rank form); on record +8.20 / +9.75')
RES = {}
for rn, rm in (('all lag-view feeds (15,225)', slot_h >= 6), ('slots 6-8 (6,946 feeds)', (slot_h >= 6) & (slot_h <= 8)), ('slots 11-13 (8,279 feeds)', slot_h >= 11)):
    az, bz, ez = both(N, D, zh, rm); ar, br, er = both(N, D, rh, rm)
    RES[rn] = (ez, er, az, bz, ar, br)
    P(f'  {rn}: (i) files as entered: nx_rp3_z cz {ez["cz"]:+.2f} (b1 {ez["b1"]:+.4f}, dnats {ez["dnats"]:.5f}, R2 {ez["r2"]:.2f}), nx_rp3_rk cz {er["cz"]:+.2f} (b1 {er["b1"]:+.4f}) | '
      f'(ii) forms of nx_rp3_z: z {az["cz"]:+.2f} / rank {bz["cz"]:+.2f}; forms of nx_rp3_rk: z {ar["cz"]:+.2f} / rank {br["cz"]:+.2f}')
z0 = np.nan_to_num(zh, nan=0.0).astype(np.float32)
pzh = np.array([N.score(condlib.perm_within(z0, D['off'], D['M'], 3000 + s))['cz'] for s in range(20)])
P(f'  C6 holdout: 20 within-feed permutations of the nx_rp3_z file (seeds 3000-3019): max |cz| {np.abs(pzh).max():.2f}, sd {pzh.std():.2f}; values ' + ' '.join(f'{x:+.2f}' for x in pzh))
hz_, hr_ = RES['all lag-view feeds (15,225)'][0], RES['all lag-view feeds (15,225)'][1]
del N, D
# ------------------------------------------------------------------ val
ZT = np.load(TW + '/nx_rp3_z.npy'); RT = np.load(TW + '/nx_rp3_rk.npy'); ZN = np.load(f'{FC}/val/nx_rp3_z.npy'); RN = np.load(f'{FC}/val/nx_rp3_rk.npy')
VAL = {}
for base in ('TO16', 'FILE'):
    D = lens.load_split('val', base); condlib.sp = 'val'; nm_, cl = condlib.control_block(D); N = condlib.Null(D, nm_, cl)
    tz = both(N, D, ZT); tr_ = both(N, D, RT); nz = both(N, D, ZN); nr = both(N, D, RN); VAL[base] = (tz, tr_, nz, nr)
    P(f'[val, base {base}; control block {len(nm_)} columns; feeds {D["n"]}]')
    P(f'  STALE-VAL TWIN (gate): (i) files as entered: nx_rp3_z cz {tz[2]["cz"]:+.2f} (b1 {tz[2]["b1"]:+.4f}, dnats {tz[2]["dnats"]:.5f}, R2 {tz[2]["r2"]:.2f}), nx_rp3_rk cz {tr_[2]["cz"]:+.2f} | (ii) forms of z file: {tz[0]["cz"]:+.2f} / {tz[1]["cz"]:+.2f}; forms of rank file: {tr_[0]["cz"]:+.2f} / {tr_[1]["cz"]:+.2f}')
    P(f'  NATURAL installed val files (information): (i) nx_rp3_z cz {nz[2]["cz"]:+.2f} (b1 {nz[2]["b1"]:+.4f}, dnats {nz[2]["dnats"]:.5f}, R2 {nz[2]["r2"]:.2f}), nx_rp3_rk cz {nr[2]["cz"]:+.2f} | (ii) {nz[0]["cz"]:+.2f} / {nz[1]["cz"]:+.2f}; {nr[0]["cz"]:+.2f} / {nr[1]["cz"]:+.2f}')
    if base == 'TO16':
        pzv = np.array([N.score(condlib.perm_within(ZT.astype(np.float32), D['off'], D['M'], 4000 + s))['cz'] for s in range(20)])
        P(f'  C6 val twin: 20 within-feed permutations of the twin nx_rp3_z file (seeds 4000-4019): max |cz| {np.abs(pzv).max():.2f}, sd {pzv.std():.2f}; values ' + ' '.join(f'{x:+.2f}' for x in pzv))
    del N, D
tT, tF = VAL['TO16'], VAL['FILE']
c7h = max(hz_['cz'], hr_['cz']) >= 2.5; c7v = max(tT[0][2]['cz'], tT[1][2]['cz']) >= 2.5; c7f = max(tF[0][2]['cz'], tF[1][2]['cz']) >= 2.5
c6 = bool(np.abs(pzh).max() < 3.0 and np.abs(pzv).max() < 3.0)
P('=== G-info re-derived (bar: sign + and |cz| >= 2.5 for the z or the rank form, on the lag-view holdout rows AND on val with the stale-val twin; no permutation reaches 3.0)')
P(f'  C7 holdout lag view: nx_rp3_z {hz_["cz"]:+.2f} / nx_rp3_rk {hr_["cz"]:+.2f} -> {"ok" if c7h else "FAIL"}   [builder +4.98 / +4.32; critic advance +5.02 / +4.43]')
P(f'  C7 val twin, base TO16: {tT[0][2]["cz"]:+.2f} / {tT[1][2]["cz"]:+.2f} -> {"ok" if c7v else "FAIL"}   [builder +6.46 / +4.78; critic +6.64 / +4.90]')
P(f'  C7 val twin, base FILE: {tF[0][2]["cz"]:+.2f} / {tF[1][2]["cz"]:+.2f} -> {"ok" if c7f else "FAIL"}   [builder +6.51 / +4.79; critic +6.70 / +4.92]')
P(f'  C6: holdout max {np.abs(pzh).max():.2f}, twin max {np.abs(pzv).max():.2f} -> {"ok" if c6 else "FAIL"}   [builder 1.78 / 1.63]')
P(f'  all six gate cz >= 2.5: {min(hz_["cz"], hr_["cz"], tT[0][2]["cz"], tT[1][2]["cz"], tF[0][2]["cz"], tF[1][2]["cz"]) >= 2.5}')
P(f'  J2 (no bar) b1 of nx_rp3_z: holdout lag view {hz_["b1"]:.4f} (slots 6-8 {RES["slots 6-8 (6,946 feeds)"][0]["b1"]:.4f}, slots 11-13 {RES["slots 11-13 (8,279 feeds)"][0]["b1"]:.4f}); val natural {tT[2][2]["b1"]:.4f}; val twin {tT[0][2]["b1"]:.4f}')
P(f'  G-info: {"PASS" if (c7h and c7v and c7f and c6) else "FAIL"}')
P('DONE')
