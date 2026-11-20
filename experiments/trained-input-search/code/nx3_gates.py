# CPU gates that the built input had to pass before any GPU run: stored arrays reproduced, no source newer than
# the row, the same share of non-zero values on train rows as on val, public and private rows, and the
# conditional score test of condlib.py on rows built at the test lag.
# The bars were fixed in this file before any gate value was read.
# Output: ../results/walk_input_step0_gates.txt (verdict PASS; score statistic +4.98 on the lag-view holdout
# and +6.46 on the stale twin of val).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/nx3/nx3_gates.py
"""nx3 STEP 0, the gates on the BUILD OF RECORD (files of nx3_build.py), next to the critic's advance values (reports/INPUTS_1116/critic/out/c03_gates.txt,
c05_coef.txt, c06_joint.txt). Rule: ledger block "TRAINED-INPUT TEST nx3" (STEP 0 as amended by A2); definitions: BUILD_SPEC.txt section 5.
Everything below was fixed in this file BEFORE any gate value of the build of record was read (file written while the build was running).

WHAT DECIDES (the rule as written; bars unchanged):
  G-code    C1 = nx3_c1_repro.py (stored c09_val / c09_hold arrays reproduced to 1e-5 relative); C2 = nx3_build.py (every causality count 0)  [read from their logs]
  G-parity  C3  dZ <= 0.05 and largest bucket <= 0.10 for the non-zero share (raw rp3b05 > 0) by the 7 listing-age buckets of hz2_common:
                test-like train rows = POOL feeds (p_random > 0.5, M >= 50, not holdout: BUILD_SPEC C3, the hz2 gate's trPool) of slots 11-13 against PRIVATE rows
                and against PUBLIC rows of test feeds with M >= 50; val-like train rows = pool feeds of slots 6-8 against val rows of feeds with M >= 50.
            C4  PSI <= 0.25 for each of the two files, train has = 1 rows of POOL feeds against val, public, private (all rows); bins of BUILD_SPEC C4:
                10 bins at the val deciles of the non-zero file values + one bin for exact zero.
            C5  non-zero share on every test day (slots 20-23) >= 0.5 x val's (val's = the LARGEST of val overall / slot 16 / slot 18: the strictest reading).
  G-info    C7  score test given pool + own inputs (condlib.Null) of the two FILES as entered (nx_rp3_z = z form, nx_rp3_rk = rank form; NaN -> 0):
                sign + and |cz| >= 2.5 for the z form or the rank form, BOTH on the holdout rows (lag view; pool = 16 held-out wide students) AND on val rows
                with the stale-val twin (base TO16; base FILE is printed and must also pass).
            C6  none of 20 within-feed permutations of the family's z file reaches |cz| 3.0 (holdout lag view: seeds 3000-3019; val twin: seeds 4000-4019, as the critic).
  No bar    J1, J2 of amendment A2.
PRINTED FOR COMPARISON ONLY (they do not decide; a value outside a bar is flagged in the read): the critic's populations and formulas on my files
  (C3 on the holdout feeds; C4 with the critic's 9 decile edges over all val values, on holdout rows, on all has = 1 rows; C7 with forms(log1p(K raw)) by region).
CPU only, 12 threads, read-only. Output nn_x/w12/nx3/out/nx3_gates.txt (exclusive create)."""
from nx3_common import *
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows')
_argv = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
import lens
import condlib
sys.argv = _argv
P = Log(OUT / f'nx3_gates{TAG}.txt')
FC = ROOT / 'nn/feat_cache'
P(f'lens md5 {md5("<repo>/reports/INPUTS_1116/residual_rows/lens.py")}, condlib md5 {md5("<repo>/reports/INPUTS_1116/residual_rows/condlib.py")}')
# ------------------------------------------------------------------ data (label-free part)
F = {s: pl.read_parquet(FLAT / s / 'feeds.parquet') for s in SPLITS}
A = {s: dict(M=F[s]['M'].to_numpy().astype(np.int64), ts=F[s]['timestamp'].to_numpy().astype(np.int64)) for s in SPLITS}
for s in SPLITS: A[s]['slot'] = (c9.day_of(A[s]['ts']) - hc.DAY0).astype(np.int64)
Mt, Mv, Me = A['train']['M'], A['val']['M'], A['test']['M']; tslot = A['train']['slot']
has = np.load(RAW_DIR / 'train_has_feed.npy'); hasrow = np.repeat(has, Mt); hrows = np.nonzero(hasrow)[0]
raw_tr = np.load(RAW_DIR / 'train_lag_hasrows.npy'); assert len(raw_tr) == len(hrows)
nz_tr = np.zeros(NROWS['train'], np.bool_); nz_tr[hrows] = raw_tr[:, 0] > 0
VN = np.load(RAW_DIR / 'val_nat.npy'); EN = np.load(RAW_DIR / 'test_nat.npy'); VT = np.load(RAW_DIR / 'val_twin.npy'); Kr, Kp = np.load(RAW_DIR / 'K.npy')
ZF = {s: np.load(FILES_DIR / s / 'nx_rp3_z.npy') for s in SPLITS}; RF = {s: np.load(FILES_DIR / s / 'nx_rp3_rk.npy') for s in SPLITS}
ZT = np.load(TWIN_DIR / 'nx_rp3_z.npy'); RT = np.load(TWIN_DIR / 'nx_rp3_rk.npy')
assert (np.isnan(ZF['train']) == ~hasrow).all() and (np.isnan(RF['train']) == ~hasrow).all() and not np.isnan(ZF['val']).any() and not np.isnan(ZF['test']).any()
P(f'files: train {ZF["train"].shape} val {ZF["val"].shape} test {ZF["test"].shape} twin {ZT.shape}; K (rp3b05) {Kr:.6e}')
ft = F['train']; hold, prand = hc.holdout_feeds(ft); ish = np.zeros(len(Mt), bool); ish[hold] = True
pool = (prand > 0.5) & (Mt >= 50) & ~ish
P(f'pool feeds (p_random > 0.5, M >= 50, not holdout): {int(pool.sum())}; with has = 1: {int((pool & has).sum())}; slots 6-8: {int((pool & (tslot >= 6) & (tslot <= 8)).sum())}; slots 11-13: {int((pool & (tslot >= 11) & (tslot <= 13)).sum())}')
kn = hc.knots()
vslot = np.repeat(A['val']['slot'], Mv); eslot = np.repeat(A['test']['slot'], Me)
epub = np.repeat(A['test']['ts'] < hc.PUB_TS, Me); ebig = np.repeat(Me >= 50, Me); vbig = np.repeat(Mv >= 50, Mv)
bv = hc.age_bucket(hc.age_hours(np.load(FLAT / 'val/cand_f4.npy'), kn)); be = hc.age_bucket(hc.age_hours(np.load(FLAT / 'test/cand_f4.npy'), kn))
f4t = np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')
def rows_of(feedmask):
    return np.nonzero(np.repeat(feedmask, Mt))[0]
def gap(view_nz, view_b, ref_nz, ref_b):
    w = np.bincount(view_b, minlength=7) / len(view_b)
    zv = np.array([view_nz[view_b == k].mean() if (view_b == k).any() else 0.0 for k in range(7)]); zr = np.array([ref_nz[ref_b == k].mean() if (ref_b == k).any() else 0.0 for k in range(7)])
    g = np.abs(zv - zr); big = np.where(w >= 0.05, g, 0.0)
    return float((w * g).sum()), float(big.max()), hc.AGE_NAMES[int(big.argmax())], zv, zr, w
VER = {}      # gate -> bool
TAB = []      # (gate, bar, mine, critic advance, ok)
# ------------------------------------------------------------------ PART 1: G-parity
P('=== PART 1. G-parity (label-free) ===')
P('age buckets: ' + str(hc.AGE_NAMES))
vn_nz, en_nz, vt_nz = VN[:, 0] > 0, EN[:, 0] > 0, VT[:, 0] > 0
REFS = (('PRIVATE rows (M >= 50)', en_nz[~epub & ebig], be[~epub & ebig]), ('PUBLIC rows (M >= 50)', en_nz[epub & ebig], be[epub & ebig]), ('VAL rows (M >= 50)', vn_nz[vbig], bv[vbig]))
CRIT_C3 = {('hold', 0): (0.0365, 0.0665), ('hold', 1): (0.0383, 0.0695), ('hold', 2): (0.0417, 0.0675)}
c3ok = True
for popn, fm, official in (('POOL feeds [OFFICIAL, BUILD_SPEC C3]', pool, True), ('HOLDOUT feeds [critic\'s population; comparison]', ish, False)):
    P(f'--- C3, train side = {popn}')
    for ri, (regn, lo, hi) in ((0, ('test-like train rows (slots 11-13)', 11, 13)), (1, ('test-like train rows (slots 11-13)', 11, 13)), (2, ('val-like train rows (slots 6-8)', 6, 8))):
        r = rows_of(fm & (tslot >= lo) & (tslot <= hi)); b = hc.age_bucket(hc.age_hours(np.asarray(f4t[r]), kn)); refn, ref_nz, ref_b = REFS[ri]
        dZ, big, bn, zv, zr, w = gap(nz_tr[r], b, ref_nz, ref_b)
        ok = dZ <= 0.05 and big <= 0.10
        crit = CRIT_C3[('hold', ri)]
        P(f'  {regn}, {int((fm & (tslot >= lo) & (tslot <= hi)).sum())} feeds, {len(r)} rows, vs {refn}: share {nz_tr[r].mean():.4f} vs {ref_nz.mean():.4f}; dZ {dZ:.4f} ({"ok" if dZ <= 0.05 else "FAIL"}), largest bucket {big:.4f} at {bn} ({"ok" if big <= 0.10 else "FAIL"})'
          + (f'   [critic advance, holdout feeds: dZ {crit[0]:.4f}, bucket {crit[1]:.4f}]'))
        P('     view by bucket ' + ' '.join(f'{x:.3f}' for x in zv) + ' | ref ' + ' '.join(f'{x:.3f}' for x in zr) + ' | view row share ' + ' '.join(f'{x:.3f}' for x in w))
        if official:
            c3ok &= ok; TAB.append((f'C3 {regn.split(" train")[0]} vs {refn.split(" rows")[0]}: dZ / largest bucket', '<= 0.05 / <= 0.10', f'{dZ:.4f} / {big:.4f}', f'{crit[0]:.4f} / {crit[1]:.4f} (holdout feeds)', ok))
        else:
            TAB.append((f'   C3 on the HOLDOUT feeds (comparison) {regn.split(" train")[0]} vs {refn.split(" rows")[0]}', '(no bar)', f'{dZ:.4f} / {big:.4f}', f'{crit[0]:.4f} / {crit[1]:.4f}', None))
VER['C3'] = c3ok
dZ, big, bn, zv, zr, w = gap(vt_nz[vbig], bv[vbig], en_nz[~epub & ebig], be[~epub & ebig])
P(f'  information: stale-val twin (val M >= 50) vs PRIVATE: share {vt_nz[vbig].mean():.4f} vs {en_nz[~epub & ebig].mean():.4f}; dZ {dZ:.4f}, largest bucket {big:.4f} at {bn}   [critic advance: 0.1564 vs 0.1717; dZ 0.0156, bucket 0.0292]')
# C5
vshare = {s: float(vn_nz[vslot == s].mean()) for s in (16, 18)}; vall = float(vn_nz.mean()); eshare = {s: float(en_nz[eslot == s].mean()) for s in (20, 21, 22, 23)}
vref = max(vall, vshare[16], vshare[18]); c5ok = all(v >= 0.5 * vref for v in eshare.values())
hs = np.repeat(tslot[hold], Mt[hold]); hr = rows_of(ish)
P(f'--- C5 non-zero share by day: val slot 16 / 18 / all: {vshare[16]:.4f} / {vshare[18]:.4f} / {vall:.4f} | test slot 20 / 21 / 22 / 23: ' + ' / '.join(f'{eshare[s]:.4f}' for s in (20, 21, 22, 23))
  + f' | public {en_nz[epub].mean():.4f} private {en_nz[~epub].mean():.4f} | holdout lag view slot 6 / 7 / 8 / 11 / 12 / 13: ' + ' / '.join(f'{nz_tr[hr][hs == s].mean():.4f}' for s in (6, 7, 8, 11, 12, 13)))
P('     [critic advance: val 0.2086 / 0.1868 | test 0.1802 / 0.1751 / 0.1718 / 0.1705 | public 0.1795 private 0.1718 | holdout lag view 0.1665 / 0.1634 / 0.1541 / 0.1382 / 0.1370 / 0.1375]')
P(f'     bar: every test day >= 0.5 x {vref:.4f} = {0.5 * vref:.4f}: {"ok" if c5ok else "FAIL"}')
VER['C5'] = c5ok; TAB.append(('C5 non-zero share, lowest test day against 0.5 x val\'s', f'>= {0.5 * vref:.4f}', f'{min(eshare.values()):.4f}', '0.1705 (bar 0.1043)', c5ok))
# C4
def psi_bins(a, b, binf, nbin):
    pa = np.bincount(binf(a), minlength=nbin) / len(a); pb = np.bincount(binf(b), minlength=nbin) / len(b)
    pa = np.maximum(pa, 1e-6); pb = np.maximum(pb, 1e-6); return float(((pa - pb) * np.log(pa / pb)).sum())
def spec_binner(val):
    ed = np.unique(np.quantile(val[val != 0], np.linspace(0.1, 0.9, 9)))
    return (lambda x: np.where(x == 0, 0, 1 + np.searchsorted(ed, x, side='right'))), len(ed) + 2
def crit_binner(val):
    ed = np.unique(np.quantile(val, np.linspace(0.1, 0.9, 9)))
    return (lambda x: np.searchsorted(ed, x, side='right')), len(ed) + 1
CRIT_C4 = {'nx_rp3_z': (0.0884, 0.0230, 0.0109, 0.0437, 0.0545), 'nx_rp3_rk': (0.1710, 0.0681, 0.0451, 0.0993, 0.0535)}
c4ok = True; c4flag = []
pr_ = rows_of(pool & has); ar_ = hrows; hlr = rows_of(ish & has); tlr = rows_of(ish & (tslot >= 11))
P('--- C4 PSI (bar 0.25): train side against val / public / private (all rows of the reference)')
for nm, FT in (('nx_rp3_z', ZF), ('nx_rp3_rk', RF)):
    a_v, a_e = FT['val'], FT['test']
    for bn_, mk in (('BUILD_SPEC bins (10 val deciles of the non-zero values + exact-zero bin)', spec_binner), ('critic bins (9 decile edges over all val values)', crit_binner)):
        binf, nbin = mk(a_v)
        for popn, r in (('POOL feeds has = 1', pr_), ('ALL has = 1 train rows', ar_), ('HOLDOUT has = 1 rows', hlr)):
            a_h = np.asarray(FT['train'][r]); v = (psi_bins(a_h, a_v, binf, nbin), psi_bins(a_h, a_e[epub], binf, nbin), psi_bins(a_h, a_e[~epub], binf, nbin))
            official = bn_.startswith('BUILD_SPEC') and popn.startswith('POOL')
            ok = max(v) <= 0.25
            P(f'  {nm} | {bn_} | {popn} ({len(r)} rows): vs val {v[0]:.4f}, vs public {v[1]:.4f}, vs private {v[2]:.4f}  {"[OFFICIAL] " + ("ok" if ok else "FAIL") if official else ("(comparison)" if ok else "(comparison) ABOVE 0.25")}')
            if official:
                c4ok &= ok; TAB.append((f'C4 PSI {nm}, pool has = 1 rows vs val / public / private', '<= 0.25', f'{v[0]:.4f} / {v[1]:.4f} / {v[2]:.4f}', '%.4f / %.4f / %.4f (critic bins, holdout rows)' % CRIT_C4[nm][:3], ok))
            elif not ok: c4flag.append(f'{nm} | {bn_} | {popn}')
            if bn_.startswith('critic') and popn.startswith('HOLDOUT'):
                TAB.append((f'   same, critic bins on the holdout has = 1 rows (comparison)', '(no bar)', f'{v[0]:.4f} / {v[1]:.4f} / {v[2]:.4f}', '%.4f / %.4f / %.4f' % CRIT_C4[nm][:3], None))
        P(f'  {nm} | {bn_} | information: holdout test-like rows (slots 11-13) vs private {psi_bins(np.asarray(FT["train"][tlr]), a_e[~epub], binf, nbin):.4f}; val vs private {psi_bins(a_v, a_e[~epub], binf, nbin):.4f}; val vs public {psi_bins(a_v, a_e[epub], binf, nbin):.4f}; stale-val twin vs private {psi_bins(ZT if nm == "nx_rp3_z" else RT, a_e[~epub], binf, nbin):.4f}'
          + (f'   [critic advance, critic bins: {CRIT_C4[nm][3]:.4f}; val vs private {CRIT_C4[nm][4]:.4f}]'))
VER['C4'] = c4ok
P(f'  C4 comparison values above 0.25: {c4flag if c4flag else "none"}')
# J1 (no bar)
P('--- J1 (no bar): P(walk = 0 | seq_cv_invd20 > 0)   [critic advance: holdout lag view slots 6-8 0.1231, slots 11-13 0.2293; val 0.0019; public 0.0021; private 0.0033; val twin 0.1571]')
cvt = np.load(FC / 'train/seq_cv_invd20.npy', mmap_mode='r'); cvv = np.load(FC / 'val/seq_cv_invd20.npy'); cve = np.load(FC / 'test/seq_cv_invd20.npy')
J1 = {}
for nm_, r in (('holdout lag view, slots 6-8', rows_of(ish & (tslot >= 6) & (tslot <= 8))), ('holdout lag view, slots 11-13', rows_of(ish & (tslot >= 11) & (tslot <= 13))),
               ('ALL lag-view train rows, slots 6-9 (val-like)', rows_of(has & (tslot <= 9))), ('ALL lag-view train rows, slots 11-14 (test-like)', rows_of(has & (tslot >= 11))), ('ALL lag-view train rows', hrows)):
    i_ = np.asarray(cvt[r]) > 0; w_ = nz_tr[r]; J1[nm_] = 1 - w_[i_].mean()
    P(f'  {nm_}: rows {len(r)}, walk > 0 {w_.mean():.4f}, seq_cv_invd20 > 0 {i_.mean():.4f}, P(walk = 0 | invd20 > 0) {J1[nm_]:.4f}, P(walk > 0 | invd20 = 0) {w_[~i_].mean():.4f}')
for nm_, w_, i_ in (('val rows (natural)', vn_nz, cvv > 0), ('val rows (stale-val twin of the walk; seq_cv as stored)', vt_nz, cvv > 0), ('test rows, public', en_nz[epub], cve[epub] > 0), ('test rows, private', en_nz[~epub], cve[~epub] > 0)):
    J1[nm_] = 1 - w_[i_].mean()
    P(f'  {nm_}: rows {len(w_)}, walk > 0 {w_.mean():.4f}, seq_cv_invd20 > 0 {i_.mean():.4f}, P(walk = 0 | invd20 > 0) {J1[nm_]:.4f}, P(walk > 0 | invd20 = 0) {w_[~i_].mean():.4f}')
TAB.append(('J1 P(walk = 0 | seq_cv_invd20 > 0): holdout lag view slots 6-8 / 11-13 || val / public / private', '(no bar)',
            f'{J1["holdout lag view, slots 6-8"]:.3f} / {J1["holdout lag view, slots 11-13"]:.3f} || {J1["val rows (natural)"]:.3f} / {J1["test rows, public"]:.3f} / {J1["test rows, private"]:.3f}', '0.123 / 0.229 || 0.002 / 0.002 / 0.003', None))
TAB.append(('   same on ALL lag-view train rows: slots 6-9 / 11-14', '(no bar)', f'{J1["ALL lag-view train rows, slots 6-9 (val-like)"]:.3f} / {J1["ALL lag-view train rows, slots 11-14 (test-like)"]:.3f}', '(not built by the critic)', None))
del cvt, f4t
# ------------------------------------------------------------------ PART 2: G-info (labels are opened from here on)
P('\n=== PART 2. G-info: calibrated lens (score test given pool + own inputs); bar: sign + and |cz| >= 2.5 ===')
c9h = np.load(C9 / 'c09_hold.npz')
Mh = Mt[hold]; co = F['train']['cand_off'].to_numpy().astype(np.int64)[hold]; offh = np.concatenate([[0], np.cumsum(Mh)[:-1]]); rows_h = np.repeat(co - offh, Mh) + np.arange(int(Mh.sum()))
slot_h = np.repeat(tslot[hold], Mh)
zh = np.nan_to_num(np.asarray(ZF['train'][rows_h]).astype(np.float64), nan=0.0); rh = np.nan_to_num(np.asarray(RF['train'][rows_h]).astype(np.float64), nan=0.0)
rawh = np.full(len(rows_h), np.nan); hpos = np.searchsorted(hrows, rows_h[hasrow[rows_h]]); rawh[hasrow[rows_h]] = raw_tr[hpos, 0]
assert (hrows[hpos] == rows_h[hasrow[rows_h]]).all()
def nan_out(x, keep):
    y = np.array(x, np.float64); y[~keep] = np.nan; return y
def sc(N, x, D):
    Fm = lens.forms(np.asarray(x, np.float64), D, log=False); a = N.score(Fm['z']); b = N.score(Fm['rk']); return a, b, Fm
def reg0(x, keep):
    y = np.array(x, np.float64); y[~keep] = 0.0; return y
D = lens.load_split('hold', 'TO16'); condlib.sp = 'hold'; nm_, cl = condlib.control_block(D); N = condlib.Null(D, nm_, cl)
assert (D['rows'] == rows_h).all()
P(f'[holdout, pool = 16 held-out wide students; control block {len(nm_)} columns]')
a = sc(N, c9h['rp3b05'], D); P(f'  lens check: stored FRESH rp3b05 on ALL 30,000 holdout feeds: cz {a[0]["cz"]:+.2f} / {a[1]["cz"]:+.2f} (record +8.20 / +9.75; critic +8.20 / +9.75)')
reg = {'all lag-view feeds (slots 6-13; 15,225)': slot_h >= 6, 'val-like (slots 6-8; 6,946)': (slot_h >= 6) & (slot_h <= 8), 'test-like (slots 11-13; 8,279)': slot_h >= 11}
CRIT_H = {'all lag-view feeds (slots 6-13; 15,225)': (5.02, 4.43, 0.00044, 0.0299), 'val-like (slots 6-8; 6,946)': (3.51, 2.81, 0.00022, 0.0285), 'test-like (slots 11-13; 8,279)': (3.00, 2.76, 0.00016, 0.0230)}
HZ = {}
for rn, rm in reg.items():
    az = N.score(reg0(zh, rm).astype(np.float32)); ar = N.score(reg0(rh, rm).astype(np.float32)); HZ[rn] = (az, ar)
    b = sc(N, nan_out(np.log1p(Kr * np.nan_to_num(rawh)), rm), D); fr = sc(N, nan_out(np.log1p(Kr * c9h['rp3b05']), rm), D); cr = CRIT_H[rn]
    P(f'  {rn}: FILES as entered: nx_rp3_z cz {az["cz"]:+.2f} (dnats {az["dnats"]:.5f}, b1 {az["b1"]:+.4f}, R2 {az["r2"]:.2f}) | nx_rp3_rk cz {ar["cz"]:+.2f} (b1 {ar["b1"]:+.4f})')
    P(f'      critic\'s formula on my raw array, forms(log1p(K rp3b05)): z {b[0]["cz"]:+.2f} / rank {b[1]["cz"]:+.2f} (dnats {b[0]["dnats"]:.5f}, b1 {b[0]["b1"]:+.4f})   [critic advance: {cr[0]:+.2f} / {cr[1]:+.2f} ({cr[2]:.5f}), b1 {cr[3]:+.4f}]')
    P(f'      FRESH stored column (cut = own day) on the same feeds, same formula: z {fr[0]["cz"]:+.2f} / rank {fr[1]["cz"]:+.2f} (dnats {fr[0]["dnats"]:.5f}); kept share {b[0]["cz"] / fr[0]["cz"]:.2f} / {b[1]["cz"] / fr[1]["cz"]:.2f}')
pzh = np.array([N.score(condlib.perm_within(zh.astype(np.float32), D['off'], D['M'], 3000 + s))['cz'] for s in range(20)])
P(f'  C6 placebo: 20 within-feed permutations of the nx_rp3_z file on the holdout rows (lag view): sd {pzh.std():.2f}, max |cz| {np.abs(pzh).max():.2f}, values ' + ' '.join(f'{x:+.2f}' for x in pzh) + '   [critic advance: sd 0.89, max 1.93]')
hz_, hr_ = HZ['all lag-view feeds (slots 6-13; 15,225)']
del N, D
VAL = {}
for base in ('TO16', 'FILE'):
    D = lens.load_split('val', base); condlib.sp = 'val'; nm_, cl = condlib.control_block(D); N = condlib.Null(D, nm_, cl)
    nz_, nr_ = N.score(ZF['val']), N.score(RF['val']); tz_, tr_ = N.score(ZT), N.score(RT)
    bn = sc(N, np.log1p(Kr * VN[:, 0]), D); bt = sc(N, np.log1p(Kr * VT[:, 0]), D); c9v = np.load(C9 / 'c09_val.npz'); bs = sc(N, np.log1p(Kr * c9v['rp3b05']), D)
    VAL[base] = (nz_, nr_, tz_, tr_)
    crn, crt = ((6.84, 5.97, 0.00168, 0.0556), (6.64, 4.90, 0.00163, 0.0513)) if base == 'TO16' else ((6.91, 6.00, 0.00171, None), (6.70, 4.92, 0.00165, None))
    P(f'[val, base {base}; control block {len(nm_)} columns]')
    P(f'  NATURAL files (information): nx_rp3_z cz {nz_["cz"]:+.2f} (dnats {nz_["dnats"]:.5f}, b1 {nz_["b1"]:+.4f}, R2 {nz_["r2"]:.2f}) | nx_rp3_rk cz {nr_["cz"]:+.2f} (b1 {nr_["b1"]:+.4f})'
      f' | critic\'s formula on my raw: {bn[0]["cz"]:+.2f} / {bn[1]["cz"]:+.2f} | the stored c09_val column, same formula: {bs[0]["cz"]:+.2f} / {bs[1]["cz"]:+.2f}   [critic advance: {crn[0]:+.2f} / {crn[1]:+.2f} ({crn[2]:.5f})' + (f', b1 {crn[3]:+.4f}]' if crn[3] else ']'))
    P(f'  STALE-VAL TWIN files (gate): nx_rp3_z cz {tz_["cz"]:+.2f} (dnats {tz_["dnats"]:.5f}, b1 {tz_["b1"]:+.4f}, R2 {tz_["r2"]:.2f}) | nx_rp3_rk cz {tr_["cz"]:+.2f} (b1 {tr_["b1"]:+.4f})'
      f' | critic\'s formula on my raw: {bt[0]["cz"]:+.2f} / {bt[1]["cz"]:+.2f}   [critic advance: {crt[0]:+.2f} / {crt[1]:+.2f} ({crt[2]:.5f})' + (f', b1 {crt[3]:+.4f}]' if crt[3] else ']'))
    for vs in (16, 18):
        mk = vslot == vs
        P(f'      val slot {vs}: natural z {N.score(reg0(ZF["val"], mk).astype(np.float32))["cz"]:+.2f} / rank {N.score(reg0(RF["val"], mk).astype(np.float32))["cz"]:+.2f} | twin z {N.score(reg0(ZT, mk).astype(np.float32))["cz"]:+.2f} / rank {N.score(reg0(RT, mk).astype(np.float32))["cz"]:+.2f}'
          + ('   [critic advance, raw rp3b05 form: slot 16 natural +3.30 / +3.23, twin +3.97 / +2.94; slot 18 natural +3.84 / +2.83, twin +4.17 / +2.48]' if base == 'TO16' and vs == 16 else ''))
    if base == 'TO16':
        pzv = np.array([N.score(condlib.perm_within(ZT.astype(np.float32), D['off'], D['M'], 4000 + s))['cz'] for s in range(20)])
        P(f'  C6 placebo: 20 within-feed permutations of the stale-val twin nx_rp3_z file: sd {pzv.std():.2f}, max |cz| {np.abs(pzv).max():.2f}, values ' + ' '.join(f'{x:+.2f}' for x in pzv) + '   [critic advance: sd 0.91, max 1.55]')
        pzn = np.array([N.score(condlib.perm_within(ZF['val'].astype(np.float32), D['off'], D['M'], 5000 + s))['cz'] for s in range(20)])
        P(f'  information: 20 within-feed permutations of the natural val nx_rp3_z file (seeds 5000-5019): sd {pzn.std():.2f}, max |cz| {np.abs(pzn).max():.2f}')
        Fa, Fb = ZF['val'], ZT
        P(f'  val rows: correlation natural vs twin, nx_rp3_z {np.corrcoef(Fa, Fb)[0, 1]:.3f}, nx_rp3_rk {np.corrcoef(RF["val"], RT)[0, 1]:.3f}; non-zero in natural {vn_nz.mean():.4f}, in twin {vt_nz.mean():.4f}, natural only {np.mean(vn_nz & ~vt_nz):.4f}, twin only {np.mean(~vn_nz & vt_nz):.4f}   [critic advance: 0.778 / 0.835; 0.1982 / 0.1569 / 0.0449 / 0.0035]')
    del N, D
def passes(*czs):
    return any(c >= 2.5 for c in czs)
tzT, trT = VAL['TO16'][2], VAL['TO16'][3]; tzF, trF = VAL['FILE'][2], VAL['FILE'][3]
c7h = passes(hz_['cz'], hr_['cz']); c7v = passes(tzT['cz'], trT['cz']); c7vF = passes(tzF['cz'], trF['cz'])
same_form = (hz_['cz'] >= 2.5 and tzT['cz'] >= 2.5 and tzF['cz'] >= 2.5) or (hr_['cz'] >= 2.5 and trT['cz'] >= 2.5 and trF['cz'] >= 2.5)
all_four = min(hz_['cz'], hr_['cz'], tzT['cz'], trT['cz'], tzF['cz'], trF['cz']) >= 2.5
VER['C7'] = c7h and c7v and c7vF
VER['C6'] = bool(np.abs(pzh).max() < 3.0 and np.abs(pzv).max() < 3.0)
TAB.append(('C7 holdout rows, lag view (15,225 feeds): cz of nx_rp3_z / nx_rp3_rk', 'sign +, >= 2.5 (z or rank)', f'{hz_["cz"]:+.2f} / {hr_["cz"]:+.2f}', '+5.02 / +4.43', c7h))
TAB.append(('C7 val rows, stale-val twin, base TO16: cz of nx_rp3_z / nx_rp3_rk', 'sign +, >= 2.5 (z or rank)', f'{tzT["cz"]:+.2f} / {trT["cz"]:+.2f}', '+6.64 / +4.90', c7v))
TAB.append(('C7 val rows, stale-val twin, base FILE (FD16P20)', 'sign +, >= 2.5 (z or rank)', f'{tzF["cz"]:+.2f} / {trF["cz"]:+.2f}', '+6.70 / +4.92', c7vF))
TAB.append(('   val rows, natural files, base TO16 / FILE (information): z, rank', '(no bar)', f'{VAL["TO16"][0]["cz"]:+.2f}, {VAL["TO16"][1]["cz"]:+.2f} / {VAL["FILE"][0]["cz"]:+.2f}, {VAL["FILE"][1]["cz"]:+.2f}', '+6.84, +5.97 / +6.91, +6.00', None))
TAB.append(('C6 20 permutations, max |cz|: holdout lag view / val twin', '< 3.0', f'{np.abs(pzh).max():.2f} / {np.abs(pzv).max():.2f} (sd {pzh.std():.2f} / {pzv.std():.2f})', '1.93 / 1.55 (sd 0.89 / 0.91)', VER['C6']))
TAB.append(('J2 one-step coefficient b1 of nx_rp3_z: holdout lag view (all / slots 6-8 / slots 11-13) || val natural / twin (TO16)', '(no bar)',
            f'{hz_["b1"]:.4f} / {HZ["val-like (slots 6-8; 6,946)"][0]["b1"]:.4f} / {HZ["test-like (slots 11-13; 8,279)"][0]["b1"]:.4f} || {VAL["TO16"][0]["b1"]:.4f} / {tzT["b1"]:.4f}', '0.0299 / 0.0285 / 0.0230 || 0.0556 / 0.0513', None))
TAB.append(('   J2 R2 of nx_rp3_z with pool + own inputs: holdout lag view all || val natural / twin', '(no bar)', f'{hz_["r2"]:.2f} || {VAL["TO16"][0]["r2"]:.2f} / {tzT["r2"]:.2f}', '0.25 || 0.52 / 0.42', None))
# ------------------------------------------------------------------ verdict
def grep_verdict(path, key):
    try:
        for ln in open(path):
            if key in ln: return ln.strip().endswith('PASS')
    except FileNotFoundError:
        pass
    return None
c1 = grep_verdict(OUT / f'c1_repro{TAG}.txt', 'C1 VERDICT'); c2 = grep_verdict(OUT / f'build{TAG}.log', 'C2 VERDICT'); rd = grep_verdict(OUT / f'build{TAG}.log', 'RE-DERIVATION VERDICT')
P('\n=== STEP 0 TABLE: gate | bar | build of record | critic advance (c03_gates.txt / c05_coef.txt / c06_joint.txt) | verdict ===')
P(f'G-code C1 stored c09_val / c09_hold reproduced | 1e-5 relative | see out/c1_repro{TAG}.txt | max abs diff 0 (val; holdout slots 6, 13) | {"PASS" if c1 else ("FAIL" if c1 is False else "NOT READ")}')
P(f'G-code C2 causality counts (+ numpy re-derivation on a sample) | all 0 | see out/build{TAG}.log | (not run by the critic) | {"PASS" if (c2 and rd) else ("FAIL" if (c2 is False or rd is False) else "NOT READ")}')
for g, bar, mine, crit, ok in TAB:
    P(f'{g} | {bar} | {mine} | {crit} | {"" if ok is None else ("ok" if ok else "FAIL")}')
gcode = bool(c1 and c2 and rd); gpar = VER['C3'] and VER['C4'] and VER['C5']; ginfo = VER['C7'] and VER['C6']
P(f'G-code {"PASS" if gcode else "FAIL"} (C1 {c1}, C2 {c2}, re-derivation {rd}); G-parity {"PASS" if gpar else "FAIL"} (C3 {VER["C3"]}, C4 {VER["C4"]}, C5 {VER["C5"]}); G-info {"PASS" if ginfo else "FAIL"} (C7 {VER["C7"]}, C6 {VER["C6"]})')
P(f'G-info detail: the same form passes on both witnesses and both val bases: {same_form}; all six cz (z and rank; holdout, twin TO16, twin FILE) >= 2.5: {all_four}')
if gcode and gpar and ginfo: v = 'PASS'
elif not gcode or not ginfo: v = 'FAIL G-code or G-info -> NO ARM, closed'
else: v = 'FAIL G-parity only -> NO ARM, recorded as NOT TESTABLE AT PARITY IN 14 DAYS OF DATA (not a closure of the information)'
P(f'STEP 0 VERDICT (amended rule A2): {v}')
P('DONE')
