"""nx3 verifier v05 (label-free): on the INSTALLED files nn/feat_cache/<split>/nx_rp3_{z,rk}.npy
 A  FULL train file against my first-principles walk of ALL has = 1 feeds (out/v03_trainfull_*.npz): row-by-row equality of both forms, NaN pattern
 B  the has flag: NaN exactly on the rows of feeds of slots 0-5 (Sep 6-11); what the loader makes of it on val / test
 C  (b) lag: (row time - newest contributing source time) on lag-view train rows against val, public, private rows
 D  (b) non-zero share and PSI by day
 E  the label-free STEP 0 gates C3, C4, C5 and J1 (definitions of BUILD_SPEC 5 / the ledger rule; my own code)
 F  alignment invariants that do not use my walk: per-feed rank sum, item-absence invariant, relation to the stored seq_cv_invd20, with a deliberately
    shifted array for power
No target array is opened. usage: v05_parity_align.py"""
from v_common import *
from v_common import _gather
P = Log(f'{OUT}/v05_parity_align.txt')
FC = ROOT + '/nn/feat_cache'
sys.path.insert(0, ROOT + '/nn_x/w12/stale/hz2')
import hz2_common as hc                      # listing-age buckets of record (AGE_EDGES_D, knots) only
TR = Split('train'); VA = Split('val', hist=False); TE = Split('test', hist=False)
Mt = TR.M; has = TR.slot >= 6; hasrow = np.repeat(has, Mt); srow = np.repeat(TR.slot, Mt).astype(np.int8)
ZI = {s: np.load(f'{FC}/{s}/nx_rp3_z.npy') for s in ('train', 'val', 'test')}; RI = {s: np.load(f'{FC}/{s}/nx_rp3_rk.npy') for s in ('train', 'val', 'test')}
for s in ('train', 'val', 'test'):
    assert ZI[s].shape == (NROWS[s],) and RI[s].shape == (NROWS[s],) and ZI[s].dtype == np.float32 and RI[s].dtype == np.float32
    for nm in ('cand_fidx', 'cand_item_id', 'cand_f4'):
        assert np.load(f'{FLAT}/{s}/{nm}.npy', mmap_mode='r').shape == (NROWS[s],)
P('row counts: every installed file has the length of data/flat/<split>/cand_*.npy (train 246,745,836; val 1,389,971; test 2,904,806), float32, 1-D')
for s, S in (('train', TR), ('val', VA), ('test', TE)):
    cf = np.load(f'{FLAT}/{s}/cand_fidx.npy'); ok = bool((cf == np.repeat(np.arange(S.n, dtype=np.int32), S.M)).all()); P(f'  {s}: cand_fidx == repeat(arange(feeds), M) on all rows: {ok}'); del cf

# ---------------------------------------------------------------- A
P('=== A. FULL train file against my first-principles walk of all has = 1 feeds')
V = np.load(f'{OUT}/v01_natural_val.npz'); K = 1.0 / np.median(V['rp3'][0][V['rp3'][0] > 0]); P(f'K (mine, from my val walk) {K:.6e}')
raw = np.full(NROWS['train'], np.nan); newest_d = np.full(NROWS['train'], -1, np.int64); newest_s = np.full(NROWS['train'], -1, np.int64)
fr_sp = np.zeros(NROWS['train'], np.float32); msrc = np.zeros(NROWS['train']); cnt = np.zeros(8, np.int64); nfe = 0; dv = np.zeros(3, np.int64); overcap = 0
for k in range(3):
    d = np.load(f'{OUT}/v03_trainfull_{k}of3.npz'); r = d['rows']
    raw[r] = d['rp3']; newest_d[r] = d['newest_d']; newest_s[r] = d['newest_s']; fr_sp[r] = d['fr_sparse']; msrc[r] = d['mean_src_ts']; cnt += d['cnt']; nfe += len(d['qf'])
    dv += np.array([int(d['dv1'].sum()), int(d['dv2'].sum()), int(d['dv3'].sum())]); overcap += int((d['nocc'] > CAP).sum()); del d, r
P(f'my walk: {nfe} feeds; rows filled {int((~np.isnan(raw)).sum())}; filled rows == rows of has = 1 feeds: {bool(((~np.isnan(raw)) == hasrow).all())}')
P(f'audit of my own V0 pairs over ALL has = 1 train rows [admitted, dense, sparse, dense slot >= cut, sparse ts >= own, own non-zero user, own feed, any member ts >= own]: {cnt.tolist()}')
P(f'sensitivity over all has = 1 rows: raw walk would differ on {int(dv[0])} rows with the own user admitted, {int(dv[1])} rows with sparse members at/after the own time admitted, {int(dv[2])} rows with user 0 linked; rows over the 500 cap {overcap}')
hf = np.nonzero(has)[0]; Mh = Mt[hf]; hrows = np.nonzero(hasrow)[0]
offh = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64)
zc, rc = my_forms(raw[hrows], offh, Mh, K)
zi = ZI['train'][hrows].astype(np.float64); ri = RI['train'][hrows].astype(np.float64)
bz = ~(np.abs(zi - zc) <= 2e-5); br = ~(np.abs(ri - rc) <= 1e-6)
P(f'has = 1 rows {len(hrows)}: max |dz| {np.nanmax(np.abs(zi - zc)):.2e}, max |drank| {np.nanmax(np.abs(ri - rc)):.2e}; rows differing: z {int(bz.sum())}, rank {int(br.sum())}; NaN in installed has = 1 rows: {int(np.isnan(zi).sum())} / {int(np.isnan(ri).sum())}')
P(f'has = 0 rows {int((~hasrow).sum())}: installed NaN on all of them: z {bool(np.isnan(ZI["train"][~hasrow]).all())}, rank {bool(np.isnan(RI["train"][~hasrow]).all())}')
P(f'A VERDICT: every one of the {NROWS["train"]} installed train rows equals the admissible-only recomputation at its data/flat row index: {bool(bz.sum() == 0 and br.sum() == 0 and np.isnan(ZI["train"][~hasrow]).all() and np.isnan(RI["train"][~hasrow]).all())}')
nz_tr = np.zeros(NROWS['train'], bool); nz_tr[hrows] = raw[hrows] > 0
del zc, rc, zi, ri, bz, br
Vn = np.load(f'{OUT}/v01_natural_val.npz'); En = np.load(f'{OUT}/v01_natural_test.npz')
assert (Vn['rows'] == np.arange(NROWS['val'])).all() and (En['rows'] == np.arange(NROWS['test'])).all()
nz_v = Vn['p3a'][0] > 0; nz_e = En['p3a'][0] > 0

# ---------------------------------------------------------------- B
P('=== B. the has flag (non-NaN pattern of nx_rp3_z, which is what nn_x/abl/data.py attach_xt turns into has:nx_rp3_z)')
hz = ~np.isnan(ZI['train']); hr = ~np.isnan(RI['train'])
P(f'train: NaN pattern of the two files identical: {bool((hz == hr).all())}; has mean over rows {hz.mean():.4f}')
fs = np.add.reduceat(hz.astype(np.int64), TR.co)
P(f'  constant inside every feed: {bool(((fs == 0) | (fs == Mt)).all())}; feeds with has = 1: {int((fs == Mt).sum())}; equal to (slot >= 6) feed by feed: {bool(((fs == Mt) == has).all())}')
for s in np.unique(TR.slot):
    m = TR.slot == s; P(f'  slot {s:2d} (Sep {6 + s:2d}): feeds {int(m.sum())}, rows {int(Mt[m].sum())}, has = 1 on {int((fs[m] == Mt[m]).sum())} feeds, has = 0 on {int((fs[m] == 0).sum())}')
hold_p = np.load(ROOT + '/eda/labels_position/out/train_prandom.npz'); p = np.nan_to_num(hold_p['p_random'].astype(np.float32), nan=0.0); p[Mt < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((p > 0.9) & (Mt >= 50))[0], 30000, replace=False)); ish = np.zeros(TR.n, bool); ish[hold] = True
pool = (p > 0.5) & (Mt >= 50) & ~ish
P(f'  holdout feeds 30000: has = 1 on {int((has & ish).sum())}, has = 0 on {int((~has & ish).sum())}; pool feeds (p_random > 0.5, M >= 50, not holdout) {int(pool.sum())}, with has = 1 {int((pool & has).sum())}')
for s in ('val', 'test'):
    P(f'{s}: NaN in nx_rp3_z {int(np.isnan(ZI[s]).sum())}, in nx_rp3_rk {int(np.isnan(RI[s]).sum())} -> the loader gives has:nx_rp3_z = 1 on every {s} row; min / max z {ZI[s].min():.3f} / {ZI[s].max():.3f}, rank {RI[s].min():.3f} / {RI[s].max():.3f}')
zt = np.nan_to_num(np.clip(ZI['train'], -5, 5), nan=0.0); rt = np.nan_to_num(RI['train'], nan=0.0)
P(f'as the loader reads train (clip, NaN -> 0): nx_rp3_z mean {zt.mean():+.4f} sd {zt.std():.4f}; nx_rp3_rk mean {rt.mean():+.5f} sd {rt.std():.4f}; has mean {hz.mean():.4f}; on has = 1 rows only: z sd {ZI["train"][hrows].std():.4f}, rank sd {RI["train"][hrows].std():.4f}; val z sd {ZI["val"].std():.4f}, test z sd {ZI["test"].std():.4f}')
del zt, rt, hr

# ---------------------------------------------------------------- C
P('=== C. lag = (row time - time of the NEWEST contributing source sequence), days; non-zero rows only. "sparse-newest" = the newest contributing source is a sparse member')
epub_f = TE.ts < PUB_TS
def lagstats(tag, qts_row, nd, ns, nzm, frs, ms):
    newest = np.maximum(nd, ns)[nzm]; lag = (qts_row[nzm] - newest) / 86400.0; spn = (ns[nzm] > nd[nzm])
    dl = (qts_row[nzm] - nd[nzm]) / 86400.0; hasd = nd[nzm] >= 0
    q = np.quantile(lag, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95])
    qd = np.quantile(dl[hasd], [0, 0.01, 0.25, 0.5, 0.75]) if hasd.any() else np.full(5, np.nan)
    P(f'  {tag}: non-zero rows {int(nzm.sum())} | newest-source lag min {q[0]:.3f} p1 {q[1]:.3f} p5 {q[2]:.2f} p25 {q[3]:.2f} p50 {q[4]:.2f} p75 {q[5]:.2f} p95 {q[6]:.2f} | share < 1 d {np.mean(lag < 1):.4f}, < 2 d {np.mean(lag < 2):.4f}, < 6 d {np.mean(lag < 6):.4f} | '
      f'sparse-newest {spn.mean():.4f} | newest DENSE source lag min {qd[0]:.3f} p1 {qd[1]:.2f} p25 {qd[2]:.2f} p50 {qd[3]:.2f} p75 {qd[4]:.2f} | value share through sparse members: mean {frs[nzm].mean():.4f}, rows with any {np.mean(frs[nzm] > 0):.4f} | '
      f'value-weighted mean source age: mean {np.mean((qts_row[nzm] - ms[nzm]) / 86400.0):.2f} d, median {np.median((qts_row[nzm] - ms[nzm]) / 86400.0):.2f} d')
    return lag
tsrow = np.repeat(TR.ts, Mt); poolrow = np.repeat(pool, Mt); holdrow = np.repeat(ish, Mt)
LAG = {}
for nm, m in (('train lag view, ALL rows, slots 6-8 (val-like)', (srow >= 6) & (srow <= 8)), ('train lag view, ALL rows, slots 11-13 (test-like)', (srow >= 11) & (srow <= 13)),
              ('train lag view, POOL rows, slots 6-8', poolrow & (srow >= 6) & (srow <= 8)), ('train lag view, POOL rows, slots 11-13', poolrow & (srow >= 11) & (srow <= 13)),
              ('train lag view, HOLDOUT rows, slots 6-8', holdrow & (srow >= 6) & (srow <= 8)), ('train lag view, HOLDOUT rows, slots 11-13', holdrow & (srow >= 11) & (srow <= 13))):
    LAG[nm] = lagstats(nm, tsrow, newest_d, newest_s, m & nz_tr, fr_sp, msrc)
for s in (6, 7, 8, 9, 11, 12, 13, 14):
    lagstats(f'train lag view, ALL rows, slot {s} (Sep {6 + s})', tsrow, newest_d, newest_s, (srow == s) & nz_tr, fr_sp, msrc)
vts = np.repeat(VA.ts, VA.M); ets = np.repeat(TE.ts, TE.M); vsl = np.repeat(VA.slot, VA.M); esl = np.repeat(TE.slot, TE.M); epub = np.repeat(epub_f, TE.M)
vfr = (Vn['tot_sparse'] / np.maximum(Vn['p3a'][0], 1e-300)); efr = (En['tot_sparse'] / np.maximum(En['p3a'][0], 1e-300)); vms = Vn['wts'] / np.maximum(Vn['p3a'][0], 1e-300); ems = En['wts'] / np.maximum(En['p3a'][0], 1e-300)
LAG['val'] = lagstats('val rows (natural), all', vts, Vn['newest_d'], Vn['newest_s'], nz_v, vfr, vms)
for s in (16, 18):
    lagstats(f'val rows, slot {s}', vts, Vn['newest_d'], Vn['newest_s'], nz_v & (vsl == s), vfr, vms)
LAG['pub'] = lagstats('test rows, PUBLIC', ets, En['newest_d'], En['newest_s'], nz_e & epub, efr, ems)
LAG['priv'] = lagstats('test rows, PRIVATE', ets, En['newest_d'], En['newest_s'], nz_e & ~epub, efr, ems)
for s in (20, 21, 22, 23):
    lagstats(f'test rows, slot {s}', ets, En['newest_d'], En['newest_s'], nz_e & (esl == s), efr, ems)
P('  fresher-than-reference reading (share of non-zero rows whose newest contributing source is younger than x days):')
for x in (0.25, 0.5, 1, 2, 3, 6, 7):
    P(f'    < {x:4} d: train ALL val-like {np.mean(LAG["train lag view, ALL rows, slots 6-8 (val-like)"] < x):.4f} | POOL val-like {np.mean(LAG["train lag view, POOL rows, slots 6-8"] < x):.4f} | val {np.mean(LAG["val"] < x):.4f} || '
      f'train ALL test-like {np.mean(LAG["train lag view, ALL rows, slots 11-13 (test-like)"] < x):.4f} | POOL test-like {np.mean(LAG["train lag view, POOL rows, slots 11-13"] < x):.4f} | public {np.mean(LAG["pub"] < x):.4f} | private {np.mean(LAG["priv"] < x):.4f}')
del tsrow, newest_d, newest_s, msrc, fr_sp, LAG

# ---------------------------------------------------------------- D / E helpers
def psi_from(pa, pb):
    pa = np.maximum(pa, 1e-6); pb = np.maximum(pb, 1e-6); return float(((pa - pb) * np.log(pa / pb)).sum())
class Bins:
    def __init__(self, val, kind):
        self.kind = kind
        if kind == 'spec':      # BUILD_SPEC C4, literal on the file values: 10 bins at the val deciles of the non-zero values + one bin for exact zero
            self.ed = np.unique(np.quantile(val[val != 0], np.linspace(0.1, 0.9, 9))); self.n = len(self.ed) + 2
        else:                   # the critic's advance read (ledger A2): 9 decile edges over all val values
            self.ed = np.unique(np.quantile(val, np.linspace(0.1, 0.9, 9))); self.n = len(self.ed) + 1
    def hist(self, x):
        b = np.searchsorted(self.ed, x, side='right')
        if self.kind == 'spec': b = np.where(x == 0, 0, 1 + b)
        return np.bincount(b, minlength=self.n) / len(x)
kn = hc.knots()
bv = hc.age_bucket(hc.age_hours(np.load(f'{FLAT}/val/cand_f4.npy'), kn)); be = hc.age_bucket(hc.age_hours(np.load(f'{FLAT}/test/cand_f4.npy'), kn))
f4t = np.load(f'{FLAT}/train/cand_f4.npy', mmap_mode='r')
vbig = np.repeat(VA.M >= 50, VA.M); ebig = np.repeat(TE.M >= 50, TE.M)
def rows_of(fm):
    return np.nonzero(np.repeat(fm, Mt))[0]

P('=== D. non-zero share of the walk and PSI of the two installed files BY TRAIN DAY against val / public / private (PSI: BUILD_SPEC bins | critic bins)')
P(f'  references: non-zero share val {nz_v.mean():.4f} (slot 16 {nz_v[vsl == 16].mean():.4f}, slot 18 {nz_v[vsl == 18].mean():.4f}); public {nz_e[epub].mean():.4f}; private {nz_e[~epub].mean():.4f}; test slot 20 / 21 / 22 / 23 '
  + ' / '.join(f'{nz_e[esl == s].mean():.4f}' for s in (20, 21, 22, 23)))
BIN = {(nm, kd): Bins(X['val'], kd) for nm, X in (('z', ZI), ('rk', RI)) for kd in ('spec', 'crit')}
REFH = {(nm, kd, rf): BIN[(nm, kd)].hist(xx) for nm, X in (('z', ZI), ('rk', RI)) for kd in ('spec', 'crit') for rf, xx in (('val', X['val']), ('pub', X['test'][epub]), ('priv', X['test'][~epub]))}
def psi_line(r):
    out = []
    for nm, X in (('z', ZI), ('rk', RI)):
        a = X['train'][r]; cells = []
        for kd in ('spec', 'crit'):
            h = BIN[(nm, kd)].hist(a); cells.append(' / '.join(f'{psi_from(h, REFH[(nm, kd, rf)]):.4f}' for rf in ('val', 'pub', 'priv')))
        out.append(f'{nm}: {cells[0]} | {cells[1]}')
    return '; '.join(out)
for popn, fm in (('ALL', np.ones(TR.n, bool)), ('POOL', pool), ('HOLDOUT', ish)):
    for s in (6, 7, 8, 11, 12, 13):
        r = rows_of(fm & (TR.slot == s)); P(f'  {popn:7s} slot {s:2d} (Sep {6 + s}): feeds {int((fm & (TR.slot == s)).sum())}, rows {len(r)}, non-zero {nz_tr[r].mean():.4f}; PSI vs val / public / private  {psi_line(r)}')
    if popn == 'ALL':
        for s in (9, 14):
            r = rows_of(fm & (TR.slot == s)); P(f'  {popn:7s} slot {s:2d} (Sep {6 + s}): feeds {int((fm & (TR.slot == s)).sum())}, rows {len(r)}, non-zero {nz_tr[r].mean():.4f} (stragglers)')

P('=== E. label-free STEP 0 gates on the installed files (non-zero indicator = my own walk > 0)')
def gap(view_nz, view_b, ref_nz, ref_b):
    w = np.bincount(view_b, minlength=7) / len(view_b)
    zv = np.array([view_nz[view_b == k].mean() if (view_b == k).any() else 0.0 for k in range(7)]); zr = np.array([ref_nz[ref_b == k].mean() if (ref_b == k).any() else 0.0 for k in range(7)])
    g = np.abs(zv - zr); big = np.where(w >= 0.05, g, 0.0)
    return float((w * g).sum()), float(big.max()), hc.AGE_NAMES[int(big.argmax())], w
REFS = {'PRIVATE': (nz_e[~epub & ebig], be[~epub & ebig]), 'PUBLIC': (nz_e[epub & ebig], be[epub & ebig]), 'VAL': (nz_v[vbig], bv[vbig])}
G = {}
for popn, fm in (('POOL feeds (BUILD_SPEC C3; decides)', pool), ('HOLDOUT feeds (the critic\'s population)', ish), ('ALL has = 1 feeds (comparison)', np.ones(TR.n, bool))):
    for regn, lo, hi, rf in (('test-like slots 11-13', 11, 13, 'PRIVATE'), ('test-like slots 11-13', 11, 13, 'PUBLIC'), ('val-like slots 6-8', 6, 8, 'VAL')):
        r = rows_of(fm & (TR.slot >= lo) & (TR.slot <= hi)); b = hc.age_bucket(hc.age_hours(np.asarray(f4t[r]), kn))
        dZ, big, bn, w = gap(nz_tr[r], b, *REFS[rf]); G[(popn[:4], regn, rf)] = (dZ, big)
        P(f'  C3 {popn} | {regn} ({len(r)} rows) vs {rf} rows of feeds with M >= 50: non-zero {nz_tr[r].mean():.4f} vs {REFS[rf][0].mean():.4f}; dZ {dZ:.4f} (bar 0.05), largest bucket {big:.4f} at {bn} (bar 0.10)')
c3 = all(G[('POOL', a, b)][0] <= 0.05 and G[('POOL', a, b)][1] <= 0.10 for a, b in (('test-like slots 11-13', 'PRIVATE'), ('test-like slots 11-13', 'PUBLIC'), ('val-like slots 6-8', 'VAL')))
P(f'  C3 (pool feeds): {"PASS" if c3 else "FAIL"}')
c4 = {}
for popn, fm in (('POOL feeds with has = 1 (BUILD_SPEC C4; decides)', pool & has), ('HOLDOUT feeds with has = 1 (the critic\'s population)', ish & has), ('ALL has = 1 train rows (the ledger\'s shorter wording read literally)', has)):
    r = rows_of(fm)
    for nm, X in (('nx_rp3_z', ZI), ('nx_rp3_rk', RI)):
        a = X['train'][r]
        for kd, kdn in (('spec', 'BUILD_SPEC bins'), ('crit', 'critic bins')):
            h = BIN[(nm[7:], kd)].hist(a); v = [psi_from(h, REFH[(nm[7:], kd, rf)]) for rf in ('val', 'pub', 'priv')]; c4[(popn[:4], nm, kd)] = v
            P(f'  C4 {popn} ({len(r)} rows) | {nm} | {kdn}: PSI vs val {v[0]:.4f}, public {v[1]:.4f}, private {v[2]:.4f} (bar 0.25){"   ABOVE THE BAR" if max(v) > 0.25 else ""}')
for nm, X in (('nx_rp3_z', ZI), ('nx_rp3_rk', RI)):
    P(f'  C4 information | {nm} | BUILD_SPEC bins: val against private {psi_from(REFH[(nm[7:], "spec", "val")], REFH[(nm[7:], "spec", "priv")]):.4f}, val against public {psi_from(REFH[(nm[7:], "spec", "val")], REFH[(nm[7:], "spec", "pub")]):.4f}')
c4ok = all(max(c4[('POOL', nm, 'spec')]) <= 0.25 for nm in ('nx_rp3_z', 'nx_rp3_rk'))
c4all = all(max(c4[('ALL ', nm, 'spec')]) <= 0.25 for nm in ('nx_rp3_z', 'nx_rp3_rk'))
P(f'  C4 (pool feeds, BUILD_SPEC bins): {"PASS" if c4ok else "FAIL"};   C4 if ALL has = 1 rows were the population: {"PASS" if c4all else "FAIL"}')
vref = max(nz_v.mean(), nz_v[vsl == 16].mean(), nz_v[vsl == 18].mean()); tmin = min(nz_e[esl == s].mean() for s in (20, 21, 22, 23)); c5 = tmin >= 0.5 * vref
P(f'  C5 lowest non-zero share of a test day {tmin:.4f} against 0.5 x val\'s largest ({vref:.4f}) = {0.5 * vref:.4f}: {"PASS" if c5 else "FAIL"}')
P(f'  G-parity (C3 pool, C4 pool, C5): {"PASS" if (c3 and c4ok and c5) else "FAIL"}')
P('  J1 (no bar) P(walk = 0 | seq_cv_invd20 > 0), stored seq_cv_invd20 of nn/feat_cache:')
cvt = np.load(f'{FC}/train/seq_cv_invd20.npy', mmap_mode='r'); cvv = np.load(f'{FC}/val/seq_cv_invd20.npy'); cve = np.load(f'{FC}/test/seq_cv_invd20.npy')
for nm, r in (('holdout lag view slots 6-8', rows_of(ish & (TR.slot >= 6) & (TR.slot <= 8))), ('holdout lag view slots 11-13', rows_of(ish & (TR.slot >= 11) & (TR.slot <= 13))),
              ('all lag-view train rows slots 6-9', rows_of(has & (TR.slot <= 9))), ('all lag-view train rows slots 11-14', rows_of(has & (TR.slot >= 11))), ('pool lag-view rows', rows_of(pool & has))):
    i_ = np.asarray(cvt[r]) > 0; w_ = nz_tr[r]
    P(f'    {nm}: rows {len(r)}, walk > 0 {w_.mean():.4f}, seq_cv_invd20 > 0 {i_.mean():.4f}, P(walk = 0 | invd20 > 0) {1 - w_[i_].mean():.4f}, P(walk > 0 | invd20 = 0) {w_[~i_].mean():.4f}')
for nm, w_, i_ in (('val', nz_v, cvv > 0), ('public', nz_e[epub], cve[epub] > 0), ('private', nz_e[~epub], cve[~epub] > 0)):
    P(f'    {nm}: rows {len(w_)}, walk > 0 {w_.mean():.4f}, seq_cv_invd20 > 0 {i_.mean():.4f}, P(walk = 0 | invd20 > 0) {1 - w_[i_].mean():.4f}, P(walk > 0 | invd20 = 0) {w_[~i_].mean():.4f}')

# ---------------------------------------------------------------- F
P('=== F. alignment invariants on the installed files alone (they do not use my walk)')
maxid = 111319574
def feedmin(x, off):
    return np.minimum.reduceat(x, off)
def inv(tag, z, rk, off, M, absent_row, cv_pos):
    """absent_row: candidate item occurs in NO sequence of the feed's source universe -> the walk is 0 -> the value must be the feed minimum (both files)"""
    q = np.repeat(np.arange(len(M)), M)
    rs = np.add.reduceat(rk.astype(np.float64), off)
    zm = feedmin(z, off)[q]; rm = feedmin(rk, off)[q]
    v1 = int((np.abs(rs) > 1e-3).sum()); v2 = int((absent_row & ((z != zm) | (rk != rm))).sum())
    low = (rk == rm)                                   # rows in the lowest tie group = "walk 0" unless the feed has no zero
    j = float(np.mean(low[cv_pos])) if cv_pos.any() else float('nan')
    P(f'  {tag}: feeds {len(M)}, rows {len(z)} | feeds with |sum of rank| > 1e-3: {v1} | rows whose item is in no source sequence: {int(absent_row.sum())} ({absent_row.mean():.4f}), of them NOT at the feed minimum: {v2} | P(lowest rank group | seq_cv_invd20 > 0) {j:.4f}')
    return v1, v2, j
def shifted(x, k):
    return np.roll(x, k)
# presence of items in the source universes (my own member selection; all sparse members counted whatever their time: a superset of the admissible sources)
spf = sparse_flag(TR.fid)
def presence(parts):
    pr = np.zeros(maxid + 1, bool)
    for S, sel in parts:
        sel = np.asarray(sel, np.int64); h = S.H[sel]; off = np.concatenate([[0], np.cumsum(h)[:-1]]).astype(np.int64); ev = np.empty(int(h.sum()), np.int32); _gather(sel, S.ho, S.H, S.hist, off, ev); pr[ev] = True
    return pr
L = np.load(ROOT + '/nn_x/w12/stale/hz2/feats_L/train/hz2_lag.npy').astype(np.int64); cut = np.maximum(TR.slot - L, 5)
absent = np.zeros(NROWS['train'], bool)
for c in np.unique(cut[has]):
    pr = presence([(TR, dense_members(TR, int(c))), (TR, np.nonzero(spf & (TR.slot >= c))[0])])
    r = rows_of(has & (cut == c)); it = np.asarray(TR.cand[r]); absent[r] = ~pr[it]; del pr, r, it
zi = ZI['train'][hrows]; ri = RI['train'][hrows]; ab = absent[hrows]; cvp = np.asarray(cvt[hrows]) > 0
inv('train has = 1 rows, as installed', zi, ri, offh, Mh, ab, cvp)
inv('train has = 1 rows, array shifted by 1 row (power)', shifted(zi, 1), shifted(ri, 1), offh, Mh, ab, cvp)
inv('train has = 1 rows, array shifted by 137 rows (about one feed; power)', shifted(zi, 137), shifted(ri, 137), offh, Mh, ab, cvp)
del zi, ri, ab, cvp, absent
VAh = Split('val'); TEh = Split('test')
prn = presence([(TR, dense_members(TR, 15)), (VAh, np.arange(VAh.n)), (TEh, np.arange(TEh.n))])
for s, S, cv in (('val', VAh, cvv), ('test', TEh, cve)):
    ab = ~prn[np.asarray(S.cand[:])]
    inv(f'{s}, as installed', ZI[s], RI[s], S.co, S.M, ab, cv > 0)
    inv(f'{s}, array shifted by 1 row (power)', shifted(ZI[s], 1), shifted(RI[s], 1), S.co, S.M, ab, cv > 0)
P('DONE')
