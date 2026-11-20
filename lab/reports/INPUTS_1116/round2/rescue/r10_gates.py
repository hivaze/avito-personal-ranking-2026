"""RESCUE r10: the gates of PREDECLARED.txt section 3 on the D5 builds (form of nn_x/w12/nx3/nx3_gates.py; bars unchanged).
  PART 1 (label-free)  C3 non-default share by listing-age bucket (dZ <= 0.05, largest bucket <= 0.10), C4 PSI of the entered forms (<= 0.25), C5 by test day.
  PART 2 (labels)      score test of condlib.Null with the null model = pool + control block of the ranker's own inputs + nx_rp3_z + nx_rp3_rk:
                       witness H = holdout rows of the 15,225 lag-view feeds; witness T = val rows with the stale-val twin of the column (TO16).
                       Printed next to it (information): the same reads without nx3 in the null model, with the nx3 twin in it, on the FILE base,
                       the natural val column, the regions, the critic's nx3-style lag view of bt.
Transforms of the entered forms, fixed here before any read: identity for every column except ns2_5 (log1p); flags are entered as the within-feed z of
the 0 / 1 value; z = lens.feed_z (NaN -> 0, clip 5), rk = lens.feed_rk (NaN -> 0).
usage: python r10_gates.py <tag> <piece,piece,...>     pieces: bt,coec,dens,surf,gap,user,als,alsp1
CPU only, read-only. Output out/r10_gates_<tag>.txt and out/r10_gates_<tag>.json (exclusive create)."""
import json
from rs_common import *
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows')
_argv = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
import lens
import condlib
sys.argv = _argv
nb.set_num_threads(NT)
TAG = sys.argv[1]; PIECES = sys.argv[2].split(',')
P = Log(OUT / f'r10_gates_{TAG}.txt')
V = Views(P)
# piece -> list of (column, transform, [(form, sign on record)]) ; first column's z form is the placebo column ; nd = non-default indicator
SPEC = {
    'bt':   dict(name='P1 click strength bt', cols=[('bt5', None, [('z', +1), ('rk', +1)]), ('bt5_seen', 'flag', [('z', -1)])], nd=lambda g: g('bt5_seen') > 0),
    'coec': dict(name='P2 all-feed click rate', cols=[('coec5', None, [('z', +1), ('rk', +1)])], nd=lambda g: g('imp5') > 0, info=[('ctr5', None)]),
    'dens': dict(name='P3 density', cols=[('kn1_5', None, [('z', -1), ('rk', -1)]), ('kn20_5', None, [('z', -1), ('rk', -1)])], nd='median:kn1_5'),
    'surf': dict(name='P4 surface split', cols=[('X3_5', None, [('z', +1), ('rk', +1)]), ('ns2_5', 'log1p', [('z', -1), ('rk', -1)])], nd=lambda g: (g('ns1_5') + g('ns2_5')) > 0),
    'gap':  dict(name='P5 first-seen gap', cols=[('gapf5', None, [('z', -1), ('rk', -1)]), ('never5', 'flag', [('z', 0)])], nd=lambda g: g('imp5') > 0),
    'user': dict(name='P6 same-user earlier feeds', cols=[('uc5', None, [('z', +1), ('rk', +1)]), ('reshown5', 'flag', [('z', -1)])], nd=lambda g: ~np.isnan(g('uc5'))),
    'als':  dict(name='P7 ALS', cols=[('als5', None, [('z', +1), ('rk', +1)])], nd=lambda g: ~np.isnan(g('als5'))),
    'alsp1': dict(name='P7 ALS (first pass: windows of cut 5, natural, twin cuts 9-11)', cols=[('als5p1', None, [('z', +1), ('rk', +1)])], nd=lambda g: ~np.isnan(g('als5p1'))),
}
QS = ('hold', 'pool', 'val_nat', 'val_twin', 'test_nat')
R = {q: V.rows(q) for q in QS}
OFF = {q: R[q][2] for q in QS}; MM = {q: R[q][3] for q in QS}
VIEW = {q: V.row_wkey(q) > 0 for q in QS}


def tr(x, t):
    x = np.array(x, np.float64); x[~np.isfinite(x)] = np.nan       # as lens.forms: a non-finite value is a missing value
    if t == 'log1p':
        x = np.log1p(np.maximum(x, 0))
    return x


_FC = {}


def form(q, col, t, fm):
    k = (q, col, t, fm)
    if k not in _FC:
        x = tr(get(q, col), t)
        _FC[k] = lens.feed_z(x, OFF[q], MM[q], 5.0) if fm == 'z' else lens.feed_rk(x, OFF[q], MM[q])
    return _FC[k]


# ------------------------------------------------------------------ PART 1: G-parity (label-free)
P(f'=== PART 1. G-parity (label-free); pieces {PIECES} ===')
kn = hc.knots()
F4 = {s: np.load(FLAT / s / 'cand_f4.npy', mmap_mode='r') for s in SPLITS}
BK = {q: hc.age_bucket(hc.age_hours(np.asarray(F4[V.Q[q][0]][R[q][0]]), kn)) for q in QS}
SL = {q: np.repeat(V.A[V.Q[q][0]]['slot'][V.Q[q][1]], MM[q]) for q in QS}
BIG = {q: np.repeat(MM[q] >= 50, MM[q]) for q in QS}
epub = np.repeat(V.A['test']['ts'] < hc.PUB_TS, MM['test_nat'])
P('age buckets: ' + str(hc.AGE_NAMES))


def gap(view_nz, view_b, ref_nz, ref_b):
    w = np.bincount(view_b, minlength=7) / len(view_b)
    zv = np.array([view_nz[view_b == k].mean() if (view_b == k).any() else 0.0 for k in range(7)]); zr = np.array([ref_nz[ref_b == k].mean() if (ref_b == k).any() else 0.0 for k in range(7)])
    g = np.abs(zv - zr); big = np.where(w >= 0.05, g, 0.0)
    return float((w * g).sum()), float(big.max()), hc.AGE_NAMES[int(big.argmax())], zv, zr, w


def psi_bins(a, b, binf, nbin):
    pa = np.bincount(binf(a), minlength=nbin) / len(a); pb = np.bincount(binf(b), minlength=nbin) / len(b)
    pa = np.maximum(pa, 1e-6); pb = np.maximum(pb, 1e-6); return float(((pa - pb) * np.log(pa / pb)).sum())


def spec_binner(val):
    ed = np.unique(np.quantile(val[val != 0], np.linspace(0.1, 0.9, 9)))
    return (lambda x: np.where(x == 0, 0, 1 + np.searchsorted(ed, x, side='right'))), len(ed) + 2


RES = {}
for pc in PIECES:
    S = SPEC[pc]; P(f'--- {S["name"]}')
    if isinstance(S['nd'], str):
        col = S['nd'].split(':')[1]; med = float(np.nanmedian(get('val_nat', col)))
        ND = {q: np.nan_to_num(get(q, col), nan=-9.0) > med for q in QS}
        P(f'  non-default substitute: {col} above the natural-val median {med:.4f}')
    else:
        ND = {q: np.asarray(S['nd'](lambda n, q=q: get(q, n))) & VIEW[q] for q in QS}
    refs = (('PRIVATE rows (M >= 50)', ND['test_nat'][~epub & BIG['test_nat']], BK['test_nat'][~epub & BIG['test_nat']], 11, 13),
            ('PUBLIC rows (M >= 50)', ND['test_nat'][epub & BIG['test_nat']], BK['test_nat'][epub & BIG['test_nat']], 11, 13),
            ('VAL rows (M >= 50)', ND['val_nat'][BIG['val_nat']], BK['val_nat'][BIG['val_nat']], 6, 8))
    c3 = True; c3rows = []
    for popn, q, official in (('POOL feeds [decides]', 'pool', True), ('HOLDOUT feeds (comparison)', 'hold', False)):
        for refn, ref_nz, ref_b, lo, hi in refs:
            m = (SL[q] >= lo) & (SL[q] <= hi) & VIEW[q]
            dZ, big, bn, zv, zr, w = gap(ND[q][m], BK[q][m], ref_nz, ref_b); ok = dZ <= 0.05 and big <= 0.10
            P(f'  C3 {popn}, slots {lo}-{hi} ({int(m.sum())} rows) vs {refn}: share {ND[q][m].mean():.4f} vs {ref_nz.mean():.4f}; dZ {dZ:.4f} ({"ok" if dZ <= 0.05 else "FAIL"}), largest bucket {big:.4f} at {bn} ({"ok" if big <= 0.10 else "FAIL"})')
            P('       view by bucket ' + ' '.join(f'{x:.3f}' for x in zv) + ' | ref ' + ' '.join(f'{x:.3f}' for x in zr) + ' | view row share ' + ' '.join(f'{x:.3f}' for x in w))
            if official:
                c3 &= ok; c3rows.append((refn.split(' rows')[0], round(dZ, 4), round(big, 4), float(ND[q][m].mean()), float(ref_nz.mean())))
    dZ, big, bn, zv, zr, w = gap(ND['val_twin'][BIG['val_twin']], BK['val_twin'][BIG['val_twin']], refs[0][1], refs[0][2])
    P(f'  information: stale-val twin (val M >= 50) vs PRIVATE: share {ND["val_twin"][BIG["val_twin"]].mean():.4f} vs {refs[0][1].mean():.4f}; dZ {dZ:.4f}, largest bucket {big:.4f} at {bn}')
    vsh = {s: float(ND['val_nat'][SL['val_nat'] == s].mean()) for s in (16, 18)}; vall = float(ND['val_nat'].mean()); esh = {s: float(ND['test_nat'][SL['test_nat'] == s].mean()) for s in (20, 21, 22, 23)}
    vref = max(vall, vsh[16], vsh[18]); c5 = all(v >= 0.5 * vref for v in esh.values())
    P(f'  C5 non-default share by day: val slot 16 / 18 / all {vsh[16]:.4f} / {vsh[18]:.4f} / {vall:.4f} | test slot 20 / 21 / 22 / 23 ' + ' / '.join(f'{esh[s]:.4f}' for s in (20, 21, 22, 23))
      + f' | public {ND["test_nat"][epub].mean():.4f} private {ND["test_nat"][~epub].mean():.4f} | holdout lag view slot 6 / 7 / 8 / 11 / 12 / 13 ' + ' / '.join(f'{ND["hold"][(SL["hold"] == s) & VIEW["hold"]].mean():.4f}' for s in (6, 7, 8, 11, 12, 13))
      + f' | val twin {ND["val_twin"].mean():.4f} | bar {0.5 * vref:.4f}: {"ok" if c5 else "FAIL"}')
    c4 = True; c4rows = {}
    for col, t, fms in S['cols']:
        if t == 'flag':
            continue
        for fm in ('z', 'rk'):
            fv, fe, fp, fh, ft = form('val_nat', col, t, fm), form('test_nat', col, t, fm), form('pool', col, t, fm), form('hold', col, t, fm)[VIEW['hold']], form('val_twin', col, t, fm)
            binf, nbin = spec_binner(fv)
            v = (psi_bins(fp, fv, binf, nbin), psi_bins(fp, fe[epub], binf, nbin), psi_bins(fp, fe[~epub], binf, nbin))
            h = (psi_bins(fh, fv, binf, nbin), psi_bins(fh, fe[epub], binf, nbin), psi_bins(fh, fe[~epub], binf, nbin))
            ok = max(v) <= 0.25; c4 &= ok; c4rows[f'{col}:{fm}'] = [round(x, 4) for x in v]
            P(f'  C4 PSI {col}:{fm} | pool has = 1 rows vs val / public / private: {v[0]:.4f} / {v[1]:.4f} / {v[2]:.4f} {"ok" if ok else "FAIL"} | holdout has = 1 rows (comparison): {h[0]:.4f} / {h[1]:.4f} / {h[2]:.4f}'
              f' | information: val vs private {psi_bins(fv, fe[~epub], binf, nbin):.4f}, twin vs private {psi_bins(ft, fe[~epub], binf, nbin):.4f}; sd pool / val / test / twin {fp.std():.3f} / {fv.std():.3f} / {fe.std():.3f} / {ft.std():.3f}')
    RES[pc] = dict(name=S['name'], C3=bool(c3), C3rows=c3rows, C4=bool(c4), C4rows=c4rows, C5=bool(c5), C5min=min(esh.values()), C5bar=0.5 * vref, parity=bool(c3 and c4 and c5))
    P(f'  G-parity {S["name"]}: C3 {c3}, C4 {c4}, C5 {c5} -> {"PASS" if (c3 and c4 and c5) else "FAIL"}')
del F4

# ------------------------------------------------------------------ PART 2: G-info (labels are opened from here on)
P('\n=== PART 2. G-info: score test given pool + own inputs + nx_rp3_z + nx_rp3_rk; bar |cz| >= 2.5 with the sign on record on H and on T ===')
P(f'lens md5 {md5("<repo>/reports/INPUTS_1116/residual_rows/lens.py")}, condlib md5 {md5("<repo>/reports/INPUTS_1116/residual_rows/condlib.py")}')
tslot = V.A['train']['slot']; Mt = V.A['train']['M']
rows_h = R['hold'][0]; slot_h = SL['hold']; hasrow = VIEW['hold']
NXZ = np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r'); NXR = np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')
nxz_h = np.nan_to_num(np.asarray(NXZ[rows_h]).astype(np.float32), nan=0.0); nxr_h = np.nan_to_num(np.asarray(NXR[rows_h]).astype(np.float32), nan=0.0)
assert (np.isnan(np.asarray(NXZ[rows_h])) == ~hasrow).all()
nxz_v = np.load(ROOT / 'data/feats/val/nx_rp3_z.npy').astype(np.float32); nxr_v = np.load(ROOT / 'data/feats/val/nx_rp3_rk.npy').astype(np.float32)
nxz_t = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nxr_t = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
READS = {}      # (piece, col:form) -> dict of cz per read
PLAC = {'H': [], 'T': []}


def score_all(N, q, keyname, region=None, placebo_seed=None, D=None):
    """every pre-declared column-form of the pieces on query set q under null model N; region = row mask (others set to 0)"""
    for pc in PIECES:
        S = SPEC[pc]
        for ci_, (col, t, fms) in enumerate(S['cols']):
            for fm, sg in fms:
                x = form(q, col, None if t == 'flag' else t, fm)
                if region is not None:
                    x = np.where(region, x, 0.0)
                r = N.score(x.astype(np.float32))
                READS.setdefault((pc, f'{col}:{fm}'), {})[keyname] = dict(cz=float(r['cz']), b1=float(r['b1']), dnats=float(r['dnats']), r2=float(r['r2']))
                if placebo_seed is not None and ci_ == 0 and fm == 'z':
                    pz = [float(N.score(condlib.perm_within(x.astype(np.float32), D['off'], D['M'], placebo_seed + s))['cz']) for s in range(20)]
                    PLAC[keyname].extend(pz); READS[(pc, f'{col}:{fm}')][keyname + '_placebo'] = pz
        for col, t in S.get('info', []):
            for fm in ('z', 'rk'):
                x = form(q, col, t, fm)
                if region is not None:
                    x = np.where(region, x, 0.0)
                READS.setdefault((pc, f'[info] {col}:{fm}'), {})[keyname] = dict(cz=float(N.score(x.astype(np.float32))['cz']))


# ---- witness H
D = lens.load_split('hold', 'TO16'); condlib.sp = 'hold'; nm_, cl = condlib.control_block(D)
assert (D['rows'] == rows_h).all() and (D['off'] == OFF['hold']).all()
P(f'[holdout, pool = 16 held-out wide students; control block {len(nm_)} columns; + nx_rp3_z, nx_rp3_rk (lag view, NaN -> 0)]')
N = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk'], cl + [nxz_h, nxr_h])
chk = N.score(nxz_h); P(f'  check: nx_rp3_z under a null model that holds it: cz {chk["cz"]:+.3f} (must be 0), R2 {chk["r2"]:.3f}')
score_all(N, 'hold', 'H', placebo_seed=3000, D=D)
score_all(N, 'hold', 'H slots 6-8', region=(slot_h >= 6) & (slot_h <= 8))
score_all(N, 'hold', 'H slots 11-13', region=slot_h >= 11)
if 'bt' in PIECES:
    B = np.load(ROOT / 'reports/INPUTS_1116/critic/out/c02_bt.npz'); x = np.where(hasrow, B['hold_lag_theta'], np.nan)
    for fm, fn in (('z', lambda x_: lens.feed_z(x_, OFF['hold'], MM['hold'], 5.0)), ('rk', lambda x_: lens.feed_rk(x_, OFF['hold'], MM['hold']))):
        READS.setdefault(('bt', f'[info] critic nx3-style lag theta:{fm}'), {})['H'] = dict(cz=float(N.score(fn(x))['cz']))
del N
N0 = condlib.Null(D, nm_, cl)
c0 = N0.score(nxz_h); P(f'  check: nx_rp3_z under the null model WITHOUT nx3 (record +4.98): cz {c0["cz"]:+.2f}')
score_all(N0, 'hold', 'H without nx3')
del N0, D
# ---- witness T (and the natural val column)
vslot = SL['val_nat']
for base, nxn, keyT, keyN in (('TO16', 'nat', 'T', 'val natural'), ('TO16', 'twin', 'T (nx3 twin in the null)', None), ('FILE', 'nat', 'T on FILE', None)):
    D = lens.load_split('val', base); condlib.sp = 'val'; nm_, cl = condlib.control_block(D)
    add = [nxz_v, nxr_v] if nxn == 'nat' else [nxz_t, nxr_t]
    N = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk'], cl + add)
    P(f'[val, base {base}; control block {len(nm_)} columns; + nx_rp3_z, nx_rp3_rk ({"data/feats/val" if nxn == "nat" else "stale-val twin"})]')
    score_all(N, 'val_twin', keyT, placebo_seed=4000 if keyT == 'T' else None, D=D)
    if keyN:
        score_all(N, 'val_nat', keyN)
        score_all(N, 'val_twin', 'T slot 16', region=vslot == 16); score_all(N, 'val_twin', 'T slot 18', region=vslot == 18)
        if 'bt' in PIECES:
            x = B['val_twin_theta']
            for fm, fn in (('z', lambda x_: lens.feed_z(x_, OFF['val_twin'], MM['val_twin'], 5.0)), ('rk', lambda x_: lens.feed_rk(x_, OFF['val_twin'], MM['val_twin']))):
                READS[('bt', f'[info] critic nx3-style lag theta:{fm}')]['T'] = dict(cz=float(N.score(fn(np.asarray(x, np.float64)))['cz']))
        del N
        N0 = condlib.Null(D, nm_, cl); score_all(N0, 'val_twin', 'T without nx3'); score_all(N0, 'val_nat', 'val natural without nx3'); del N0
    else:
        del N
    del D
# ------------------------------------------------------------------ tables
pH, pT = np.array(PLAC['H']), np.array(PLAC['T'])
P(f'\nplacebo (within-feed permutations of each piece\'s first z column, 20 each): H: n {len(pH)}, sd {pH.std():.2f}, max |cz| {np.abs(pH).max():.2f} | T: n {len(pT)}, sd {pT.std():.2f}, max |cz| {np.abs(pT).max():.2f}')
thrH = 2.5 * max(1.0, pH.std()); thrT = 2.5 * max(1.0, pT.std())
P(f'threshold: |cz| >= {thrH:.2f} on H and >= {thrT:.2f} on T (2.5 x max(1, placebo sd)), same sign as the record, and above every placebo of the witness ({np.abs(pH).max():.2f} / {np.abs(pT).max():.2f})')
keys = ['H', 'T', 'H without nx3', 'T without nx3', 'T (nx3 twin in the null)', 'T on FILE', 'val natural', 'val natural without nx3', 'H slots 6-8', 'H slots 11-13', 'T slot 16', 'T slot 18']
P('\ncolumn:form | sign on record | ' + ' | '.join(keys) + ' | R2 with the null model (H / T) | dnats (H / T) | passes')
nlook = 0
for pc in PIECES:
    S = SPEC[pc]; signs = {f'{col}:{fm}': sg for col, t, fms in S['cols'] for fm, sg in fms}; any_pass = []
    for (p_, cf), rd in READS.items():
        if p_ != pc:
            continue
        cells = [f'{rd[k]["cz"]:+.2f}' if k in rd else '-' for k in keys]
        if cf in signs:
            nlook += 2; sg = signs[cf]; h, t_ = rd['H']['cz'], rd['T']['cz']
            okH = abs(h) >= thrH and abs(h) > np.abs(pH).max() and (sg == 0 or np.sign(h) == sg)
            okT = abs(t_) >= thrT and abs(t_) > np.abs(pT).max() and (sg == 0 or np.sign(t_) == sg)
            ok = okH and okT and np.sign(h) == np.sign(t_)
            if ok:
                any_pass.append(cf)
            P(f'{pc} {cf} | {"+" if sg > 0 else ("-" if sg < 0 else "any")} | ' + ' | '.join(cells) + f' | {rd["H"]["r2"]:.2f} / {rd["T"]["r2"]:.2f} | {rd["H"]["dnats"]:.5f} / {rd["T"]["dnats"]:.5f} | H {"ok" if okH else "no"}, T {"ok" if okT else "no"} -> {"PASS" if ok else "no"}')
        else:
            P(f'{pc} {cf} | information | ' + ' | '.join(cells))
    RES[pc]['info_pass'] = any_pass; RES[pc]['reads'] = {cf: rd for (p_, cf), rd in READS.items() if p_ == pc}
    for (p_, cf), rd in READS.items():
        if p_ == pc and 'H_placebo' in rd:
            P(f'   placebo of {cf}: H sd {np.std(rd["H_placebo"]):.2f} max {np.max(np.abs(rd["H_placebo"])):.2f} | T sd {np.std(rd["T_placebo"]):.2f} max {np.max(np.abs(rd["T_placebo"])):.2f}')
P(f'\ndeciding reads in this run: {nlook} (column-forms x 2 witnesses); every other number above is information')
P('\npiece | G-parity (C3 / C4 / C5) | column-forms passing G-info on both witnesses | verdict')
for pc in PIECES:
    r = RES[pc]
    v = 'SURVIVES' if (r['parity'] and r['info_pass']) else ('INFORMATION WITHOUT PARITY' if r['info_pass'] else 'NOT SHOWN')
    r['verdict'] = v
    P(f'{r["name"]} | {"PASS" if r["parity"] else "FAIL"} ({r["C3"]} / {r["C4"]} / {r["C5"]}) | {r["info_pass"] if r["info_pass"] else "none"} | {v}')
RES['_placebo'] = dict(H=PLAC['H'], T=PLAC['T'], thrH=thrH, thrT=thrT); RES['_nlook'] = nlook
with open(OUT / f'r10_gates_{TAG}.json', 'x') as f:
    json.dump(RES, f, indent=1, default=float)
P('DONE')
