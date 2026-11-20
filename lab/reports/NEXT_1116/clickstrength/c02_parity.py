"""Lane A c02: gate g3 of RULE.txt (parity; LABEL-FREE: no label array and no prediction file is opened here).
  C3p test-like pool rows vs PRIVATE rows, C3v val-like pool rows vs VAL rows: non-default (e > 0) share by listing-age bucket, dZ <= 0.05, ALL 7 buckets <= 0.10
  C3u test-like pool rows vs PUBLIC rows: bar of record (bucket counted only with view row share >= 0.05); all-bucket value printed
  C4  PSI <= 0.25 of each entered value, pool has = 1 rows vs val / public / private (binner of r10_gates.py)
  C5  non-default share on every test day >= 0.5 x val's largest
usage: c02_parity.py   output out/c02_parity.txt, out/c02_parity.json"""
import json
from cs_common import *
P = Log(OUT / 'c02_parity.txt')
P('RULE.txt md5 ' + md5(CS + '/RULE.txt'))
V = Views(P)
FORMS = {'F1 cs3a_c': ('cs3a_c', 'cs3a_e'), 'F2 cs3x_c': ('cs3x_c', 'cs3x_e')}
R = {q: V.rows(q) for q in QS}; MM = {q: R[q][3] for q in QS}
VIEW = {q: V.row_e(q) >= 0 for q in QS}
kn = hc.knots()
F4 = {s: np.load(FLAT / s / 'cand_f4.npy', mmap_mode='r') for s in SPLITS}
BK = {q: hc.age_bucket(hc.age_hours(np.asarray(F4[V.Q[q][0]][R[q][0]]), kn)) for q in QS}
SL = {q: V.row_slot(q) for q in QS}
LAG = {q: np.load(arr_path(q, 'lag_row')).astype(np.int64) for q in QS}
BIG = {q: np.repeat(MM[q] >= 50, MM[q]) for q in QS}
epub = np.repeat(V.A['test']['ts'] < hc.PUB_TS, MM['test_nat'])
P('age buckets: ' + str(hc.AGE_NAMES))


def gap(view_nz, view_b, ref_nz, ref_b):
    w = np.bincount(view_b, minlength=7) / len(view_b)
    zv = np.array([view_nz[view_b == k].mean() if (view_b == k).any() else 0.0 for k in range(7)]); zr = np.array([ref_nz[ref_b == k].mean() if (ref_b == k).any() else 0.0 for k in range(7)])
    g = np.abs(zv - zr); big = np.where(w >= 0.05, g, 0.0)
    return float((w * g).sum()), float(g.max()), hc.AGE_NAMES[int(g.argmax())], float(big.max()), hc.AGE_NAMES[int(big.argmax())], zv, zr, w


def psi_bins(a, b, binf, nbin):
    pa = np.bincount(binf(a), minlength=nbin) / len(a); pb = np.bincount(binf(b), minlength=nbin) / len(b)
    pa = np.maximum(pa, 1e-6); pb = np.maximum(pb, 1e-6); return float(((pa - pb) * np.log(pa / pb)).sum())


def spec_binner(val):
    ed = np.unique(np.quantile(val[val != 0], np.linspace(0.1, 0.9, 9)))
    return (lambda x: np.where(x == 0, 0, 1 + np.searchsorted(ed, x, side='right'))), len(ed) + 2


RES = {}
for fn, (col, ecol) in FORMS.items():
    P(f'--- {fn}: non-default indicator = {ecol} > 0')
    ND = {q: (np.nan_to_num(get(q, ecol)) > 0) & VIEW[q] for q in QS}
    tp = ~epub & BIG['test_nat']; tu = epub & BIG['test_nat']; vb = BIG['val_nat']
    r = {}
    for key, popq, lo, hi, refn, rnz, rb, noex in (('C3p', 'pool', 11, 13, 'PRIVATE rows (M >= 50)', ND['test_nat'][tp], BK['test_nat'][tp], True),
                                                 ('C3v', 'pool', 6, 8, 'VAL rows (M >= 50)', ND['val_nat'][vb], BK['val_nat'][vb], True),
                                                 ('C3u', 'pool', 11, 13, 'PUBLIC rows (M >= 50)', ND['test_nat'][tu], BK['test_nat'][tu], False),
                                                 ('cmp hold C3p', 'hold', 11, 13, 'PRIVATE rows (M >= 50)', ND['test_nat'][tp], BK['test_nat'][tp], None),
                                                 ('cmp hold C3v', 'hold', 6, 8, 'VAL rows (M >= 50)', ND['val_nat'][vb], BK['val_nat'][vb], None),
                                                 ('cmp hold C3u', 'hold', 11, 13, 'PUBLIC rows (M >= 50)', ND['test_nat'][tu], BK['test_nat'][tu], None)):
        m = (SL[popq] >= lo) & (SL[popq] <= hi) & VIEW[popq]
        dZ, gall, ball, gex, bex, zv, zr, w = gap(ND[popq][m], BK[popq][m], rnz, rb)
        if noex is True:
            ok = dZ <= 0.05 and gall <= 0.10; verdict = f'dZ {dZ:.4f} ({"ok" if dZ <= 0.05 else "FAIL"}), largest of ALL 7 buckets {gall:.4f} at {ball} ({"ok" if gall <= 0.10 else "FAIL"}) [no exemption; with the 5% exemption {gex:.4f}]'
        elif noex is False:
            ok = dZ <= 0.05 and gex <= 0.10; verdict = f'dZ {dZ:.4f} ({"ok" if dZ <= 0.05 else "FAIL"}), largest bucket with row share >= 0.05 {gex:.4f} at {bex} ({"ok" if gex <= 0.10 else "FAIL"}) [bar of record]; all-bucket value (no bar) {gall:.4f} at {ball}'
        else:
            ok = None; verdict = f'dZ {dZ:.4f}, largest of all buckets {gall:.4f} at {ball}, with the 5% exemption {gex:.4f} (comparison, no bar)'
        P(f'  {key} {popq} slots {lo}-{hi} ({int(m.sum())} rows) vs {refn}: share {ND[popq][m].mean():.4f} vs {rnz.mean():.4f}; {verdict}')
        P('       view by bucket ' + ' '.join(f'{x:.3f}' for x in zv) + ' | ref ' + ' '.join(f'{x:.3f}' for x in zr) + ' | view row share ' + ' '.join(f'{x:.3f}' for x in w))
        if ok is not None:
            r[key] = dict(ok=bool(ok), dZ=dZ, all_bucket=gall, all_bucket_at=ball, exempt_bucket=gex, share_view=float(ND[popq][m].mean()), share_ref=float(rnz.mean()))
    dZ, gall, ball, gex, bex, zv, zr, w = gap(ND['val_twin'][BIG['val_twin']], BK['val_twin'][BIG['val_twin']], ND['test_nat'][tp], BK['test_nat'][tp])
    P(f'  information: private-like val twin (val M >= 50) vs PRIVATE: share {ND["val_twin"][BIG["val_twin"]].mean():.4f} vs {ND["test_nat"][tp].mean():.4f}; dZ {dZ:.4f}, largest of all buckets {gall:.4f} at {ball}')
    dZ, gall, ball, gex, bex, zv, zr, w = gap(ND['test_nat'][tu], BK['test_nat'][tu], ND['test_nat'][tp], BK['test_nat'][tp])
    P(f'  information: PUBLIC rows vs PRIVATE rows themselves: share {ND["test_nat"][tu].mean():.4f} vs {ND["test_nat"][tp].mean():.4f}; dZ {dZ:.4f}, largest of all buckets {gall:.4f} at {ball}')
    P('       public by bucket ' + ' '.join(f'{x:.3f}' for x in zv) + ' | private ' + ' '.join(f'{x:.3f}' for x in zr))
    # seen share by lag
    for q in ('pool', 'hold', 'val_nat', 'val_twin', 'test_nat'):
        lg = LAG[q]; P(f'  information: non-default share by lag, {q}: ' + ', '.join(f'{int(l)} d {ND[q][lg == l].mean():.4f} ({int((lg == l).sum())} rows)' for l in np.unique(lg[lg >= 0])))
    vsh = {s: float(ND['val_nat'][SL['val_nat'] == s].mean()) for s in (16, 18)}; vall = float(ND['val_nat'].mean()); esh = {s: float(ND['test_nat'][SL['test_nat'] == s].mean()) for s in (20, 21, 22, 23)}
    vref = max(vall, vsh[16], vsh[18]); c5 = all(v >= 0.5 * vref for v in esh.values())
    P(f'  C5 non-default share by day: val slot 16 / 18 / all {vsh[16]:.4f} / {vsh[18]:.4f} / {vall:.4f} | test slot 20 / 21 / 22 / 23 ' + ' / '.join(f'{esh[s]:.4f}' for s in (20, 21, 22, 23))
      + f' | public {ND["test_nat"][epub].mean():.4f} private {ND["test_nat"][~epub].mean():.4f} | pool slot 6 / 7 / 8 / 11 / 12 / 13 ' + ' / '.join(f'{ND["pool"][SL["pool"] == s].mean():.4f}' for s in (6, 7, 8, 11, 12, 13))
      + f' | val twin {ND["val_twin"].mean():.4f} | bar {0.5 * vref:.4f}: {"ok" if c5 else "FAIL"}')
    fv = np.nan_to_num(get('val_nat', col)); fe = np.nan_to_num(get('test_nat', col)); fp = np.nan_to_num(get('pool', col)); fh = np.nan_to_num(get('hold', col))[VIEW['hold']]; ft = np.nan_to_num(get('val_twin', col))
    binf, nbin = spec_binner(fv)
    v = (psi_bins(fp, fv, binf, nbin), psi_bins(fp, fe[epub], binf, nbin), psi_bins(fp, fe[~epub], binf, nbin))
    h = (psi_bins(fh, fv, binf, nbin), psi_bins(fh, fe[epub], binf, nbin), psi_bins(fh, fe[~epub], binf, nbin))
    pv = SL['pool'] <= 8; hv = SL['hold'][VIEW['hold']] <= 8
    half = (psi_bins(fp[pv], fv, binf, nbin), psi_bins(fp[~pv], fe[~epub], binf, nbin), psi_bins(fp[~pv], fe[epub], binf, nbin))
    c4 = max(v) <= 0.25
    P(f'  C4 PSI {col} | pool has = 1 rows vs val / public / private: {v[0]:.4f} / {v[1]:.4f} / {v[2]:.4f} {"ok" if c4 else "FAIL"} | per half (no bar): val-like vs val {half[0]:.4f}, test-like vs private {half[1]:.4f}, test-like vs public {half[2]:.4f}'
      f' | holdout has = 1 rows (comparison): {h[0]:.4f} / {h[1]:.4f} / {h[2]:.4f} | information: val vs private {psi_bins(fv, fe[~epub], binf, nbin):.4f}, twin vs private {psi_bins(ft, fe[~epub], binf, nbin):.4f}, public vs private {psi_bins(fe[epub], fe[~epub], binf, nbin):.4f}'
      f'; sd pool / val / test / twin {fp.std():.5f} / {fv.std():.5f} / {fe.std():.5f} / {ft.std():.5f}; exact-zero share pool / val / public / private {np.mean(fp == 0):.4f} / {np.mean(fv == 0):.4f} / {np.mean(fe[epub] == 0):.4f} / {np.mean(fe[~epub] == 0):.4f}')
    ok = r['C3p']['ok'] and r['C3v']['ok'] and r['C3u']['ok'] and c4 and c5
    RES[fn] = dict(C3=r, C4=dict(ok=bool(c4), pool_vs_val_public_private=v, per_half=half), C5=dict(ok=bool(c5), min_test_day=min(esh.values()), bar=0.5 * vref), g3=bool(ok))
    P(f'  g3 {fn}: C3p {r["C3p"]["ok"]}, C3v {r["C3v"]["ok"]}, C3u {r["C3u"]["ok"]}, C4 {c4}, C5 {c5} -> {"PASS" if ok else "FAIL"}')
# label-free exposure itself, for the record
P('--- information: imp3 > 0 (label-free exposure of the window); equal to the F1 indicator when every source feed has a click')
for q in QS:
    P(f'  {q}: imp3 > 0 {float((np.nan_to_num(get(q, "imp3"))[VIEW[q]] > 0).mean()):.4f}; mean imp3 {float(np.nan_to_num(get(q, "imp3"))[VIEW[q]].mean()):.2f}')
json.dump(RES, open(OUT / 'c02_parity.json', 'x'), indent=1, default=float)
P('NOT RUN (only after a GO): the same gate on all has = 1 train rows (limit J0 of nx3).')
P('DONE')
