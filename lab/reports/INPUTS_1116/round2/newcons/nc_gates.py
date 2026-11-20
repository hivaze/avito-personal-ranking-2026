"""newcons step 3: parity gates C3 / C4 / C5 and the calibrated score test (cz) of the six constructions, in the form of nn_x/w12/nx3/nx3_gates.py.
Everything here was fixed in PRELIST.txt (2026-11-16T04:56:50Z) before any array existed; this file was written while the build was running, before any cz was read.

Train side of the parity gates = the HOLDOUT feeds in the lag view (no full-train array is built; the critic's population in nx3_gates.py).
  C3  non-zero share by the 7 listing-age buckets: holdout slots 11-13 vs PRIVATE and vs PUBLIC rows (test feeds M >= 50); holdout slots 6-8 vs val rows (M >= 50):
      dZ <= 0.05 and largest bucket (view row share >= 0.05) <= 0.10
  C4  PSI <= 0.25 for each entered form (z, rk): holdout has = 1 rows vs val / public / private; bins = 10 val deciles of the non-zero file values + exact-zero bin
  C5  non-zero share on every test day >= 0.5 x val's (largest of val overall / slot 16 / slot 18)
  cz  condlib.Null score test, null model = pool + control block of the ranker's own inputs + nx3 (nx_rp3_z, nx_rp3_rk):
        witness H: holdout rows, lag view, pool = 16 held-out wide students, nx3 = the files of record (NaN -> 0)
        witness V: val rows, STALE-VAL TWIN of the column, nx3 = nn_x/w12/nx3/staleval files, base TO16 (base FILE printed)
      printed for information: the same without nx3 in the null model; natural val (natural nx3); cold / warm rows (form re-computed among those rows only).
  Placebo: 5 within-feed permutations of each construction's z column per witness (seeds 7000 + 10 k + s, 8000 + 10 k + s).
  PASS = C3, C4, C5 ok AND the same form has the same sign and |cz| >= 2.5 on H and on V (TO16) AND max |placebo cz| < 3.0 on each witness.
CPU only, 12 threads, read-only. Output reports/INPUTS_1116/round2/newcons/out/gates.txt (exclusive create)."""
from nc_common import *
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows')
_argv = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
import lens
import condlib
sys.argv = _argv
import json
P = Log(NCO / f'gates{os.environ.get("NC_TAG", "")}.txt')
nb.set_num_threads(NT)
P(f'lens md5 {md5("<repo>/reports/INPUTS_1116/residual_rows/lens.py")}, condlib md5 {md5("<repo>/reports/INPUTS_1116/residual_rows/condlib.py")}')
CONS = [('N1 cw_item', 4, True), ('N2 cw_cell', 5, True), ('N3 aud_cell', 6, False), ('N4 fwd20', 0, True), ('N5 rp3_l8', 9, True), ('N6 aud_item', 2, False)]
F = {s: pl.read_parquet(FLAT / s / 'feeds.parquet') for s in SPLITS}
A = {s: dict(M=F[s]['M'].to_numpy().astype(np.int64), ts=F[s]['timestamp'].to_numpy().astype(np.int64)) for s in SPLITS}
for s in SPLITS: A[s]['slot'] = (c9.day_of(A[s]['ts']) - hc.DAY0).astype(np.int64)
Mt, Mv, Me = A['train']['M'], A['val']['M'], A['test']['M']; tslot = A['train']['slot']


def offs(M):
    return np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)


hold, _p = hc.holdout_feeds(F['train']); Mh = Mt[hold]; offh = offs(Mh); offv = offs(Mv); offe = offs(Me)
co = F['train']['cand_off'].to_numpy().astype(np.int64)[hold]; rows_h = np.repeat(co - offh, Mh) + np.arange(int(Mh.sum()))
HOLD = np.load(NRAW / 'hold_lag.npy'); VN = np.load(NRAW / 'val_nat.npy'); VT = np.load(NRAW / 'val_twin.npy'); EN = np.load(NRAW / 'test_nat.npy')
hasrow = np.load(NRAW / 'hold_has_row.npy'); slot_h = np.repeat(tslot[hold], Mh); assert (hasrow == (slot_h >= 6)).all()
vslot = np.repeat(A['val']['slot'], Mv); eslot = np.repeat(A['test']['slot'], Me)
epub = np.repeat(A['test']['ts'] < hc.PUB_TS, Me); ebig = np.repeat(Me >= 50, Me); vbig = np.repeat(Mv >= 50, Mv)
kn = hc.knots()
bv = hc.age_bucket(hc.age_hours(np.load(FLAT / 'val/cand_f4.npy'), kn)); be = hc.age_bucket(hc.age_hours(np.load(FLAT / 'test/cand_f4.npy'), kn))
bh = hc.age_bucket(hc.age_hours(np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')[rows_h]), kn))
KK = {}
for nm, j, lg_ in CONS:
    KK[nm] = float(1.0 / np.median(VN[:, j][VN[:, j] > 0])) if lg_ else 1.0


def forms_of(x, off, M, k, lg_):
    x = np.asarray(x, np.float64)
    return lens.feed_z(np.log1p(k * x) if lg_ else x, off, M, 5.0), lens.feed_rk(x, off, M)


FORM = {}        # (construction, view) -> (z, rk)
for nm, j, lg_ in CONS:
    FORM[nm, 'hold'] = forms_of(HOLD[:, j], offh, Mh, KK[nm], lg_); FORM[nm, 'val'] = forms_of(VN[:, j], offv, Mv, KK[nm], lg_)
    FORM[nm, 'twin'] = forms_of(VT[:, j], offv, Mv, KK[nm], lg_); FORM[nm, 'test'] = forms_of(EN[:, j], offe, Me, KK[nm], lg_)
# ------------------------------------------------------------------ PART 0: kernel sanity, label-free
P('=== PART 0. label-free sanity ===')
cvv = np.load(ROOT / 'nn/feat_cache/val/seq_cv_invd20.npy')
s20 = VN[:, 0] + VN[:, 1]
P(f'fwd20 + bwd20 against the stored input seq_cv_invd20 on val rows: corr raw {np.corrcoef(s20, cvv)[0, 1]:.4f}, corr log1p {np.corrcoef(np.log1p(s20), np.log1p(cvv))[0, 1]:.4f}; non-zero share mine {np.mean(s20 > 0):.4f}, stored {np.mean(cvv > 0):.4f}, mine only {np.mean((s20 > 0) & (cvv == 0)):.4f}, stored only {np.mean((s20 == 0) & (cvv > 0)):.4f}')
m_ = s20 > 0
P(f'forward share fwd / (fwd + bwd) on rows with a windowed hit: val natural mean {np.mean(VN[m_, 0] / s20[m_]):.4f}; share of those rows with fwd = 0: {np.mean(VN[m_, 0] == 0):.4f}, with bwd = 0: {np.mean(VN[m_, 1] == 0):.4f}')
P('K (1 / median non-zero value on natural val rows): ' + ', '.join(f'{nm} {KK[nm]:.4e}' for nm, j, lg_ in CONS if lg_))
P('cold rows (no admitted item-level posting in the row\'s own view): hold lag %.4f | val natural %.4f | val twin %.4f | public %.4f | private %.4f' % (
    np.mean(HOLD[hasrow, 3] == 0), np.mean(VN[:, 3] == 0), np.mean(VT[:, 3] == 0), np.mean(EN[epub, 3] == 0), np.mean(EN[~epub, 3] == 0)))
P('distinct admitted sequences per row, median (item postings | cell postings): hold lag %.0f | %.0f ; val natural %.0f | %.0f ; val twin %.0f | %.0f ; test %.0f | %.0f' % (
    np.median(HOLD[hasrow, 3]), np.median(HOLD[hasrow, 7]), np.median(VN[:, 3]), np.median(VN[:, 7]), np.median(VT[:, 3]), np.median(VT[:, 7]), np.median(EN[:, 3]), np.median(EN[:, 7])))
P('rows at the 500-posting cap of the CELL scan (n_cell >= 250 distinct sequences as a proxy): hold lag %.4f | val natural %.4f | val twin %.4f | test %.4f' % (
    np.mean(HOLD[hasrow, 7] >= 250), np.mean(VN[:, 7] >= 250), np.mean(VT[:, 7] >= 250), np.mean(EN[:, 7] >= 250)))
# ------------------------------------------------------------------ PART 1: parity
P('\n=== PART 1. parity (label-free); train side = holdout feeds, lag view ===')
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
    nzv = val[val != 0]
    ed = np.unique(np.quantile(nzv, np.linspace(0.1, 0.9, 9))) if len(nzv) else np.zeros(0)
    return (lambda x: np.where(x == 0, 0, 1 + np.searchsorted(ed, x, side='right'))), len(ed) + 2


PAR = {}; COV = {}
tl = hasrow & (slot_h >= 11) & (slot_h <= 13); vl = hasrow & (slot_h >= 6) & (slot_h <= 8)
for nm, j, lg_ in CONS:
    hz = np.nan_to_num(HOLD[:, j]) != 0; vz = VN[:, j] != 0; tz = VT[:, j] != 0; ez = EN[:, j] != 0
    COV[nm] = dict(hold=float(hz[hasrow].mean()), hold_tl=float(hz[tl].mean()), hold_vl=float(hz[vl].mean()), val=float(vz.mean()), twin=float(tz.mean()), pub=float(ez[epub].mean()), priv=float(ez[~epub].mean()))
    P(f'--- {nm}: non-zero share | holdout lag view all {COV[nm]["hold"]:.4f} (slots 6-8 {COV[nm]["hold_vl"]:.4f}, slots 11-13 {COV[nm]["hold_tl"]:.4f}) | val natural {COV[nm]["val"]:.4f} | val twin {COV[nm]["twin"]:.4f} | public {COV[nm]["pub"]:.4f} | private {COV[nm]["priv"]:.4f}')
    c3ok = True; c3s = []
    for regn, vm, refn, ref_nz, ref_b in (('test-like (slots 11-13)', tl, 'PRIVATE (M >= 50)', ez[~epub & ebig], be[~epub & ebig]), ('test-like (slots 11-13)', tl, 'PUBLIC (M >= 50)', ez[epub & ebig], be[epub & ebig]),
                                          ('val-like (slots 6-8)', vl, 'VAL (M >= 50)', vz[vbig], bv[vbig])):
        dZ, big, bn, zv, zr, w = gap(hz[vm], bh[vm], ref_nz, ref_b); ok = dZ <= 0.05 and big <= 0.10; c3ok &= ok; c3s.append(f'{dZ:.3f}/{big:.3f}')
        P(f'  C3 {regn} vs {refn}: share {hz[vm].mean():.4f} vs {ref_nz.mean():.4f}; dZ {dZ:.4f}, largest bucket {big:.4f} at {bn} -> {"ok" if ok else "FAIL"} | view ' + ' '.join(f'{x:.3f}' for x in zv) + ' | ref ' + ' '.join(f'{x:.3f}' for x in zr))
    c4ok = True; c4s = []
    for fi, fn in ((0, 'z'), (1, 'rk')):
        a_v = FORM[nm, 'val'][fi]; a_e = FORM[nm, 'test'][fi]; a_h = FORM[nm, 'hold'][fi][hasrow]; binf, nbin = spec_binner(a_v)
        v = (psi_bins(a_h, a_v, binf, nbin), psi_bins(a_h, a_e[epub], binf, nbin), psi_bins(a_h, a_e[~epub], binf, nbin)); ok = max(v) <= 0.25; c4ok &= ok; c4s.append('/'.join(f'{x:.3f}' for x in v))
        P(f'  C4 PSI {fn} form, holdout has = 1 rows vs val / public / private: {v[0]:.4f} / {v[1]:.4f} / {v[2]:.4f} -> {"ok" if ok else "FAIL"} | information: slots 11-13 vs private {psi_bins(FORM[nm, "hold"][fi][tl], a_e[~epub], binf, nbin):.4f}; slots 6-8 vs val {psi_bins(FORM[nm, "hold"][fi][vl], a_v, binf, nbin):.4f}; val twin vs private {psi_bins(FORM[nm, "twin"][fi], a_e[~epub], binf, nbin):.4f}; val vs private {psi_bins(a_v, a_e[~epub], binf, nbin):.4f}')
    vshare = max(float(vz.mean()), float(vz[vslot == 16].mean()), float(vz[vslot == 18].mean())); es = {s: float(ez[eslot == s].mean()) for s in (20, 21, 22, 23)}; c5ok = all(x >= 0.5 * vshare for x in es.values())
    P(f'  C5 non-zero share by test day: ' + ' / '.join(f'{es[s]:.4f}' for s in (20, 21, 22, 23)) + f' against 0.5 x {vshare:.4f} -> {"ok" if c5ok else "FAIL"}')
    PAR[nm] = dict(c3=c3ok, c4=c4ok, c5=c5ok, c3s=c3s, c4s=c4s)
    P(f'  raw-level information: mean of the raw value on non-zero rows | holdout slots 11-13 {np.nanmean(np.where(hz & tl, HOLD[:, j], np.nan)):.4e} | holdout slots 6-8 {np.nanmean(np.where(hz & vl, HOLD[:, j], np.nan)):.4e} | val {VN[vz, j].mean():.4e} | val twin {VT[tz, j].mean():.4e} | private {EN[ez & ~epub, j].mean():.4e}')
# ------------------------------------------------------------------ PART 2: cz (labels are opened from here on)
P('\n=== PART 2. calibrated score test; null model = pool + own inputs + nx3; bar |cz| >= 2.5 with the same sign on both witnesses ===')
RES = {nm: {} for nm, _, _ in CONS}; PLA = {}


def within(x, mask, off, M, k, lg_):
    y = np.array(x, np.float64); y[~mask] = np.nan
    return forms_of(y, off, M, k, lg_)


def read_block(tag, D, Nx, N0, view, rawarr, off, M, coldmask, valid, seed0):
    """Nx = null model with nx3, N0 = without. Stores cz of both forms, cold / warm reads, placebo."""
    pl_ = []
    for ci, (nm, j, lg_) in enumerate(CONS):
        z, rk = FORM[nm, view]; a = Nx.score(z); b = Nx.score(rk); a0 = N0.score(z); b0 = N0.score(rk)
        zc, rc = within(rawarr[:, j], coldmask & valid, off, M, KK[nm], lg_); zw, rw = within(rawarr[:, j], ~coldmask & valid, off, M, KK[nm], lg_)
        ac, bc, aw, bw = Nx.score(zc), Nx.score(rc), Nx.score(zw), Nx.score(rw)
        RES[nm][tag] = dict(z=a['cz'], rk=b['cz'], z0=a0['cz'], rk0=b0['cz'], zc=ac['cz'], rkc=bc['cz'], zw=aw['cz'], rkw=bw['cz'], b1=a['b1'], r2=a['r2'], dnats=a['dnats'])
        pp = [Nx.score(condlib.perm_within(z.astype(np.float32), D['off'], D['M'], seed0 + 10 * ci + s))['cz'] for s in range(5)]; pl_ += pp
        P(f'  [{tag}] {nm}: GIVEN nx3  z {a["cz"]:+.2f} / rk {b["cz"]:+.2f}  (b1 {a["b1"]:+.4f}, dnats {a["dnats"]:.5f}, R2 with the null model {a["r2"]:.2f}) | without nx3 in the null: {a0["cz"]:+.2f} / {b0["cz"]:+.2f}'
          f' | COLD rows only: {ac["cz"]:+.2f} / {bc["cz"]:+.2f} | warm rows only: {aw["cz"]:+.2f} / {bw["cz"]:+.2f} | placebo (5 perms of z): ' + ' '.join(f'{x:+.2f}' for x in pp))
    pl_ = np.array(pl_); PLA[tag] = pl_
    P(f'  [{tag}] placebo block: 30 within-feed permutations, mean {pl_.mean():+.2f}, sd {pl_.std():.2f}, max |cz| {np.abs(pl_).max():.2f}')


# ---- witness H
D = lens.load_split('hold', 'TO16'); condlib.sp = 'hold'; nm_, cl = condlib.control_block(D); assert (D['rows'] == rows_h).all()
zx = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows_h]).astype(np.float32), nan=0.0)
rx = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows_h]).astype(np.float32), nan=0.0)
N0 = condlib.Null(D, nm_, cl)
s1, s2 = N0.score(zx), N0.score(rx)
P(f'[H holdout lag view; pool = 16 held-out wide students; control block {len(nm_)} columns] pipeline check: nx3 files under the null WITHOUT nx3: cz {s1["cz"]:+.2f} / {s2["cz"]:+.2f} (record nn_x/w12/nx3/out/nx3_gates.txt: +4.98 / +4.32)')
Nx = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk'], cl + [zx, rx])
P(f'  null model with nx3: {Nx.X.shape[1]} columns; loglik gain of the nx3 pair per feed {(Nx.r["L"] - N0.r["L"]) / D["n"]:.5f} nats')
read_block('H', D, Nx, N0, 'hold', HOLD, offh, Mh, np.nan_to_num(HOLD[:, 3], nan=1.0) == 0, hasrow, 7000)
for rn, rm in (('slots 6-8', (slot_h >= 6) & (slot_h <= 8)), ('slots 11-13', slot_h >= 11)):
    P(f'  [H by period, {rn}] ' + ' | '.join(f'{nm}: {Nx.score(np.where(rm, FORM[nm, "hold"][0], 0).astype(np.float32))["cz"]:+.2f} / {Nx.score(np.where(rm, FORM[nm, "hold"][1], 0).astype(np.float32))["cz"]:+.2f}' for nm, _, _ in CONS))
del N0, Nx, D
# ---- witness V
ZT = np.load(NX3 / 'staleval/nx_rp3_z.npy'); RT = np.load(NX3 / 'staleval/nx_rp3_rk.npy'); ZN = np.load(NX3 / 'out/files/val/nx_rp3_z.npy'); RN = np.load(NX3 / 'out/files/val/nx_rp3_rk.npy')
allv = np.ones(NROWS['val'], bool)
for base in ('TO16', 'FILE'):
    D = lens.load_split('val', base); condlib.sp = 'val'; nm_, cl = condlib.control_block(D)
    N0 = condlib.Null(D, nm_, cl); s1, s2 = N0.score(ZT), N0.score(RT)
    P(f'[V val rows, base {base}; control block {len(nm_)} columns] pipeline check: nx3 TWIN files under the null WITHOUT nx3: cz {s1["cz"]:+.2f} / {s2["cz"]:+.2f} (record: TO16 +6.46 / +4.78, FILE +6.51 / +4.79)')
    Nx = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk'], cl + [ZT, RT])
    read_block('V' if base == 'TO16' else 'V_FILE', D, Nx, N0, 'twin', VT, offv, Mv, VT[:, 3] == 0, allv, 8000 if base == 'TO16' else 9000)
    if base == 'TO16':
        for vs in (16, 18):
            mk = vslot == vs
            P(f'  [V by val day, slot {vs}] ' + ' | '.join(f'{nm}: {Nx.score(np.where(mk, FORM[nm, "twin"][0], 0).astype(np.float32))["cz"]:+.2f} / {Nx.score(np.where(mk, FORM[nm, "twin"][1], 0).astype(np.float32))["cz"]:+.2f}' for nm, _, _ in CONS))
        Nn = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk'], cl + [ZN, RN])
        read_block('V_NAT', D, Nn, N0, 'val', VN, offv, Mv, VN[:, 3] == 0, allv, 9500)
        del Nn
    del N0, Nx, D
# ------------------------------------------------------------------ verdict
P('\n=== TABLE: construction | coverage (non-zero share: holdout lag / val nat / val twin / public / private) | parity C3 (dZ/bucket: private, public, val) ; C4 PSI z ; C4 PSI rk ; C5 | cz H (z / rk) | cz V twin TO16 (z / rk) [FILE] | cz cold rows H ; V | verdict ===')
plok = {t: bool(np.abs(PLA[t]).max() < 3.0) for t in ('H', 'V')}
OUTJ = {}
for nm, j, lg_ in CONS:
    c = COV[nm]; p = PAR[nm]; h = RES[nm]['H']; v = RES[nm]['V']; vf = RES[nm]['V_FILE']
    par_ok = p['c3'] and p['c4'] and p['c5']
    info = any(abs(h[k]) >= 2.5 and abs(v[k]) >= 2.5 and np.sign(h[k]) == np.sign(v[k]) for k in ('z', 'rk'))
    verdict = 'PASS' if (par_ok and info and plok['H'] and plok['V']) else ('no information beyond nx3' if not info else ('information, parity FAIL' if not par_ok else 'placebo out of calibration'))
    P(f'{nm} | {c["hold"]:.3f} / {c["val"]:.3f} / {c["twin"]:.3f} / {c["pub"]:.3f} / {c["priv"]:.3f} | C3 {", ".join(p["c3s"])} {"ok" if p["c3"] else "FAIL"}; C4 z {p["c4s"][0]}; rk {p["c4s"][1]} {"ok" if p["c4"] else "FAIL"}; C5 {"ok" if p["c5"] else "FAIL"}'
      f' | {h["z"]:+.2f} / {h["rk"]:+.2f} | {v["z"]:+.2f} / {v["rk"]:+.2f} [{vf["z"]:+.2f} / {vf["rk"]:+.2f}] | {h["zc"]:+.2f} / {h["rkc"]:+.2f} ; {v["zc"]:+.2f} / {v["rkc"]:+.2f} | {verdict}')
    OUTJ[nm] = dict(coverage=c, parity=dict(c3=p['c3'], c4=p['c4'], c5=p['c5'], c3s=p['c3s'], c4s=p['c4s']), H=h, V=v, V_FILE=vf, V_NAT=RES[nm]['V_NAT'], verdict=verdict)
P(f'placebo calibration: H sd {PLA["H"].std():.2f} max {np.abs(PLA["H"]).max():.2f}; V sd {PLA["V"].std():.2f} max {np.abs(PLA["V"]).max():.2f}; V_FILE sd {PLA["V_FILE"].std():.2f}; V_NAT sd {PLA["V_NAT"].std():.2f}')
with open(NCO / f'gates{os.environ.get("NC_TAG", "")}.json', 'x') as f:
    json.dump(OUTJ, f, indent=1, default=float)
P('DONE')
