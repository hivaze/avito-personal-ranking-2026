"""CRITIC2 c05: re-read of the six "new constructions" from the explorer's raw arrays with my own score path (labels opened).
Null model = pool + own inputs + nx3 (lag view on holdout rows; stale-val twin on val rows, as the explorer). Whole-feed z / rank forms and the
cold-row forms of N1-N3 (cold = no admitted item-level posting, column n_item = 0; form re-computed among the cold rows, 0 elsewhere).
Also: the cold-row forms given the rescue trio (kn1_5:z, ns2_5:z, bt5:rk), to see whether the two rounds found the same thing.
usage: c05_newcons.py    output out/c05_newcons.txt"""
from cc_common import *
P = Log(OUT / 'c05_newcons.txt')
V = rs.Views(None)
NRAW = ROOT / 'reports/INPUTS_1116/round2/newcons/out/raw'
CONS = [('N1 cw_item', 4, True), ('N2 cw_cell', 5, True), ('N3 aud_cell', 6, False), ('N4 fwd20', 0, True), ('N5 rp3_l8', 9, True), ('N6 aud_item', 2, False)]
VN = np.load(NRAW / 'val_nat.npy')
KK = {nm: (float(1.0 / np.median(VN[:, j][VN[:, j] > 0])) if lg else 1.0) for nm, j, lg in CONS}
for wit in ('H', 'T'):
    sp = 'hold' if wit == 'H' else 'val'; q = 'hold' if wit == 'H' else 'val_twin'
    D = lens.load_split(sp, 'TO16'); condlib.sp = sp; cn, cl = condlib.control_block(D); off, M = D['off'], D['M']; rows = V.rows(q)[0]
    if wit == 'H':
        nz = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        A = np.load(NRAW / 'hold_lag.npy'); has = np.load(NRAW / 'hold_has_row.npy').astype(bool)
    else:
        nz = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nr = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
        A = np.load(NRAW / 'val_twin.npy'); has = np.ones(len(A), bool)
    assert len(A) == len(rows)
    N = MyNull(D, cl + [nz, nr])
    trio = [form(rget(q, 'kn1_5'), off, M, None, 'z'), form(rget(q, 'ns2_5'), off, M, 'log1p', 'z'), form(rget(q, 'bt5'), off, M, None, 'rk')]
    N3 = MyNull(D, cl + [nz, nr] + trio, theta0=list(N.theta) + [0.0] * 3)
    cold = has & (A[:, 3] == 0)
    P(f'\n=== witness {wit}: rows {len(rows)}, with a view {has.mean():.4f}, cold among them {cold[has].mean():.4f} ===')
    P('construction | whole feed z / rank | cold rows z / rank | cold rows given the rescue trio z / rank')
    for nm, j, lg in CONS:
        x = A[:, j].astype(np.float64); xz = np.where(has, np.log1p(KK[nm] * x) if lg else x, np.nan); xr = np.where(has, x, np.nan)
        z = form(xz, off, M, None, 'z'); rk = form(xr, off, M, None, 'rk')
        s_ = f'  {nm:12s} | {N.cz(z)[0]:+.2f} / {N.cz(rk)[0]:+.2f}'
        if nm[:2] in ('N1', 'N2', 'N3'):
            zc = form(np.where(cold, xz, np.nan), off, M, None, 'z'); rc = form(np.where(cold, xr, np.nan), off, M, None, 'rk')
            s_ += f' | {N.cz(zc)[0]:+.2f} / {N.cz(rc)[0]:+.2f} | {N3.cz(zc)[0]:+.2f} / {N3.cz(rc)[0]:+.2f}'
        P(s_)
    del N, N3, D, A
P('DONE')
