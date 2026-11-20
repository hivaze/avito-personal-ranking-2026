"""CRITIC2 c11: the cold / warm split of the model's OWN content-affinity inputs (c08), characterised (labels opened).
Segment definitions: A = lag-view cold (the explorer's n_item = 0: item in no admitted source sequence of the row's view);
                     B = own-input cold (it_hist_all_luo = 0, the stored input the ranker reads).
For cos_u and seq_max_all: z re-computed among the cold rows / among the warm rows (0 elsewhere), scored under pool + own inputs + nx3;
the 2-df likelihood gain of (cold form, warm form) by a joint fit; and the plain segment flag (within-feed z of the 0 / 1 flag).
usage: c11_segment.py    output out/c11_segment.txt"""
from cc_common import *
import cand_specs as cs
import pmix
P = Log(OUT / 'c11_segment.txt')
V = rs.Views(None)
NRAW = ROOT / 'reports/INPUTS_1116/round2/newcons/out/raw'
for wit in ('H', 'T'):
    sp = 'hold' if wit == 'H' else 'val'; q = 'hold' if wit == 'H' else 'val_twin'; fsp = 'train' if wit == 'H' else 'val'
    D = lens.load_split(sp, 'TO16'); condlib.sp = sp; cn, cl = condlib.control_block(D); off, M = D['off'], D['M']; rows = V.rows(q)[0]
    if wit == 'H':
        nz = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        A = np.load(NRAW / 'hold_lag.npy'); has = np.load(NRAW / 'hold_has_row.npy').astype(bool); SL = [('all', None)]
    else:
        nz = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nr = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
        A = np.load(NRAW / 'val_twin.npy'); has = np.ones(len(A), bool); W = pmix.weights(); SL = [('all', None), ('pmix-private w', W['private'])]
    luo = np.asarray(np.load(ROOT / f'nn/feat_cache/{fsp}/it_hist_all_luo.npy', mmap_mode='r')[rows])
    N = MyNull(D, cl + [nz, nr])
    SEG = {'A lag-view cold (n_item = 0)': has & (A[:, 3] == 0), 'B own-input cold (it_hist_all_luo = 0)': has & (luo == 0)}
    P(f'\n=== witness {wit}: rows with a view {has.mean():.4f}; cold share A {SEG["A lag-view cold (n_item = 0)"][has].mean():.4f}, B {SEG["B own-input cold (it_hist_all_luo = 0)"][has].mean():.4f}; '
      f'B cold among A cold {SEG["B own-input cold (it_hist_all_luo = 0)"][SEG["A lag-view cold (n_item = 0)"]].mean():.4f}, A cold among B cold {SEG["A lag-view cold (n_item = 0)"][SEG["B own-input cold (it_hist_all_luo = 0)"]].mean():.4f} ===')
    for sn, cold in SEG.items():
        warm = has & ~cold
        fl = form(np.where(has, cold.astype(np.float64), np.nan), off, M, None, 'z')
        P(f'  segment {sn}: flag z-form cz ' + ' / '.join(f'{czw(D, N.score_f(fl)[0], w):+.2f}' for _, w in SL) + '  (' + ' / '.join(n_ for n_, _ in SL) + ')')
        for n_ in ('cos_u', 'seq_max_all'):
            x = cs.load(condlib.STORE[n_][sp], D)
            zc = form(np.where(cold, x, np.nan), off, M, None, 'z'); zw = form(np.where(warm, x, np.nan), off, M, None, 'z')
            Uc = N.score_f(zc)[0]; Uw = N.score_f(zw)[0]
            Nj = MyNull(D, cl + [nz, nr, zc, zw], theta0=list(N.theta) + [0.0, 0.0])
            P(f'     {n_:12s} cold rows ' + ' / '.join(f'{czw(D, Uc, w):+.2f}' for _, w in SL) + ' | warm rows ' + ' / '.join(f'{czw(D, Uw, w):+.2f}' for _, w in SL)
              + f' | joint 2-df fit: LR {2 * (Nj.L - N.L):.1f}, nats/feed {(Nj.L - N.L) / D["n"]:.5f}, coefficients cold {Nj.theta[-2]:+.4f} warm {Nj.theta[-1]:+.4f}')
            del Nj
    del N, D, A
P('DONE')
