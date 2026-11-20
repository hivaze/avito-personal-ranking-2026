"""CRITIC2 c08: is the cold / warm split read of the new-construction columns (cold rows +, warm rows -) a new input, or the model's own content affinity
read by segment?  (labels opened)
For the explorer's N2 cw_cell, N3 aud_cell and for inputs the ranker ALREADY reads (cos_u, dot_u, maxsim32, seq_max_all, attn32): the within-feed z of the
column re-computed among the cold rows only (0 elsewhere) and among the warm rows only, scored under the null model pool + own inputs + nx3.
cold = no admitted item-level posting in the row's view (column n_item of the explorer's raw arrays = 0).
If an existing input shows the same split, the split is a segment-wise re-weighting of what the model reads, not a missing input.
Also: the cold forms at the pmix weights (the private mix has more cold rows).
usage: c08_coldsplit.py    output out/c08_coldsplit.txt"""
from cc_common import *
import cand_specs as cs
import pmix
P = Log(OUT / 'c08_coldsplit.txt')
V = rs.Views(None)
NRAW = ROOT / 'reports/INPUTS_1116/round2/newcons/out/raw'
OWN = ['cos_u', 'dot_u', 'maxsim32', 'seq_max_all', 'attn32']
for wit in ('H', 'T'):
    sp = 'hold' if wit == 'H' else 'val'; q = 'hold' if wit == 'H' else 'val_twin'
    D = lens.load_split(sp, 'TO16'); condlib.sp = sp; cn, cl = condlib.control_block(D); off, M = D['off'], D['M']; rows = V.rows(q)[0]
    if wit == 'H':
        nz = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        A = np.load(NRAW / 'hold_lag.npy'); has = np.load(NRAW / 'hold_has_row.npy').astype(bool); SL = [('all', None)]
    else:
        nz = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nr = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
        A = np.load(NRAW / 'val_twin.npy'); has = np.ones(len(A), bool); W = pmix.weights(); SL = [('all', None), ('pmix-public w', W['public']), ('pmix-private w', W['private'])]
    N = MyNull(D, cl + [nz, nr]); cold = has & (A[:, 3] == 0); warm = has & (A[:, 3] > 0)
    cols = {'N3 aud_cell': A[:, 6].astype(np.float64), 'N2 cw_cell': np.log1p(A[:, 5].astype(np.float64) / np.median(A[:, 5][A[:, 5] > 0]))}
    for n_ in OWN:
        cols['own input ' + n_] = cs.load(condlib.STORE[n_][sp], D)
    P(f'\n=== witness {wit}: rows {len(rows)}; cold share among rows with a view {cold[has].mean():.4f} ===')
    P('column | cold rows only: ' + ' / '.join(n_ for n_, _ in SL) + ' | warm rows only: ' + ' / '.join(n_ for n_, _ in SL) + ' | whole feed (rows with a view)')
    for k, x in cols.items():
        zc = form(np.where(cold, x, np.nan), off, M, None, 'z'); zw = form(np.where(warm, x, np.nan), off, M, None, 'z'); za = form(np.where(has, x, np.nan), off, M, None, 'z')
        Uc = N.score_f(zc)[0]; Uw = N.score_f(zw)[0]
        P(f'  {k:22s} | ' + ' / '.join(f'{czw(D, Uc, w):+.2f}' for _, w in SL) + ' | ' + ' / '.join(f'{czw(D, Uw, w):+.2f}' for _, w in SL) + f' | {N.cz(za)[0]:+.2f}')
    del N, D, A
P('DONE')
