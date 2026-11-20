"""CRITIC c05: one-step coefficient b1 (score / information, per unit of the entered column) and R2 with pool + own inputs of the ENTERED forms
(within-feed z of log1p(K x), within-feed rank) on lag-view train rows (holdout) against val rows (natural build = what the model will see on val / test,
and the stale-val twin). A trained weight is learned on the first and applied to the second: equal b1 = relation parity. CPU only, read-only."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
CR = '<repo>/reports/INPUTS_1116/critic'
os.environ['NUMBA_CACHE_DIR'] = CR + '/out/numba_cache'
sys.dont_write_bytecode = True
import numba  # noqa
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows')
sys.argv = [sys.argv[0], 'val', 'TO16']
from lens import *
import condlib
out = open(CR + '/out/c05_coef.txt', 'w')
def Pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.write(s + '\n'); out.flush()
W = np.load(CR + '/out/c01_walk.npz'); B = np.load(CR + '/out/c02_bt.npz')
Kr = 1.0 / np.median(W['val_nat'][:, 0][W['val_nat'][:, 0] > 0]); Kp = 1.0 / np.median(W['val_nat'][:, 1][W['val_nat'][:, 1] > 0])
slot_h = W['hold_slot'].astype(np.int64)
def ent(x, D, keep=None):
    """entered forms: zeros are values; feeds outside `keep` are set constant (contribute nothing)"""
    y = np.array(x, np.float64)
    if keep is not None: y[~keep] = np.nan
    return forms(y, D, log=False)
Pr('witness | column | z form: cz, b1, R2 | rank form: cz, b1')
D = load_split('hold', 'TO16'); condlib.sp = 'hold'; nm_, cl = condlib.control_block(D); N = condlib.Null(D, nm_, cl)
for rn, rm in (('holdout lag view, slots 6-8', (slot_h >= 6) & (slot_h <= 8)), ('holdout lag view, slots 11-13', slot_h >= 11), ('holdout lag view, all 15,225', slot_h >= 6)):
    for cn, x in (('nx_rp3', np.log1p(Kr * np.nan_to_num(W['hold_lag'][:, 0]))), ('nx_p3a', np.log1p(Kp * np.nan_to_num(W['hold_lag'][:, 1]))), ('nx_bt', np.nan_to_num(B['hold_lag_theta'])), ('nx_bt_seen', np.nan_to_num(B['hold_lag_seen']))):
        F = ent(x, D, rm); a = N.score(F['z']); b = N.score(F['rk'])
        Pr(f'{rn} | {cn} | {a["cz"]:+.2f}, {a["b1"]:+.4f}, {a["r2"]:.2f} | {b["cz"]:+.2f}, {b["b1"]:+.4f}')
del N, D
D = load_split('val', 'TO16'); condlib.sp = 'val'; nm_, cl = condlib.control_block(D); N = condlib.Null(D, nm_, cl)
for rn, key, bk in (('val natural (what the model reads on val)', 'val_nat', 'val_nat'), ('val stale-val twin', 'val_twin', 'val_twin')):
    for cn, x in (('nx_rp3', np.log1p(Kr * W[key][:, 0])), ('nx_p3a', np.log1p(Kp * W[key][:, 1])), ('nx_bt', B[bk + '_theta']), ('nx_bt_seen', B[bk + '_seen'])):
        F = ent(x, D); a = N.score(F['z']); b = N.score(F['rk'])
        Pr(f'{rn} | {cn} | {a["cz"]:+.2f}, {a["b1"]:+.4f}, {a["r2"]:.2f} | {b["cz"]:+.2f}, {b["b1"]:+.4f}')
# agreement of the twin with the natural build on val rows
for cn, a_, b_ in (('rp3b05', W['val_nat'][:, 0], W['val_twin'][:, 0]),):
    Fa = ent(np.log1p(Kr * a_), D); Fb = ent(np.log1p(Kr * b_), D)
    Pr(f'val rows: correlation natural vs twin, entered z form {np.corrcoef(Fa["z"], Fb["z"])[0, 1]:.3f}, rank form {np.corrcoef(Fa["rk"], Fb["rk"])[0, 1]:.3f}; rows non-zero in natural {np.mean(a_ > 0):.4f}, in twin {np.mean(b_ > 0):.4f}, in natural only {np.mean((a_ > 0) & (b_ == 0)):.4f}, in twin only {np.mean((a_ == 0) & (b_ > 0)):.4f}')
out.close()
