"""CRITIC c07: is nx_p3a_z a third piece next to nx_rp3_z / nx_rp3_rk?  cz of the p3a forms when the two rp3b05 forms are in the null model
(pool + own inputs + nx_rp3_z + nx_rp3_rk), on lag-view holdout rows and on val (natural and twin). CPU only, read-only."""
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
out = open(CR + '/out/c07_dup.txt', 'w')
def Pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.write(s + '\n'); out.flush()
W = np.load(CR + '/out/c01_walk.npz')
Kr = 1.0 / np.median(W['val_nat'][:, 0][W['val_nat'][:, 0] > 0]); Kp = 1.0 / np.median(W['val_nat'][:, 1][W['val_nat'][:, 1] > 0])
slot_h = W['hold_slot'].astype(np.int64)
def run(sp, lab, r, p, keep=None):
    D = load_split(sp, 'TO16'); condlib.sp = sp; nm_, cl = condlib.control_block(D)
    def ent(x):
        y = np.array(x, np.float64)
        if keep is not None: y[~keep] = np.nan
        return forms(y, D, log=False)
    Fr = ent(np.log1p(Kr * np.nan_to_num(r))); Fp = ent(np.log1p(Kp * np.nan_to_num(p)))
    N0 = condlib.Null(D, nm_, cl); a0 = N0.score(Fp['z']); r0 = N0.score(Fr['z'])
    N1 = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk'], cl + [Fr['z'], Fr['rk']]); a1 = N1.score(Fp['z'])
    N2 = condlib.Null(D, nm_ + ['nx_rp3_z'], cl + [Fr['z']]); a2 = N2.score(Fr['rk'])
    Pr(f'{lab}: nx_rp3_z given own inputs {r0["cz"]:+.2f}; nx_p3a_z given own inputs {a0["cz"]:+.2f}; nx_p3a_z given own inputs + nx_rp3_z + nx_rp3_rk {a1["cz"]:+.2f} (R2 {a1["r2"]:.3f}); nx_rp3_rk given own inputs + nx_rp3_z {a2["cz"]:+.2f} (R2 {a2["r2"]:.3f}); corr(nx_rp3_z, nx_p3a_z) {np.corrcoef(Fr["z"], Fp["z"])[0, 1]:.3f}')
run('hold', 'holdout lag view (15,225 feeds)', W['hold_lag'][:, 0], W['hold_lag'][:, 1], slot_h >= 6)
run('val', 'val natural', W['val_nat'][:, 0], W['val_nat'][:, 1])
run('val', 'val stale-val twin', W['val_twin'][:, 0], W['val_twin'][:, 1])
out.close()
