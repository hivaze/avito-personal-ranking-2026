"""INVENTORY D, POSITIVE CONTROL of the LR lens (post hoc, calibration only): how large is the LR of inputs that ARE known to be worth something as
trained inputs?  Control score = pool of the 4 all-data popularity-FREE students (abl_fd10_pfkd05w192 s951-954, train-only val predictions; the view
without the 8 popularity inputs), x = one of the removed inputs.  Reference = the same x against the size-matched full-view pool (4 students
w5_fd10_kd05w192 s951-954), which reads the input (expected LR ~ 0).  CPU only, read-only."""
import sys
sys.path.insert(0, '<repo>/reports/INPUTS_1116/raw_fields')
from d03_lib import *
from d03_lib import _z
L = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); L.append(s)
V = load_val()
pool = lambda names: np.mean([_z(V.off, V.M, np.load(PR / f'nn_{n}_trainonly_val.npy').astype(np.float64)) for n in names], 0)
pf = pool([f'abl_fd10_pfkd05w192_s{s}' for s in range(951, 955)]); fu = pool([f'w5_fd10_kd05w192_s{s}' for s in range(951, 955)]); p1 = pool([f'abl_fd10_pop1kd05w192_s{s}' for s in range(951, 955)])
V.score.update(PF=pf, FU=fu, P1=p1); V.ctrl.update(PF=V.controls(pf), FU=V.controls(fu), P1=V.controls(p1))
nd = {k: ndcg_pf(V.score[k], V.y, V.off, V.M) for k in ('PF', 'FU', 'P1', 'TO')}
d = nd['FU'] - nd['PF']; d1 = nd['FU'] - nd['P1']
P(f'val NDCG (4-student train-only pools): full view {nd["FU"].mean():.5f} | popularity-free {nd["PF"].mean():.5f} | pop1 {nd["P1"].mean():.5f} | FD16P20 TO {nd["TO"].mean():.5f}')
P(f'worth of the 8 popularity inputs as TRAINED inputs on val (full - popularity-free, 4 vs 4 students): {d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f} | full - pop1: {d1.mean():+.5f}+-{d1.std() / np.sqrt(len(d1)):.5f}')
FE = R / 'data/feats/val'
xs = {'it_hist_all_luo': np.log1p(np.maximum(np.load(FE / 'it_hist_all_luo.npy'), 0)), 'it_hist_et1_luo': np.log1p(np.maximum(np.load(FE / 'it_hist_et1_luo.npy'), 0)),
      'seq_cv_invd20': np.log1p(np.maximum(np.load(FE / 'seq_cv_invd20.npy'), 0)), 'seq_cv_pop': np.log1p(np.maximum(np.load(FE / 'seq_cv_pop.npy'), 0)),
      'xp_imp_c50': np.load(FE / 'xp_imp_c50.npy')}
for n, x in xs.items():
    for ctrl in ('PF', 'P1', 'FU', 'TO'):
        P(fmt('PC_' + n, 'val', ctrl, lr_test(V, np.nan_to_num(x.astype(np.float64)), ctrl)))
# joint 8-column LR against the popularity-free pool (8 dof)
names8 = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'it_hist_all_luo'] + [f'it_hist_et{e}_luo' for e in range(1, 5)]
X8 = np.column_stack([np.log1p(np.maximum(np.nan_to_num(np.load(FE / f'{n}.npy').astype(np.float64)), 0)) for n in names8])
for ctrl in ('PF', 'FU'):
    C = V.ctrl[ctrl]; th0, ll0 = fit(V.off, V.M, V.y, C); th, ll1 = fit(V.off, V.M, V.y, np.column_stack([C, X8]), th0=np.r_[th0, np.zeros(8)])
    sc = np.column_stack([C, X8]) @ th; dn = ndcg_pf(sc, V.y, V.off, V.M) - ndcg_pf(C @ th0, V.y, V.off, V.M)
    P(f'PC joint 8 popularity inputs, ctrl {ctrl}: LR(8 dof) {2 * (ll1 - ll0):.1f} | IN-SAMPLE linear-residual dNDCG {dn.mean():+.5f}+-{dn.std() / np.sqrt(len(dn)):.5f}')
(OUT / 'd06_poscontrol.txt').write_text('\n'.join(L) + '\n')
