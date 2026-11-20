"""ftarm_pathcheck.py (2026-11-16, reports/NEXT_1116/fillers, lane B): write-new copy of reports/TRICKS_1115/followup/mixft_pathcheck.py. CPU, read-only.
PATH CHECK OF THE DAY for the paired fine-tune arms: are the stored controls of Q4 MIXFT (suffix mixctl3, run 2026-11-15T23:13Z .. 2026-11-16T02:11Z) still what the
fine-tune path produces today? One control is re-run today on member 1 and compared with the two stored control realisations of the same member.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python reports/NEXT_1116/fillers/ftarm_pathcheck.py [suffix of today's control = ftctl3c]

Member 1 = w5_fd10_kd05w192_s951, all runs 3 seeds, production flags, script reports/TRICKS_1115/launch/finetune_mixft.py without --feed_w:
  A  = <v>_mixctl3  (stored)     A' = <v>_mixctl3b (stored, same command, other --out)
  C  = <v>_<suffix> (today's re-run of the same command, job "<v>:ftctl3c:ctl" of ftarm_lane.sh)
  S  = <v>_n2cpv3s3 (the stored production fine-tune of nn_x/w5/finetune.py, same flags; reported, NOT part of the criterion)

CRITERION (the criterion of mixft_pathcheck.py with O replaced by C; written 2026-11-16 before any run of this task; stored values of the A/A band:
  d(A, A') = 6.559e-09, n(A, A') = 0.000131, 2 x se(A, A') = 0.000129, reports/TRICKS_1115/followup/logs/pathcheck_out.txt):
  d(X, Y) = 1 - Pearson correlation of the two out-of-fold score vectors (raw saved scores, all val candidates);
  n(X, Y) = |member val NDCG(X) - member val NDCG(Y)|;  se(A, A') = paired SE of the per-feed NDCG difference A - A'.
  PASS iff  max(d(C, A), d(C, A')) <= 1.5 x d(A, A')
       AND  max(n(C, A), n(C, A')) <= max(n(A, A'), 2 x se(A, A')).
  FAIL -> stop: no arm is launched and no arm is read against the stored controls (the 8 controls would have to be re-run first).
Self-tests on stored files (no new run): suffix mixone3 (a stored run of the same path with weights of 1) must PASS; suffix mixft3 (a different fine-tune) must FAIL.
"""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
W5 = Path('<repo>/nn_x/w5/preds'); v = 'w5_fd10_kd05w192_s951'; csuf = sys.argv[1] if len(sys.argv) > 1 else 'ftctl3c'
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT / 'val/cand_target.npy')
zv = lambda x: feed_rank_z(ov, Mv, np.ascontiguousarray(np.asarray(x, np.float32)))[:, 1]
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv).astype(np.float64)
SUF = {'A': 'mixctl3', "A'": 'mixctl3b', 'C': csuf, 'S': 'n2cpv3s3'}
miss = [s for s in SUF.values() if not (W5 / f'nn_{v}_{s}_val_oof.npy').exists()]
assert not miss, f'missing val_oof files of {v}: {miss}'
X = {k: np.load(W5 / f'nn_{v}_{s}_val_oof.npy').astype(np.float64) for k, s in SUF.items()}
Zr = {k: zv(x).astype(np.float64) for k, x in X.items()}; E = {k: ev(x) for k, x in X.items()}
print(f'path check of the day, member {v}; C = suffix {csuf}')
print('member val NDCG: ' + ' | '.join(f'{k} ({SUF[k]}) {E[k].mean():.6f}' for k in X))
def pair(a, b):
    d = 1 - np.corrcoef(X[a], X[b])[0, 1]; dz = 1 - np.corrcoef(Zr[a], Zr[b])[0, 1]; e = E[a] - E[b]
    se = e.std() / np.sqrt(len(e)); same = (e == 0).mean()
    print(f'  {a:2s} vs {b:2s}: 1-corr raw {d:.3e} (corr {1 - d:.7f}) | 1-corr rank-z {dz:.3e} | max |raw diff| {np.abs(X[a] - X[b]).max():.4f} | NDCG diff {e.mean():+.6f} +- {se:.6f} | feeds with identical NDCG {same:.3f}')
    return d, abs(e.mean()), se
print('stored control vs stored control (A/A band):'); dAA, nAA, seAA = pair('A', "A'"); pair('A', 'S'); pair("A'", 'S')
print("today's control vs the stored controls:"); dCA, nCA, _ = pair('C', 'A'); dCB, nCB, _ = pair('C', "A'"); pair('C', 'S')
c1 = max(dCA, dCB) <= 1.5 * dAA; c2 = max(nCA, nCB) <= max(nAA, 2 * seAA)
print(f'CRITERION 1 (1-corr): max(d(C,A), d(C,A\')) = {max(dCA, dCB):.3e} <= 1.5 x d(A,A\') = {1.5 * dAA:.3e}: {"PASS" if c1 else "FAIL"}')
print(f'CRITERION 2 (member NDCG): max(n(C,A), n(C,A\')) = {max(nCA, nCB):.6f} <= max(n(A,A\') = {nAA:.6f}, 2 x se = {2 * seAA:.6f}): {"PASS" if c2 else "FAIL"}')
print('PATH CHECK: ' + ('PASS' if c1 and c2 else 'FAIL -> stop'))
