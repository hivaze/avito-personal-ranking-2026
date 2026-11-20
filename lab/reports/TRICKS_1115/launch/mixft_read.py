"""MIXFT read-out (reports/TRICKS_1115/SYNTHESIS.md G2; rule there, to be copied into the ledger BEFORE the first fine-tune). CPU, read-only.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python reports/TRICKS_1115/launch/mixft_read.py [arm suffix = mixft1] [control suffix = mixctl1]
Members: the 8 all-data wide students of FD8P20 (w5_fd10_kd05w192 / kd05t25w192, s951-954), one fine-tune seed each, arm and control from the SAME script and seeds
(reports/TRICKS_1115/launch/finetune_mixft.py with / without --feed_w). FILE(x) = 0.5 z(0.8 z(FT pool of the 8 members, x) + 0.2 z(stored 3-seed FT pool of the 4 popularity-light
students)) + 0.5 z(stored train-only branch preds/ens_FD8P20kd_to_val.npy). Only the wide fine-tuned pool differs between arm and control."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
from pathlib import Path
R = Path('<repo>'); P = R / 'preds'; W5 = R / 'nn_x/w5/preds'
A = sys.argv[1] if len(sys.argv) > 1 else 'mixft1'; C = sys.argv[2] if len(sys.argv) > 2 else 'mixctl1'
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT / 'val/cand_target.npy')
zv = lambda x: feed_rank_z(ov, Mv, np.ascontiguousarray(np.asarray(x, np.float32)))[:, 1]
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv).astype(np.float64)
WT = pmix.weights()
MEM = [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(951, 955)]
have = [v for v in MEM if (W5 / f'nn_{v}_{A}_val_oof.npy').exists() and (W5 / f'nn_{v}_{C}_val_oof.npy').exists()]
print(f'members with arm and control on disk: {len(have)} of 8'); assert len(have) == 8, 'the rule is read on all 8 members'
zpop = zv(np.mean([zv(np.load(P / f'nn_abl_fd10_pop1kd05w192_s{s}_n2cpv3s3_val_oof.npy')) for s in range(951, 955)], 0))
zto = zv(np.load(P / 'ens_FD8P20kd_to_val.npy'))
def file_(suf):
    ft = zv(np.mean([zv(np.load(W5 / f'nn_{v}_{suf}_val_oof.npy')) for v in MEM], 0))
    return ev(0.5 * zv(0.8 * ft + 0.2 * zpop) + 0.5 * zto), ev(ft)
(fa, pa), (fc, pc) = file_(A), file_(C)
def f(d, w=None, m=None):
    if m is not None: d = d[m]; w = None if w is None else w[m]
    mu, s = (d.mean(), d.std() / np.sqrt(len(d))) if w is None else pmix.wmean_se(d, w)
    return mu, s, f'{mu:+.5f} +- {s:.5f} (z {mu / s:+.2f})'
d = fa - fc
print(f'FILE control {fc.mean():.5f} | FILE arm {fa.mean():.5f} | wide FT pool alone: control {pc.mean():.5f}, arm {pa.mean():.5f}')
r = {k: f(d, w, m) for k, (w, m) in {'val': (None, None), 'pmix-private': (WT['private'], None), 'pmix-public': (WT['public'], None), 'M<50': (None, Mv < 50), 'M>=50': (None, Mv >= 50)}.items()}
print('FILE arm minus control: ' + ' | '.join(f'{k} {v[2]}' for k, v in r.items()))
print('wide FT pool arm minus control: val ' + f(pa - pc)[2] + ' | pmix-private ' + f(pa - pc, WT['private'])[2])
ok = r['pmix-private'][0] >= 1.5 * r['pmix-private'][1] and r['val'][0] >= -r['val'][1] and r['M<50'][0] >= -r['M<50'][1]
print(f"RULE (pmix-private >= +1.5 SE AND val >= -1 SE AND M<50 >= -1 SE): {'PASS -> fine-tune the 16 + 4 all-data members with 3 seeds and build the pick-2 candidate' if ok else 'NOT PASSED -> closed'}")
