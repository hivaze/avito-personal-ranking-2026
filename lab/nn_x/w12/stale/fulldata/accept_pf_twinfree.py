"""Twin-free acceptance of the full-data POPULARITY-FREE students (abl_fd10_pfkd05w192_s951-958). No held-out twin of that view exists, so the checks of accept_twins.py are
made against the accepted full-data pop1 students (abl_fd10_pop1kd05w192, same seeds, same lane and code) and against the held-out pop1 students (abl_e10_pop1kd05w192).
  A1 structure  : train feeds 3,151,208 and total steps 99,840 in the log; --holdout_in_train in the command; all four prediction files with the right row counts, finite.
  A2 positive control: IN-SAMPLE holdout NDCG of each pf student minus the mean out-of-sample holdout NDCG of the held-out pop1 wide students (accepted pop1 FD students: +0.016..+0.018).
  A3 level on val: train-only val of each student and of the 4-pools vs the pop1 FD students (information: the views differ, so no band applies).
CPU, read-only.  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/stale/fulldata/accept_pf_twinfree.py"""
import os, sys, json, re
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
A = ROOT / 'nn_x/abl'; bad = 0
def one(v):
    global bad
    log = (A / f'logs/{v}.log').read_text(); js = json.load(open(A / f'logs/{v}.json'))
    ok_feeds = 'train feeds 3151208' in log and 'total steps 99840' in log; ok_flag = 'holdout_in_train' in log or js.get('args', {}).get('holdout_in_train') is True or 'holdout_in_train' in json.dumps(js)
    tv = np.load(ROOT / f'preds/nn_{v}_trainonly_val.npy') if (ROOT / f'preds/nn_{v}_trainonly_val.npy').exists() else np.load(A / f'preds/nn_{v}_trainonly_val.npy')
    files = {s: (ROOT / f'preds/nn_{v}_{s}.npy') for s in ('trainonly_val', 'trainonly_test', 'n2cpv3s3_val_oof', 'n2cpv3s3_test')}
    shp = {s: (np.load(f, mmap_mode='r').shape[0] if f.exists() else None) for s, f in files.items()}
    ok_files = shp['trainonly_val'] == m.n['val'] == shp['n2cpv3s3_val_oof'] and shp['trainonly_test'] == m.n['test'] == shp['n2cpv3s3_test']
    hi = np.load(A / f'preds/holdout/nn_{v}_holdoutINSAMPLE.npy').astype(np.float32)
    a1 = ok_feeds and ok_flag and ok_files and np.isfinite(tv).all() and np.isfinite(hi).all(); bad += (not a1)
    return a1, ev(m.z('val', tv.astype(np.float32))).mean(), eh(m.z('hold', hi)).mean(), (ok_feeds, ok_flag, ok_files, shp)
ho = np.mean([eh(m.z('hold', np.load(A / f'preds/holdout/nn_abl_e10_pop1kd05w192_s{s}_holdout.npy').astype(np.float32))).mean() for s in (951, 952, 953, 954)])
print(f'held-out pop1 wide students (abl_e10_pop1kd05w192 s951-954): mean out-of-sample holdout NDCG {ho:.5f}')
for arm, seeds in (('pop1kd05w192', range(951, 955)), ('pfkd05w192', range(951, 959))):
    for s in seeds:
        v = f'abl_fd10_{arm}_s{s}'; a1, vl, hl, d = one(v)
        print(f'  {v}: A1 {"ok" if a1 else "DEFECT " + str(d)} | A2 in-sample holdout {hl:.5f} (lift {hl - ho:+.5f}) | A3 train-only val {vl:.5f}')
pool = lambda arm, seeds: ev(np.mean([m.z('val', np.load(ROOT / f'preds/nn_abl_fd10_{arm}_s{s}_trainonly_val.npy').astype(np.float32)) for s in seeds], 0))
p1 = pool('pop1kd05w192', range(951, 955)); f = lambda d: f'{d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f}'
for nm, sd in (('pf s951-954', range(951, 955)), ('pf s955-958', range(955, 959))):
    q = pool('pfkd05w192', sd); print(f'A3 4-pool train-only val {nm}: {q.mean():.5f} | minus pop1 FD s951-954 ({p1.mean():.5f}): {f(q - p1)}')
print('RESULT:', 'no structural defect' if bad == 0 else f'{bad} DEFECT(S)')
