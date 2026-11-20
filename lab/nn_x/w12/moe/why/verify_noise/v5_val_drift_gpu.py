"""verify_noise V5 (GPU N, read-only, ~5-10 min): were the 11-14 reference val predictions and the 11-19 arm val predictions made on the
same val inputs? Re-predict val with the CURRENT code + feature files from the stored 11-14 reference checkpoints (selected) s951 and s955,
exactly as train.py's final val predict (SplitData('val') + attach_xt(cfg xt_names) + common.predict, tf32 on), and compare with the stored
train-only val preds (corr, max |d|, val NDCG). A drift in val inputs after 11-14 (tree move 11-16, feature rebuilds) would bias every
arm-vs-reference val delta in the screen. Writes only a .txt next to this script.
  cd <repo>/lab/nn_x/w5 && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 ../../../.venv/bin/python ../w12/moe/why/verify_noise/v5_val_drift_gpu.py
"""
import os, sys, time
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '12'
sys.path.insert(0, '<repo>/lab/nn_x/w5')
import numpy as np
import torch
torch.set_num_threads(12)
import data as Dm  # noqa
from common import CKPT, PREDS, eval_ndcg, predict  # noqa
from model import Ranker  # noqa

OUT = '<repo>/lab/nn_x/w12/moe/why/verify_noise/v5_val_drift_gpu.txt'
lines = []
P = lambda s: (print(s, flush=True), lines.append(s))
dev = torch.device('cuda:0')
torch.backends.cuda.matmul.allow_tf32 = True
t0 = time.time()
va = Dm.SplitData('val', dev)
P(f'val split loaded: {va.n} feeds, {va.n_cand} candidates ({time.time() - t0:.0f}s)')
for v in ('w5_e10_kd05w192_s951', 'w5_e10_kd05w192_s955', 'w5_e10_kd05w192_s952'):
    ck = torch.load(CKPT / f'{v}.pt', map_location=dev, weights_only=False)
    m = Ranker(**ck['cfg']).to(dev); m.load_state_dict(ck['state_dict'])
    va.xt = None; va.attach_xt(ck['cfg'].get('xt_names') or [])
    with torch.no_grad():
        pb = predict(m, va)
    r = np.load(PREDS / f'nn_{v}_trainonly_val.npy')
    P(f'{v}: re-predicted val NDCG {eval_ndcg(pb, va):.5f} vs stored {eval_ndcg(r, va):.5f} | corr {np.corrcoef(r, pb)[0, 1]:.7f} max|d| {np.abs(r - pb).max():.2e}'
      f' mean|d| {np.abs(r - pb).mean():.2e} ({time.time() - t0:.0f}s)')
    del m; torch.cuda.empty_cache()
open(OUT, 'w').write('\n'.join(lines) + '\n')
