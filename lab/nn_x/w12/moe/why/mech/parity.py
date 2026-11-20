"""Parity check: my loaders + predict reproduce the stored val / holdout predictions of the selected checkpoints."""
import sys
import time
import numpy as np
import torch
sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/mech')
from mech_common import *  # noqa

dev = torch.device('cuda:0')
names = sys.argv[1].split(',')
m0, cfg = build(names[0], dev)
t0 = time.time(); va = load_val(dev, cfg['xt_names']); print('val load', time.time() - t0, flush=True)
ho = load_hold(dev, cfg['xt_names'])
yh = np.load(W5 / 'preds/holdout/holdout_y.npy'); Mh, oh = np.load(W5 / 'preds/holdout/holdout_meta.npy')
assert (ho.M_np == Mh).all() and (ho.cand_off_np == oh).all() and (ho.target_np == yh).all(), 'holdout layout mismatch'
print('holdout layout matches holdout_meta / holdout_y', flush=True)
for v in names:
    m, _ = build(v, dev)
    pv_s, ph_s = pred_paths(v)
    t0 = time.time(); pv = predict(m, va, tf32=True); t1 = time.time(); ph = predict(m, ho, tf32=False); t2 = time.time()
    sv, sh = np.load(pv_s), np.load(ph_s)
    nv, nvs = split_ndcg(va, pv), split_ndcg(va, sv)
    nh, nhs = split_ndcg(ho, ph), split_ndcg(ho, sh)
    print(f'{v}: val max|d| {np.abs(pv - sv).max():.2e} ndcg mine {nv.mean():.6f} stored {nvs.mean():.6f} | hold max|d| {np.abs(ph - sh).max():.2e} '
          f'ndcg mine {nh.mean():.6f} stored {nhs.mean():.6f} | t {t1 - t0:.1f}s / {t2 - t1:.1f}s', flush=True)
