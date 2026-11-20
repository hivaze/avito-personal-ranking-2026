"""History truncation for the tiny attention model: keep only events with hpos < K (train-fit, val NDCG)."""
import sys, time
import numpy as np
import torch
sys.argv = [sys.argv[0], '200000']
src = open('06c_history_attention.py').read().split("t0 = time.time()")[0]
exec(src)
t0 = time.time()
Dv = prep(load_split('val'))
rv = np.random.default_rng(0).random(len(Dv['t_np'])) * 1e-6
fidx = np.sort(np.random.default_rng(42).choice(len(feeds('train')), 200000, replace=False))
Dt = prep(load_split('train', fidx))
Va_t, Va_v = Dt['Va'].clone(), Dv['Va'].clone()
nfv = len(Dv['M_np'])
ref = None
for K in [256, 128, 64, 32, 16, 8]:
    ar = torch.arange(HL, device=dev)[None, :] < K
    Dt['Va'] = Va_t & ar; Dv['Va'] = Va_v & ar
    torch.manual_seed(0)
    m = Att().to(dev)
    fit(m, Dt, torch.arange(len(Dt['M_np']), device=dev), 1)
    pf = ndcg_per_feed(predict(m, Dv) + rv, Dv['t_np'], Dv['off_np'], Dv['M_np'])
    if ref is None: ref = pf
    d = pf - ref
    print(f'K={K:3d}: val NDCG {pf.mean():.4f}  vs K=256 {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(nfv):.4f})  [{time.time() - t0:.0f}s]', flush=True)
