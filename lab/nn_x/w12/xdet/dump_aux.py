"""Dump the K9 aux head (production-order predictor) for EVERY train candidate from one plateau checkpoint.
Domain token forced to 'production' for all feeds (no dependence on the old p_random). Dropout off (eval), aux head on.
  CUDA_VISIBLE_DEVICES=g python dump_aux.py <ckpt path> <tag>   -> out/aux_<tag>.npy (float16, train flat candidate order)"""
import sys
sys.path.insert(0, '<repo>/nn_x/w5')
import numpy as np, torch
from data import SplitData
from model import Ranker
torch.set_num_threads(12)
ck_path, tag = sys.argv[1], sys.argv[2]
dev = torch.device('cuda:0')
tr = SplitData('train', dev)
ck = torch.load(ck_path, map_location=dev, weights_only=False); cfg = ck['cfg']
if cfg.get('xt_names'): tr.attach_xt(cfg['xt_names'])
z = torch.zeros(tr.n, device=dev)
tr.dom = torch.stack([z + 1, z, z], 1) if cfg.get('domain') else None
m = Ranker(**cfg).to(dev); m.load_state_dict(ck['state_dict']); m.eval(); m.training = True  # top-level flag only: aux_out on, dropout off
out = torch.zeros(tr.n_cand, dtype=torch.float32, device=dev)
with torch.no_grad():
    for i, fb in enumerate(tr.batches(np.arange(tr.n), 65536, 1024, shuffle=False)):
        b = tr.gather(fb)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            m(b, use_pos=False)
        tr.scatter(out, b, m.aux_out)
        if i % 500 == 0: print(i, flush=True)
np.save(f'out/aux_{tag}.npy', out.cpu().numpy().astype(np.float16)); print('DONE', flush=True)
