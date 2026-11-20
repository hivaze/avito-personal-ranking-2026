"""Teacher logits for KD: main logits of train-only checkpoints on EVERY train feed (eval mode; domain token as in training = old p_random mix).
  CUDA_VISIBLE_DEVICES=g python dump_teacher.py <tag> <ckpt path> [<ckpt path> ...]  -> out/teacher_<tag>.npy (float32 SUM over the ckpts) + .n"""
import sys
sys.path.insert(0, '<repo>/nn_x/w5')
import numpy as np, torch
from common import predict, ROOT
from data import SplitData
from model import Ranker
torch.set_num_threads(12)
tag, cks = sys.argv[1], sys.argv[2:]
dev = torch.device('cuda:0'); tr = SplitData('train', dev)
pr = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0); pr[tr.M_np < 50] = 0.0
prt = torch.from_numpy(pr).to(dev); acc = None; xt = 'unset'
for c in cks:
    ck = torch.load(c, map_location=dev, weights_only=False); cfg = ck['cfg']
    if cfg.get('xt_names') != xt:
        tr.xt = None
        if cfg.get('xt_names'): tr.attach_xt(cfg['xt_names'])
        xt = cfg.get('xt_names')
    tr.dom = torch.stack([1 - prt, prt, torch.zeros_like(prt)], 1) if cfg.get('domain') else None
    m = Ranker(**cfg).to(dev); m.load_state_dict(ck['state_dict'])
    s = predict(m, tr); acc = s.astype(np.float32) if acc is None else acc + s
    print(c, float(np.abs(s).mean()), float(s.std()), flush=True); del m; torch.cuda.empty_cache()
np.save(f'out/teacher_{tag}.npy', acc); open(f'out/teacher_{tag}.n', 'w').write(str(len(cks))); print('DONE', flush=True)
