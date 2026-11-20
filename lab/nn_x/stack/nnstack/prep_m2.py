"""Add pool M2 (F7x+F7x70) arrays in the same layout as prep.py (member-equal pool, per-feed z)."""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from common import feed_meta
from stack_val import feed_rank_z
from metrics import ndcg_per_feed
P = Path('<repo>/preds'); D = Path('<repo>/nn_x/stack/nnstack/data')
z = lambda o, M, x: feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1]
for split in ('val', 'test'):
    _, M, o = feed_meta(split)
    for h in ('ft', 'to'): np.save(D / f'{split}_pool_M2_{h}.npy', z(o, M, np.load(P / f'ens_M2_{h}_{split}.npy')).astype(np.float32))
    sp = z(o, M, 0.5 * np.load(D / f'{split}_pool_M2_ft.npy') + 0.5 * np.load(D / f'{split}_pool_M2_to.npy'))
    np.save(D / f'{split}_pool_M2_spec.npy', sp.astype(np.float32))
    if split == 'val': print('M2 val spec', ndcg_per_feed(sp, np.load(D / 'val_y.npy'), o, M).mean())
Mh, oh = np.load(D / 'hold_meta.npy')
ms = [f'r6_cd_m02_r{i}' for i in range(101, 108)] + [f'r6_cd_m02_p70k_r{i}' for i in range(101, 108)]
s = np.mean([z(oh, Mh, np.load(P / f'holdout/nn_{m}_holdout.npy')) for m in ms], 0)
np.save(D / 'hold_pool_M2_to.npy', z(oh, Mh, s).astype(np.float32))
print('M2 hold', ndcg_per_feed(np.load(D / 'hold_pool_M2_to.npy'), np.load(D / 'hold_y.npy'), oh, Mh).mean())
