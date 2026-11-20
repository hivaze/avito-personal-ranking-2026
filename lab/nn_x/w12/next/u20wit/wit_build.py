"""u20wit step 1: witness feed lists + the restricted train sub-split cache (CPU, about 2-4 min).
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit/wit_build.py
Writes cache/sets.npz (feed lists), cache/sub/ (arrays of the union of the five sets), cache/sub_meta.npz (per sub-feed: set id, M, fold, uid, posteriors)."""
from wit_common import *
import torch
torch.set_num_threads(12)
tm = train_meta(); S = witness_sets(tm)
M, xf, uid, pold, pv2 = tm['M'], tm['xf'], tm['uid'], tm['pold'], tm['pv2']
isH = np.zeros(tm['n'], bool); isH[tm['hold']] = True
print('holdout feeds', len(tm['hold']), 'by fold', np.bincount(xf[tm['hold']], minlength=5))
for k in SETS:
    i = S[k]
    print(f'{k}: feeds {len(i)} rows {int(M[i].sum())} | M min/median/max {M[i].min()}/{int(np.median(M[i]))}/{M[i].max()} | folds {np.bincount(xf[i], minlength=5)} | user-0 feeds {int((uid[i] == 0).sum())} '
          f'| users {len(np.unique(uid[i]))} | in holdout {int(isH[i].sum())} | v2>0.9 {float((pv2[i] > 0.9).mean()):.3f} | old p>0.5 (C cooldown pool if not holdout) {float(((pold[i] > 0.5) & ~isH[i]).mean()):.3f} old p>0.9 {float((pold[i] > 0.9).mean()):.3f}')
allf = np.unique(np.concatenate([S[k] for k in SETS])); assert len(allf) == sum(len(S[k]) for k in SETS), 'sets overlap'
ck = torch.load(CKPT / 'w5_e10_xf0_s951.pt', map_location='cpu', weights_only=False); xt = ck['cfg']['xt_names']
for v in ('w5_e10_base_s951', 'w5_e10_kd05w192_s951'):
    assert torch.load(CKPT / f'{v}.pt', map_location='cpu', weights_only=False)['cfg']['xt_names'] == xt, v
build_subsplit('train', allf, CACHE / 'sub', xt_names=xt)
sid = np.full(len(allf), -1, np.int8)
for j, k in enumerate(SETS): sid[np.searchsorted(allf, S[k])] = j
np.savez(CACHE / 'sub_meta.npz', src_fidx=allf, set_id=sid, M=M[allf], xf=xf[allf], uid=uid[allf], pold=pold[allf], pv2=pv2[allf], ts=tm['ts'][allf])
print('DONE')
