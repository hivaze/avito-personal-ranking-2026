"""u20wit: CPU sub-split of a random subsample (default 8,000) of the CROSS-USER holdout feeds (the 24,026 holdout feeds whose user is NOT in fold 0), so that the inductive baseline of the
transductive read (B - A on feeds nobody trained on, of users B never saw) is a CPU fp32 replay like the witness itself (CPU vs CPU only).
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit/xh_build.py     -> cache/subxh/, cache/subxh_meta.npz"""
from wit_common import *
import torch
torch.set_num_threads(12)
tm = train_meta(); hold = tm['hold']; xf = tm['xf']; ids = hold[xf[hold] != 0]; assert (np.diff(ids) > 0).all()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000  # random subsample (machine load 160 at 03:00 UTC: a full pass of 24,026 feeds costs 10 min per checkpoint); 0 = all
if 0 < N < len(ids): ids = np.sort(np.random.default_rng(20261115).choice(ids, N, replace=False))
xt = torch.load(CKPT / 'w5_e10_u20a_s951.pt', map_location='cpu', weights_only=False)['cfg']['xt_names']
assert xt == json.loads((CACHE / 'sub/xt_names.json').read_text())
build_subsplit('train', ids, CACHE / 'subxh', xt_names=xt)
np.savez(CACHE / 'subxh_meta.npz', src_fidx=ids, M=tm['M'][ids], xf=xf[ids], uid=tm['uid'][ids], pold=tm['pold'][ids], pv2=tm['pv2'][ids])
print(f'XH: {len(ids)} of the 24,026 cross-user holdout feeds (random subsample, seed 20261115), rows {int(tm["M"][ids].sum())}, folds {np.bincount(xf[ids], minlength=5)}, user-0 feeds {int((tm["uid"][ids] == 0).sum())}, mean M {tm["M"][ids].mean():.1f}'); print('DONE')
