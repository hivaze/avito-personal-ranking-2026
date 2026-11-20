"""Build pool-of-pools candidates accepted in agg_out.txt (STUDY C). PoP = mean_s z(spec_s), spec_s = 0.5*z(mean z ft_s)+0.5*z(mean z to_s).
Writes NEW files only: preds/agg_S3_<tag>_{val,test}.npy, subs/cand_S3_<tag>.parquet.
Usage: build_pop.py TAG POOL1,POOL2,...   (pools from the fixed member lists below)"""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS']: os.environ[_k] = '12'
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from common import feed_meta, FLAT
from stack_val import feed_rank_z
from make_sub import make_sub
from metrics import ndcg_per_feed
P = Path('<repo>/preds')
FAM = {
    'F5cd': ['v15_cd_all_e10', 'v15_cd_all_s81', 'v15_cd_all_s82', 'v15_cd_allprew_s83', 'v15_cd_allprew_s84', 'v15_cd_allprew_s85'],
    'F6': ['v13_lr1_mix1', 'v14_s92', 'v14_s93', 'v14_s94', 'v14_s95', 'v14_s96'],
    'F9': [f'v22_pre2s_lr1_s{s}' for s in range(401, 407)], 'F10': [f'v23_gru_pre2s_s{s}' for s in range(501, 507)],
    'F12': [f'v26_d192_pre2s_s{s}' for s in range(701, 708)], 'F7x': [f'r6_cd_m02_r{s}' for s in range(101, 108)],
    'F7x70': [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]}
fam = lambda *fs: [m for f in fs for m in FAM[f]]
POOLS = {'PF7x': fam('F5cd', 'F6', 'F7x', 'F9', 'F10', 'F12'), 'G3': fam('F10', 'F7x', 'F7x70'), 'M2': fam('F7x', 'F7x70')}
tag, pools = sys.argv[1], sys.argv[2].split(',')
union = list(dict.fromkeys(m for p in pools for m in POOLS[p]))
print(tag, pools, 'union', len(union))
for split, ftn, ton in [('val', 'n2cpv3_val_oof', 'trainonly_val'), ('test', 'n2cpv3_test', 'trainonly_test')]:
    f, M, o = feed_meta(split); z = lambda x: feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1]
    specs = []
    for p in pools:
        ms = POOLS[p]
        ft = np.mean([z(np.load(P / f'nn_{m}_{ftn}.npy')) for m in ms], 0).astype(np.float32)
        to = np.mean([z(np.load(P / f'nn_{m}_{ton}.npy')) for m in ms], 0).astype(np.float32)
        for h, a in (('ft', ft), ('to', to)):  # parity with the frozen pool arrays
            ref = np.load(P / f'ens_{p}_{h}_{split}.npy'); print(f'  {split} {p} {h} max|diff| vs ens_{p}_{h}_{split}: {np.abs(ref - a).max():.2e}')
        specs.append(z(0.5 * z(ft) + 0.5 * z(to)))
    pop = np.mean(specs, 0).astype(np.float32)
    out = P / f'agg_S3_{tag}_{split}.npy'; assert not out.exists(), out
    np.save(out, pop); print('  wrote', out)
    if split == 'val':
        y = np.load(FLAT / 'val/cand_target.npy'); print(f'  val NDCG {ndcg_per_feed(pop, y, o, M).mean():.5f}')
    else:
        sub = Path(f'<repo>/subs/cand_S3_{tag}.parquet'); assert not sub.exists(), sub
        make_sub(pop, str(sub)); print('  wrote', sub)
print('members:', ','.join(union))
