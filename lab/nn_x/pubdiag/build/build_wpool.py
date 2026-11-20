"""Weighted member pool builder, same spec as the lead's scratchpad build_pool.py (per-feed rank-z of every member,
branch = weighted mean of member z, final = 0.5*z(FT branch) + 0.5*z(TO branch)). With all weights 1 it is
build_pool.py exactly (checked by --verify TAG against existing preds/ens_TAG_*).
Usage: build_wpool.py TAG OUT.parquet m1[:w1],m2[:w2],...   -> preds/ens_TAG_{ft,to}_{val,test}.npy + subs/OUT
       build_wpool.py --verify TAG m1,m2,...                -> compare with existing preds/ens_TAG_* (no writes)
Refuses to overwrite any existing file."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "12"
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from common import feed_meta
from stack_val import feed_rank_z
from make_sub import make_sub
P = Path('<repo>/preds'); S = Path('<repo>/subs')
verify = sys.argv[1] == '--verify'
if verify:
    tag, spec = sys.argv[2], sys.argv[3]; out = None
else:
    tag, out, spec = sys.argv[1], sys.argv[2], sys.argv[3]
ms, ws = [], []
for t in spec.split(','):
    m, w = (t.split(':') + ['1'])[:2]; ms.append(m); ws.append(float(w))
assert len(set(ms)) == len(ms); print(len(ms), 'members, weight sum', sum(ws), list(zip(ms, ws)))
targets = [P / f'ens_{tag}_{h}_{s}.npy' for h in ('ft', 'to') for s in ('val', 'test')]
if not verify:
    assert not any(t.exists() for t in targets) and not (S / out).exists(), 'refusing to overwrite'
for split, ftn, ton in [('val', 'n2cpv3_val_oof', 'trainonly_val'), ('test', 'n2cpv3_test', 'trainonly_test')]:
    f, M, o = feed_meta(split); z = lambda x: feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1]
    for h, suf in (('ft', ftn), ('to', ton)):
        if all(w == 1.0 for w in ws):  # identical arithmetic to build_pool.py
            b = np.mean([z(np.load(P / f'nn_{m}_{suf}.npy')) for m in ms], 0).astype(np.float32)
        else:
            b = (sum(w * z(np.load(P / f'nn_{m}_{suf}.npy')).astype(np.float64) for m, w in zip(ms, ws)) / sum(ws)).astype(np.float32)
        if verify:
            ref = np.load(P / f'ens_{tag}_{h}_{split}.npy'); print(f'verify {tag} {h} {split}: max abs diff {np.abs(b - ref).max():.3e}')
        else:
            np.save(P / f'ens_{tag}_{h}_{split}.npy', b)
if not verify:
    ft_, Mt, ot = feed_meta('test'); zt = lambda n: feed_rank_z(ot, Mt, np.load(P / f'{n}.npy'))[:, 1]
    make_sub((0.5 * zt(f'ens_{tag}_ft_test') + 0.5 * zt(f'ens_{tag}_to_test')).astype(np.float32), str(S / out))
    print('written subs/' + out)
