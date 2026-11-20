"""nn_x/abl/ftbag copy of nn_x/abl/popfree/build_pool.py (FT seed-bagging task, pre-declared PREDECLARED_ftbag.txt 2026-11-13T19:08:07Z).
The ONLY change to the pool computation: the FT branch reads the 3-seed-bagged fine-tune files nn_<m>_n2cpv3s3_{val_oof,test}.npy
instead of nn_<m>_n2cpv3_*; the train-only branch (nn_<m>_trainonly_{val,test}.npy) and the spec 0.5*z(ft) + 0.5*z(to) with per-feed
rank-z are unchanged. Added: refuses to run if subs/OUT or any preds/ens_TAG_* already exists (existing files are never modified).
Usage: build_pool_s3.py TAG OUT.parquet member1,member2,...   -> preds/ens_TAG_{ft,to}_{val,test}.npy + subs/OUT"""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from common import feed_meta
from stack_val import feed_rank_z
from make_sub import make_sub
P = Path('<repo>/preds'); S = Path('<repo>/subs')
FT = 'n2cpv3s3'
tag, out, ms = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
assert len(set(ms)) == len(ms); print(len(ms), 'members:', ms, '| FT suffix', FT)
clash = ([S / out] if (S / out).exists() else []) + sorted(P.glob(f'ens_{tag}_*'))
assert not clash, f'REFUSED: existing files would be overwritten: {clash}'
missing = [m for m in ms for f in (f'nn_{m}_{FT}_val_oof.npy', f'nn_{m}_{FT}_test.npy', f'nn_{m}_trainonly_val.npy', f'nn_{m}_trainonly_test.npy') if not (P / f).exists()]
assert not missing, f'REFUSED: missing member files for {sorted(set(missing))}'
for split, ftn, ton in [('val', f'{FT}_val_oof', 'trainonly_val'), ('test', f'{FT}_test', 'trainonly_test')]:
    f, M, o = feed_meta(split); z = lambda x: feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1]
    np.save(P/f'ens_{tag}_ft_{split}.npy', np.mean([z(np.load(P/f'nn_{m}_{ftn}.npy')) for m in ms], 0).astype(np.float32))
    np.save(P/f'ens_{tag}_to_{split}.npy', np.mean([z(np.load(P/f'nn_{m}_{ton}.npy')) for m in ms], 0).astype(np.float32))
ft_, Mt, ot = feed_meta('test'); zt = lambda n: feed_rank_z(ot, Mt, np.load(P/f'{n}.npy'))[:, 1]
make_sub((0.5*zt(f'ens_{tag}_ft_test') + 0.5*zt(f'ens_{tag}_to_test')).astype(np.float32), str(S / out))
print('written subs/' + out)
