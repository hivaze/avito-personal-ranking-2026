# Writes a pool as a submission file. Each member's scores become per-feed z-scores; members are
# averaged with equal weight inside the fine-tuned branch and inside the train-only branch; the
# file is 0.5 z(fine-tuned mean) + 0.5 z(train-only mean).
# How it was run (docstring): build_pool.py TAG OUT.parquet member1,member2,...   (CPU, 2026-11-12)
# What it produced: preds/ens_<TAG>_{ft,to}_{val,test}.npy and subs/<OUT>; the best hard-label
# pool (G3, 20 members, public 0.392820) and the pool of only re-cooled members (M2, public
# 0.392434) are pools of this form.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/picks/build_pool.py
"""Build a member-equal pool submission (spec 0.5*z(ft) + 0.5*z(to), per-feed rank-z; identical to build_pall.py).
Usage: build_pool.py TAG OUT.parquet member1,member2,...   -> preds/ens_TAG_{ft,to}_{val,test}.npy + subs/OUT"""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from common import feed_meta
from stack_val import feed_rank_z
from make_sub import make_sub
P = Path('<repo>/preds')
tag, out, ms = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
assert len(set(ms)) == len(ms); print(len(ms), 'members:', ms)
for split, ftn, ton in [('val','n2cpv3_val_oof','trainonly_val'), ('test','n2cpv3_test','trainonly_test')]:
    f, M, o = feed_meta(split); z = lambda x: feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1]
    np.save(P/f'ens_{tag}_ft_{split}.npy', np.mean([z(np.load(P/f'nn_{m}_{ftn}.npy')) for m in ms], 0).astype(np.float32))
    np.save(P/f'ens_{tag}_to_{split}.npy', np.mean([z(np.load(P/f'nn_{m}_{ton}.npy')) for m in ms], 0).astype(np.float32))
ft_, Mt, ot = feed_meta('test'); zt = lambda n: feed_rank_z(ot, Mt, np.load(P/f'{n}.npy'))[:, 1]
make_sub((0.5*zt(f'ens_{tag}_ft_test') + 0.5*zt(f'ens_{tag}_to_test')).astype(np.float32), f'<repo>/subs/{out}')
print('written subs/' + out)
