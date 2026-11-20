"""DESIGN p02: member-level NDCG of stored hard-label and distilled families (val TO, holdout) + sd across seeds; shows seed-block effects."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'): os.environ[k] = '12'
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition'); sys.path.insert(0, '<repo>/src')
import numpy as np, pc_common as pc, json
from pathlib import Path
m = pc.Meta(); out = open('<repo>/reports/INPUTS_1116/design/out/p02_member_levels.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n')
fam = {'w5_e10_base': range(951, 957), 'abl_e10_pop1': range(951, 959), 'abl_e10_xdrop_pop': range(951, 955), 'abl_e10_pop2': range(951, 955),
       'w5_e10_kd05w192': range(951, 959), 'w5_e10_kd05t25w192': range(951, 959)}
for f, seeds in fam.items():
    v, h, mt = [], [], []
    for s in seeds:
        pv, ph, _ = pc.paths(f'{f}_s{s}')
        v.append(pc.ndcg_fast(m.z('val', np.load(pv)), m.y['val'], m.off['val'], m.M['val']).mean())
        h.append(pc.ndcg_fast(m.z('hold', np.load(ph)), m.y['hold'], m.off['hold'], m.M['hold']).mean())
        mt.append(os.path.getmtime(pv))
    import datetime
    P(f, '| val', ' '.join(f'{x:.5f}' for x in v), f'| sd {np.std(v, ddof=1):.5f}', '| hold', ' '.join(f'{x:.5f}' for x in h), f'| sd {np.std(h, ddof=1):.5f}',
      '| val file mtimes', ' '.join(datetime.datetime.utcfromtimestamp(t).strftime('%d-%H:%M') for t in mt))
out.close()
