"""witness: cross-check of wt_read.py (val / holdout / HOLD-SHORT columns) against the lead's logged harness-control read of arm_read8.py
(E_kd_predeclared.txt 03:14:29Z: K/L ref re-cools vs the original runs, 12 pairs: val +0.00036 +- 0.00033, holdout -0.00008 +- 0.00019, member -0.00018 / -0.00015).
  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python xcheck_arm_read8.py
"""
import os
os.environ.setdefault('WT_THREADS', '4')
from wt_common import *
import wt_read as WR
wt = WR.Witness(); OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)
orig = lambda st, s: f'w5_e10_{"kd05w192" if st == "k" else "kd05t25w192"}_s{s}'
ref = lambda st, s: f'w5_{st}cd_ref_s{s}'
p12 = [('k', s) for s in S8] + [('l', s) for s in (951, 953, 955, 957)]          # the 12 pairs complete at 03:14Z (json mtimes)
p16 = [(st, s) for st in 'kl' for s in S8]
P(f'=== cross-check against arm_read8.py (harness control), {now()} ===')
for name, ps in (('12 pairs of the 03:14Z read', p12), ('all 16 pairs', p16), ('the 4 later pairs (lcd s952, 954, 956, 958)', [p for p in p16 if p not in p12])):
    r = wt.read([(1.0, [ref(*p) for p in ps])], [(1.0, [orig(*p) for p in ps])], lag_views=())
    mv = np.mean([wt.V.level([(1.0, [ref(*p)])], 'val').mean() - wt.V.level([(1.0, [orig(*p)])], 'val').mean() for p in ps])
    mh = np.mean([wt.H.level([(1.0, [ref(*p)])], 'hold').mean() - wt.H.level([(1.0, [orig(*p)])], 'hold').mean() for p in ps])
    P(f'{name}: pool val {WR.cell(r["val"])} | pool holdout {WR.cell(r["hold"])} | HOLD-SHORT {WR.cell(r["HOLD-SHORT"])} | mean member delta val {mv:+.5f} / holdout {mh:+.5f}')
P('lead (12 pairs): val +0.00036 +- 0.00033, holdout -0.00008 +- 0.00019, member -0.00018 / -0.00015')
(LOGS / 'xcheck_arm_read8.txt').write_text('\n'.join(OUT) + '\n')
