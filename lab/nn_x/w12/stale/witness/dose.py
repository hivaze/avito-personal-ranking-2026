"""witness: admixture weight of the popularity-light students under every witness (information; by-product of the tools).
  [(1 - w) z(base) + w z(pop1kd x4)] vs base, base = W8 and W16, w = 0.1 .. 0.4; train-only pools.
  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python dose.py
"""
import os
os.environ.setdefault('WT_THREADS', '4')
from wt_common import *
import wt_read as WR
wt = WR.Witness(); OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)
P(f'=== admixture dose under every witness, {now()} (train-only pools; +- 1 SE paired per feed) ===')
cols = ['val', 'val_M>=50', 'val_M<50', 'pmix_pub', 'pmix_priv', 'stale', 'SV2', 'hold', 'HOLD-SHORT', 'lag_lag0', 'lag_priv', 'lag_pub', 'D_priv']
RES = {}
for base in ('W8', 'W16'):
    P(f'\nbase {base}: w | ' + ' | '.join(cols))
    for w in (0.1, 0.2, 0.3, 0.4):
        r = wt.read(WR.parse(f'{1 - w:.1f}*{base}+{w:.1f}*pop1kd'), WR.parse(base))
        RES[f'{base}|{w}'] = WR.dump(r)
        P(f'  {w:.1f} | ' + ' | '.join((f'{r[c][0]:+.5f} +- {r[c][1]:.5f}' if c in r else 'n/a') for c in cols))
json.dump(RES, open(CACHE / 'readout/dose.json', 'w'), indent=1)
(LOGS / 'dose.txt').write_text('\n'.join(OUT) + '\n')
