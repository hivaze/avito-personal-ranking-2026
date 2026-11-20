"""witness (POST-HOC, information only; DECLARATION block 15): the hz 16-pool against each same-recipe reference pool and against their union.
  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python hz_refs.py [arm = hz]
"""
import os
os.environ.setdefault('WT_THREADS', '4')
from wt_common import *
import wt_read as WR
ARM = sys.argv[1] if len(sys.argv) > 1 else 'hz'
wt = WR.Witness(); OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)
nm = lambda a, st, s: f'w5_{st}cd_{a}_s{s}'
orig = lambda st, s: f'w5_e10_{"kd05w192" if st == "k" else "kd05t25w192"}_s{s}'
ps = [(st, s) for st in 'kl' for s in S8]
KS = ('val', 'val_M>=50', 'val_M<50', 'pmix_pub', 'pmix_priv', 'stale', 'SV2', 'SVpriv', 'hold', 'HOLD-SHORT', 'lag_lag0', 'lag_priv', 'D_priv')
P(f'=== {ARM} 16-pool against each same-recipe reference (POST-HOC, information), {now()} ===')
RES = {}
for suf in ('', '@last'):
    A = [(1.0, [nm(ARM, *p) + suf for p in ps])]
    refs = {'ref (the rule\'s reference)': [nm('ref', *p) + suf for p in ps], 'ref2': [nm('ref2', *p) + suf for p in ps]}
    if suf == '':
        refs['original wide students (W16)'] = [orig(*p) for p in ps]
        refs['ref + ref2 + originals (48)'] = refs["ref (the rule's reference)"] + refs['ref2'] + refs['original wide students (W16)']
    else:
        refs['ref + ref2 (32)'] = refs["ref (the rule's reference)"] + refs['ref2']
    P(f'\n--- {"SELECTED" if suf == "" else "@last"} checkpoints: {ARM} 16-pool minus the reference pool ---')
    P('reference | ' + ' | '.join(KS))
    for name, mem in refs.items():
        B = [(1.0, mem)]
        if not (wt.have(B, 'val') and wt.have(B, 'hold')):
            P(f'{name}: stored files missing'); continue
        r = wt.read(A, B, lag_views=('lag0', 'priv'))
        RES[f'{name}{suf}'] = WR.dump(r)
        P(f'{name} | ' + ' | '.join((f'{r[k][0]:+.5f} +- {r[k][1]:.5f} (z {r[k][0] / r[k][1]:+.1f})' if k in r else 'n/a') for k in KS))
json.dump(RES, open(CACHE / f'readout/{ARM}_refs.json', 'w'), indent=1)
(LOGS / f'{ARM}_refs.txt').write_text('\n'.join(OUT) + '\n')
