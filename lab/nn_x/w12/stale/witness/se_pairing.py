"""witness: per-feed SE of a pool contrast, paired by snapshot against independent seeds (ref re-cools vs original wide students).
  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python se_pairing.py
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
lo, hi = (951, 952, 953, 954), (955, 956, 957, 958)
P(f'=== per-feed SE of same-recipe pool contrasts, {now()} (delta +- SE; val 14,423 feeds, holdout 30,000) ===')
P('contrast | val | stale-val | SV2 | pmix-pub | in-period holdout')
def row(name, A, B):
    r = wt.read([(1.0, A)], [(1.0, B)], lag_views=(), short=False)
    P(f'{name} | ' + ' | '.join(WR.cell(r[k]) for k in ('val', 'stale', 'SV2', 'pmix_pub', 'hold')))
row('8 vs 8 PAIRED by snapshot: ref s951-954 vs their originals', [ref(st, s) for st in 'kl' for s in lo], [orig(st, s) for st in 'kl' for s in lo])
row('8 vs 8 PAIRED by snapshot: ref s955-958 vs their originals', [ref(st, s) for st in 'kl' for s in hi], [orig(st, s) for st in 'kl' for s in hi])
row('8 vs 8 INDEPENDENT: ref s951-954 vs originals s955-958', [ref(st, s) for st in 'kl' for s in lo], [orig(st, s) for st in 'kl' for s in hi])
row('8 vs 8 INDEPENDENT: ref s955-958 vs originals s951-954', [ref(st, s) for st in 'kl' for s in hi], [orig(st, s) for st in 'kl' for s in lo])
row('8 vs 8 INDEPENDENT: originals s951-954 vs originals s955-958 (= W8 vs the newer 8)', [orig(st, s) for st in 'kl' for s in lo], [orig(st, s) for st in 'kl' for s in hi])
row('16 vs 16 PAIRED by snapshot: ref vs originals', [ref(st, s) for st in 'kl' for s in S8], [orig(st, s) for st in 'kl' for s in S8])
(LOGS / 'se_pairing.txt').write_text('\n'.join(OUT) + '\n')
