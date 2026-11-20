"""witness: markdown tables of an hz-type read from cache/readout/<arm>_vs_<ref>.json (written by d_hz.py); no computation, formatting only.
  cd <repo>/nn_x/w12/stale/witness && ../../../../.venv/bin/python mk_hz_tables.py [arm = hz] [ref = ref]
"""
import json, sys
from pathlib import Path
arm = sys.argv[1] if len(sys.argv) > 1 else 'hz'; ref = sys.argv[2] if len(sys.argv) > 2 else 'ref'
R = json.load(open(Path(__file__).parent / f'cache/readout/{arm}_vs_{ref}.json'))
ROWS = [('val, all 14,423 feeds', 'val'), ('val M>=50 (8,530)', 'val_M>=50'), ('val M<50 (5,893)', 'val_M<50'), ('pmix-public', 'pmix_pub'), ('pmix-private', 'pmix_priv'),
        ('**stale-val** (CPU fp32)', 'stale'), ('stale-val M>=50', 'stale_M>=50'), ('stale-val M<50', 'stale_M<50'), ('SV2 (public-like position)', 'SV2'), ('SVpriv (private-like position)', 'SVpriv'),
        ('cold positives, val (3,422)', 'coldpos'), ('warm positives, val (9,118)', 'warmpos'), ('cold positives, stale-val', 'stale_coldpos'), ('warm positives, stale-val', 'stale_warmpos'),
        ('in-period holdout, 30,000 (reported only)', 'hold'), ('in-period holdout, subset 13,424', 'hold_subset'), ('HOLD-SHORT', 'HOLD-SHORT')]
LAG = [('lagged subset, lag 0 (CPU)', 'lag_lag0'), ('lagged subset, private-like', 'lag_priv'), ('lagged subset, public-like', 'lag_pub'), ('D = private-like - lag 0', 'D_priv'),
       ('arm pool drop under the private-like lag', 'dropA_priv'), ('reference pool drop under the private-like lag', 'dropB_priv')]
def cell(d, k, z=True):
    if d is None or k not in d: return 'not scored'
    m, s = d[k][0], d[k][1]
    return f'{m:+.5f} ± {s:.5f}' + (f' (z {m / s:+.1f})' if z else '')
def lev(d, a, b):
    return 'not scored' if d is None or a not in d else f'{d[a]:.5f} / {d[b]:.5f}'
cols = [('selected', R.get('plain')), ('@last', R.get('plain@last') or R.get('free@last')), ('selected + pop1kd 0.2', R.get('mix')), ('@last + pop1kd 0.2', R.get('mix@last'))]
print('| witness (arm - reference, paired per feed) | ' + ' | '.join(f'{n}' for n, _ in cols) + ' |')
print('|---|' + '---|' * len(cols))
print('| pool level arm / reference, val | ' + ' | '.join(lev(d, 'A_val', 'B_val') for _, d in cols) + ' |')
print('| pool level arm / reference, stale-val | ' + ' | '.join(lev(d, 'A_stale', 'B_stale') for _, d in cols) + ' |')
print('| pool level arm / reference, in-period holdout | ' + ' | '.join(lev(d, 'A_hold', 'B_hold') for _, d in cols) + ' |')
for name, k in ROWS:
    print(f'| {name} | ' + ' | '.join(cell(d, k) for _, d in cols) + ' |')
lag = [(n, (d or {}).get('lag')) for n, d in cols]
print('| pool level arm / reference, lagged subset lag 0 | ' + ' | '.join(lev(d, 'A_lag_lag0', 'B_lag_lag0') for _, d in lag) + ' |')
print('| pool level arm / reference, lagged subset private-like | ' + ' | '.join(lev(d, 'A_lag_priv', 'B_lag_priv') for _, d in lag) + ' |')
for name, k in LAG:
    print(f'| {name} | ' + ' | '.join(cell(d, k) for _, d in lag) + ' |')
print()
for n, d in cols:
    if d: print(f'{n}: pairs {d.get("n_pairs", "")} {d.get("pairs", "")}; lagged pairs {d.get("lag_pairs", "")}; verdict {d.get("verdict", "")}')
for k in ('ref_vs_orig', 'arm_vs_orig', 'ref2_vs_ref'):
    if k in R:
        d = R[k]; print(f'{k}: ' + ' | '.join(f'{kk} {cell(d, kk)}' for kk in ('val', 'stale', 'SV2', 'hold', 'lag_lag0', 'lag_priv', 'D_priv') if kk in d))
