"""witness: A/A null of a snapshot-paired re-cool contrast: ref2 (cooldown seed 2000 + seed) vs ref (1000 + seed), same snapshots, every witness.
Tells how far a 16-vs-16 re-cool contrast moves from a different cooldown seed alone, against its per-feed SE (the scale RULE hz uses).
  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python ref2_null.py
"""
import os
os.environ.setdefault('WT_THREADS', '4')
from wt_common import *
import stale_val_cpu as SVC
import wt_read as WR
wt = WR.Witness(); OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)
nm = lambda a, st, s: f'w5_{st}cd_{a}_s{s}'
orig = lambda st, s: f'w5_e10_{"kd05w192" if st == "k" else "kd05t25w192"}_s{s}'
ps = [(st, s) for st in 'kl' for s in S8 if all(done_json(nm(a, st, s)).exists() and stored_val(nm(a, st, s)).exists() and stored_holdout(nm(a, st, s)).exists()
                                                 and SVC.sv_path(nm(a, st, s)).exists() for a in ('ref2', 'ref'))]
P(f'=== ref2 vs ref (A/A of a snapshot-paired re-cool contrast), {now()}; pairs with val / holdout / CPU stale-val on both sides: {len(ps)} {[f"{st}{s}" for st, s in ps]} ===')
KS = ('val', 'val_M<50', 'stale', 'SV2', 'SVpriv', 'pmix_pub', 'hold', 'HOLD-SHORT')
RES = {}
def row(name, A, B):
    r = wt.read([(1.0, A)], [(1.0, B)], lag_views=())
    RES[name] = WR.dump(r)
    P(f'{name}: ' + ' | '.join(f'{k} {WR.cell(r[k])} (z {r[k][0] / r[k][1]:+.2f})' for k in KS if k in r))
if ps:
    row(f'ref2 - ref, {len(ps)}-pools', [nm('ref2', *p) for p in ps], [nm('ref', *p) for p in ps])
    for st in 'kl':
        q = [p for p in ps if p[0] == st]
        if len(q) >= 2 and len(q) < len(ps):
            row(f'ref2 - ref, {st}cd half ({len(q)})', [nm('ref2', *p) for p in q], [nm('ref', *p) for p in q])
    row(f'ref2 - originals, {len(ps)}-pools', [nm('ref2', *p) for p in ps], [orig(*p) for p in ps])
    row(f'ref  - originals, {len(ps)}-pools', [nm('ref', *p) for p in ps], [orig(*p) for p in ps])
    row(f'(ref + ref2) - originals, {2 * len(ps)} vs {len(ps)}', [nm(a, *p) for a in ('ref', 'ref2') for p in ps], [orig(*p) for p in ps])
    dm = {k: [] for k in ('val', 'stale', 'hold')}
    for p in ps:
        r = wt.read([(1.0, [nm('ref2', *p)])], [(1.0, [nm('ref', *p)])], lag_views=(), short=False)
        for k in dm: dm[k].append(r[k][0])
    P('single members ref2 - ref over the snapshots: ' + ' | '.join(f'{k} mean {np.mean(v):+.5f}, sd {np.std(v, ddof=1):.5f}, positive {int((np.array(v) > 0).sum())} of {len(v)}' for k, v in dm.items()))
json.dump(RES, open(CACHE / 'readout/ref2_null.json', 'w'), indent=1)
(LOGS / 'ref2_null.txt').write_text('\n'.join(OUT) + '\n')
