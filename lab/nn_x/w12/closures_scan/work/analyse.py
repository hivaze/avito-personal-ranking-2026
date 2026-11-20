import json, re, csv, math, sys
import numpy as np
from scipy import stats, optimize
sys.path.insert(0, '<repo>/lab/nn_x/w12/closures_scan/work')
from ann import A
W = '<repo>/lab/nn_x/w12/closures_scan/'
rows = json.load(open(W + 'work/rows.json'))
CLOSED = ('no effect', 'hurt', 'inconclusive')

num = r'([+-]?\d+\.\d+)'
pat = re.compile(r'([+-]\d\.\d+)(?: ± (\d\.\d+))?(?: \(z [^)]*\))?(?: / ([+-]?\d\.\d+)(?: ± (\d\.\d+))?)?')

def autoparse(t):
    m = pat.search(t)
    if not m: return None, None, None, None
    f = lambda x: float(x) if x is not None else None
    return f(m.group(1)), f(m.group(2)), f(m.group(3)), f(m.group(4))

out = []
missing = [r['line'] for r in rows if r['outcome'] in CLOSED and r['line'] not in A]
assert not missing, missing
for r in rows:
    d = dict(line=r['line'], theme=r['theme'], idea=r['idea'], outcome=r['outcome'], effect=r['effect'], where=r['where'])
    if r['line'] in A:
        x = A[r['line']]
        d.update(x); d['source_of_numbers'] = 'manual'
    else:
        v, s, h, hs = autoparse(r['effect'])
        d.update(val=v, se=s, hold=h, hold_se=hs, best=None, best_se=None, best_hold=None, level='', seeds='', epochs='',
                 reason={'adopted': 'n/a (adopted)', 'helped then superseded': 'n/a (superseded)', 'not run': 'not run (bound, probe or prior)'}[r['outcome']],
                 detail='', rtype='model', vderiv=0, multi=0, flag='', source_of_numbers='auto (first signed number in cell)')
    z = None
    if d['val'] is not None and d['se']:
        z = d['val'] / d['se']
    d['z'] = z
    hz = None
    if d['hold'] is not None and d.get('hold_se'):
        hz = d['hold'] / d['hold_se']
    d['z_hold'] = hz
    out.append(d)

# ---------------- CSV
cols = ['line', 'theme', 'idea', 'outcome', 'val', 'se', 'z', 'hold', 'hold_se', 'best', 'best_se', 'best_hold', 'level', 'seeds', 'epochs',
        'reason', 'detail', 'flag', 'rtype', 'multi', 'vderiv', 'source_of_numbers', 'effect', 'where']
with open(W + 'catalogue.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['docs07_line', 'theme', 'idea', 'outcome', 'val_delta', 'val_se', 'val_z', 'holdout_delta', 'holdout_se', 'best_arm_val', 'best_arm_se',
                'best_arm_holdout', 'test_level', 'seeds', 'epochs', 'closure_reason', 'reason_detail', 'flag', 'row_type', 'multi_arm', 'val_derived',
                'numbers_from', 'measured_effect_text', 'where'])
    for d in out:
        w.writerow([('' if d[c] is None else (f'{d[c]:+.5f}' if isinstance(d[c], float) and c not in ('z', 'se', 'hold_se', 'best_se') else
                     (f'{d[c]:.2f}' if c == 'z' and d[c] is not None else (f'{d[c]:.5f}' if isinstance(d[c], float) else d[c])))) for c in cols])

# ---------------- counts
cl = [d for d in out if d['outcome'] in CLOSED]
print('closed rows', len(cl))
from collections import Counter
print('by outcome', Counter(d['outcome'] for d in cl))
print('by rtype', Counter(d['rtype'] for d in cl))
print('by reason (all closed)', Counter(d['reason'] for d in cl).most_common())
def sign_counts(ds, key='val'):
    v = [d[key] for d in ds if d[key] is not None]
    pos = sum(x > 0 for x in v); zer = sum(x == 0 for x in v); neg = sum(x < 0 for x in v)
    return len(v), pos, zer, neg
def binom_line(n, pos, zer):
    k = pos + 0.5 * zer
    p_one = stats.binomtest(int(round(pos)), n - zer, 0.5, alternative='greater').pvalue if n - zer > 0 else float('nan')
    p_two = stats.binomtest(int(round(pos)), n - zer, 0.5).pvalue if n - zer > 0 else float('nan')
    lo, hi = stats.binomtest(int(round(pos)), n - zer, 0.5).proportion_ci(0.95)
    return f'n={n} pos={pos} zero={zer} neg={n-pos-zer} share_pos(excl. zero)={pos/(n-zer):.3f} 95%CI [{lo:.3f},{hi:.3f}] p(two-sided vs 0.5)={p_two:.3g}'

res = {}
for name, sel in [('all closed', cl),
                  ('closed, model changes', [d for d in cl if d['rtype'] == 'model']),
                  ('closed, model, no effect + inconclusive', [d for d in cl if d['rtype'] == 'model' and d['outcome'] != 'hurt']),
                  ('closed, model, hurt', [d for d in cl if d['rtype'] == 'model' and d['outcome'] == 'hurt']),
                  ('closed, model, not clear loss', [d for d in cl if d['rtype'] == 'model' and d['reason'] != 'clear loss']),
                  ('closed, model, single-arm rows', [d for d in cl if d['rtype'] == 'model' and not d['multi']]),
                  ]:
    n, pos, zer, neg = sign_counts(sel)
    print(f'[{name}] primary val: ' + binom_line(n, pos, zer))
    res[name] = (n, pos, zer, neg)

# best-arm sensitivity: use best when present else primary
def bestval(d):
    c = [x for x in (d['val'], d['best']) if x is not None]
    return max(c) if c else None
for d in cl: d['bestval'] = bestval(d)
sel = [d for d in cl if d['rtype'] == 'model']
n, pos, zer, neg = sign_counts(sel, 'bestval')
print('[closed model, most favourable arm] ' + binom_line(n, pos, zer))

# ---------------- z distribution (rows with SE)
zs = [d for d in cl if d['z'] is not None]
print('closed rows with SE', len(zs), Counter(d['rtype'] for d in zs))
zm = [d for d in zs if d['rtype'] == 'model']
zv = np.array([d['z'] for d in zm])
print('model rows with SE', len(zm))
for lo, hi in [(-99, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 99)]:
    k = int(((zv >= lo) & (zv < hi)).sum())
    pnull = stats.norm.cdf(hi) - stats.norm.cdf(lo)
    print(f'  z in [{lo},{hi}): {k} ({k/len(zv):.3f}); null expects {pnull*len(zv):.1f} ({pnull:.3f})')
k12 = [(d['line'], round(d['z'], 2), d['val'], d['se']) for d in zm if 1 <= d['z'] < 2]
print('rows with 1<=z<2:', k12)
k2 = [(d['line'], round(d['z'], 2)) for d in zm if d['z'] >= 2]
print('rows z>=2:', k2)
print('mean z', zv.mean(), 'sd', zv.std(ddof=1), 'median', np.median(zv))
# with best arm
zb = []
for d in cl:
    if d['rtype'] != 'model': continue
    cands = []
    if d['z'] is not None: cands.append(d['z'])
    if d['best'] is not None and d['best_se']: cands.append(d['best'] / d['best_se'])
    if cands: zb.append(max(cands))
zb = np.array(zb)
print('best-arm z rows', len(zb), 'in [1,2):', int(((zb >= 1) & (zb < 2)).sum()), '>=2:', int((zb >= 2).sum()), 'pos:', int((zb > 0).sum()))

json.dump([{k: v for k, v in d.items()} for d in out], open(W + 'work/merged.json', 'w'), indent=0, default=str)
