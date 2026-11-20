"""witness task C: calibration of the staleness witnesses on the public readings (rule: DECLARATION.txt block 1 (3), applied as written).
Train-only pools; A = arm, B = reference; public deltas are FINAL files (0.5 FT + 0.5 TO): a known mismatch.
  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python c_calib.py [aa]
"""
import os
os.environ.setdefault('WT_THREADS', '4')
from wt_common import *
import wt_read as WR

OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


PAIRS = [  # key, spec A, spec B, public delta (None = no public reading), note
    ('i   W8P20 - W8', '0.8*W8+0.2*pop1kd', 'W8', +0.000523, 'staleness family'),
    ('ii  W16 - W8', 'W16', 'W8', -0.000003, 'same recipe, more seeds'),
    ('iii S16 - W8', 'S16', 'W8', -0.000408, 'narrow students added'),
    ('iv  C3 - G3', 'C3', 'G3', +0.000634, 'staleness family (8 hard-label pop-light members added to 20)'),
    ('v   W8 - base4', 'W8', 'base4', None, 'scale reference, no public reading'),
    ('x1  W16P20 - W16', '0.8*W16+0.2*pop1kd', 'W16', None, 'information'),
    ('x2  W8Pw20 - W8', '0.8*W8+0.2*pop1kdw', 'W8', None, 'information (wide pop-light students)'),
]
SINGLE = ['val', 'pmix_pub', 'pmix_priv', 'stale', 'hold', 'lag_lag0', 'lag_priv', 'lag_pub', 'SV2']
COMBO = {'ivw(val, lag_pub)': ('val', 'lag_pub'), 'ivw(val, lag_priv)': ('val', 'lag_priv'), 'ivw(stale, lag_priv)': ('stale', 'lag_priv'),
         'ivw(SV2, lag_priv)': ('SV2', 'lag_priv'), 'ivw(pmix_pub, lag_pub)': ('pmix_pub', 'lag_pub')}
NPUB = 8562

wt = WR.Witness()
P(f'=== witness calibration {now()} (train-only pools; +- = 1 SE paired per feed; val 14,423 feeds, holdout 30,000, lagged subset 13,424) ===')
R = {}
for key, a, b, pub, note in PAIRS:
    A, B = WR.parse(a), WR.parse(b)
    r = wt.read(A, B)
    if 'val' not in r:
        P(f'\n{key}: val predictions missing, skipped'); continue
    r['public'] = pub; r['se_public'] = float(r['_pf']['val'].std(ddof=1) / np.sqrt(NPUB))
    for cn, (x, y) in COMBO.items():
        if x in r and y in r:
            r[cn] = WR.ivw(r[x], r[y]) + (0,)
    R[key] = r
    P('\n' + WR.show(r, f'{key}: [{a}] vs [{b}]  ({note}); PUBLIC {"n/a" if pub is None else f"{pub:+.6f} +- {r['se_public']:.5f}"}'))
    for cn in COMBO:
        if cn in r:
            P(f'   {cn:24s} {r[cn][0]:+.5f} +- {r[cn][1]:.5f}')

P('\n\n##### SIDE BY SIDE (delta A - B) #####')
cols = SINGLE + list(COMBO) + ['D_priv', 'D_pub']
P('pair | PUBLIC | ' + ' | '.join(cols))
for key, r in R.items():
    pub = 'n/a' if r['public'] is None else f"{r['public']:+.5f} +- {r['se_public']:.5f}"
    P(f'{key} | {pub} | ' + ' | '.join((f'{r[c][0]:+.5f} +- {r[c][1]:.5f}' if c in r else 'n/a') for c in cols))

P('\n##### SCORES of the candidate statistics over the pairs with a public reading (rule block 1 (3)) #####')
scored = [k for k, r in R.items() if r['public'] is not None]
P(f'scored pairs: {scored}; pairs counted for SIGN (|public| >= 1 public SE): {[k for k in scored if abs(R[k]["public"]) >= R[k]["se_public"]]}')
P('candidate | pairs available | SIGN right / counted | X2 (mean squared standardised distance to public) | slope of public on candidate through the origin | mean SE over the scored pairs')
S = {}
for c in SINGLE + list(COMBO):
    ks = [k for k in scored if c in R[k]]
    if not ks:
        continue
    cnt = [k for k in ks if abs(R[k]['public']) >= R[k]['se_public']]
    right = sum(np.sign(R[k][c][0]) == np.sign(R[k]['public']) for k in cnt)
    x2 = float(np.mean([(R[k][c][0] - R[k]['public']) ** 2 / (R[k][c][1] ** 2 + R[k]['se_public'] ** 2) for k in ks]))
    x = np.array([R[k][c][0] for k in ks]); y = np.array([R[k]['public'] for k in ks]); slope = float((x * y).sum() / (x * x).sum())
    S[c] = dict(n=len(ks), right=int(right), counted=len(cnt), x2=x2, slope=slope, se=float(np.mean([R[k][c][1] for k in ks])))
    P(f'{c:24s} | {len(ks)} | {right} / {len(cnt)} | {x2:.2f} | {slope:+.2f} | {S[c]["se"]:.5f}')
full = max(v['n'] for v in S.values())
elig = {c: v for c, v in S.items() if v['right'] == v['counted'] and v['n'] == full}
P(f'\ncandidates with no sign error on all {full} scored pairs: {sorted(elig, key=lambda c: elig[c]["x2"])}')
if elig:
    best = min(elig, key=lambda c: elig[c]['x2'])
    free = [c for c in elig if c in ('val', 'pmix_pub', 'pmix_priv')]
    bf = min(free, key=lambda c: elig[c]['x2']) if free else None
    P(f'smallest X2: {best} ({elig[best]["x2"]:.2f}); best free candidate (val / pmix): {bf} ({elig[bf]["x2"]:.2f})' if bf else f'smallest X2: {best}')
    if bf and best not in ('val', 'pmix_pub', 'pmix_priv') and elig[bf]['x2'] - elig[best]['x2'] <= 0.25:
        P(f'RULE: the CPU-inference candidate {best} is not better than the free candidate {bf} by more than 0.25 in X2 -> recommended by the rule: {bf}')
    else:
        P(f'RULE: recommended by the rule: {best}')
(CACHE / 'readout').mkdir(exist_ok=True)
json.dump({k: WR.dump(r) for k, r in R.items()}, open(CACHE / 'readout/calib.json', 'w'), indent=1)
json.dump(S, open(CACHE / 'readout/calib_scores.json', 'w'), indent=1)

if len(sys.argv) > 1 and sys.argv[1] == 'aa':
    # A/A: random N-vs-N splits of the 16 wide students (same recipe family): seed noise of each statistic against its per-feed SE
    P('\n\n##### A/A: random N-vs-N splits of the 16 wide students (W16), same-recipe null #####')
    W16 = G['W16']; rng = np.random.default_rng(20261115)
    for N, K in ((4, 40), (8, 40)):
        acc = {c: [] for c in ('val', 'stale', 'SV2', 'pmix_pub', 'pmix_priv', 'hold', 'lag_lag0', 'lag_priv', 'lag_pub', 'D_priv')}
        for _ in range(K):
            ix = rng.permutation(16); A = [(1.0, [W16[i] for i in ix[:N]])]; B = [(1.0, [W16[i] for i in ix[N:2 * N]])]
            r = wt.read(A, B, short=False)
            for c in acc:
                if c in r:
                    acc[c].append((r[c][0], r[c][1]))
        P(f' N = {N} vs {N} ({K} random disjoint splits): statistic | sd of the delta over splits | mean per-feed SE | sd / SE')
        for c, v in acc.items():
            if v:
                v = np.array(v)
                P(f'   {c:10s} | {v[:, 0].std(ddof=1):.5f} | {v[:, 1].mean():.5f} | {v[:, 0].std(ddof=1) / v[:, 1].mean():.2f}')
(LOGS / 'calib.txt').write_text('\n'.join(OUT) + '\n')
