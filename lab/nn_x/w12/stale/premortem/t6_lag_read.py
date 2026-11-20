"""premortem t6_lag_read: the lagged-holdout read of PREDECLARED block 3 (subset 13,424 feeds, CPU fp32, train-only level).
Predictions: witness cache/pred/<view>/ (W8 members, Pd, narrow students) and my cache/lag/<view>/ (Pw, W16 newer members). Writes out/t6_lag.md."""
import sys
from pm_common import *
WIT = ROOT / 'nn_x/w12/stale/witness/cache'; SUB = WIT / 'sub'; MY = CACHE / 'lag'
M = np.load(SUB / 'M.npy').astype(np.int64); off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
hpos = np.load(SUB / 'hpos.npy'); y = np.load(W5 / 'holdout/holdout_y.npy').astype(np.float32)[hpos]
from pm_common import _ndcg, _feed_z
Z = lambda x: _feed_z(off, M, np.ascontiguousarray(x, np.float32)); N = lambda s: _ndcg(np.ascontiguousarray(s, np.float32), y, off, M)
def ppath(view, v):
    for d in (MY, WIT / 'pred'):
        p = d / view / f'{v}.npy'
        if p.exists(): return p
    return None
def have(names, view): return all(ppath(view, v) is not None for v in names)
def pool(names, view): return np.mean([Z(np.load(ppath(view, v))) for v in names], 0)
out = []
def P_(s=''):
    print(s, flush=True); out.append(s)
P_(f'## T6. Lagged holdout, subset (days >= Sep 13), {len(M)} feeds (all M >= 50), train-only level, CPU fp32. S(w) = NDCG((1 - w) z(base) + w z(pop)) - NDCG(z(base))')
bases = [b for b in ('W8', 'W8b', 'W16') if have(G[b], 'lag0') and have(G[b], 'priv')]
pops = [p for p in ('Pd', 'Pw', 'Px8') if have(G[p], 'lag0') and have(G[p], 'priv')]
P_(f'Available: bases {bases}; popularity-light pools {pops}.')
PL = {(n_, v): pool(G[n_], v) for n_ in bases + pops for v in ('lag0', 'priv')}
P_(); P_('| pool | level lag 0 | level private-like | change |')
P_('|---|---|---|---|')
LV = {}
for n_ in bases + pops:
    a, b = N(Z(PL[(n_, 'lag0')])), N(Z(PL[(n_, 'priv')])); LV[n_] = (a, b)
    P_(f'| {n_} | {a.mean():.5f} | {b.mean():.5f} | {f5(wms(b - a))} |')
P_(); P_('| pop pool - base pool | lag 0 | private-like |')
P_('|---|---|---|')
for b in bases:
    for p in pops:
        P_(f'| {p} - {b} | {f5(wms(LV[p][0] - LV[b][0]))} | {f5(wms(LV[p][1] - LV[b][1]))} |')
P_(); P_('| base + pop | w | S_0(w) (lag 0) | S_lag(w) (private-like) | D = S_lag - S_0 | S_lag(w) - S_lag(0.2) | S_0(w) - S_0(0.2) |')
P_('|---|---|---|---|---|---|---|')
for b in bases:
    for p in pops:
        S = {}
        for w in (0.1, 0.2, 0.3, 0.4):
            S[w] = tuple(N((1 - w) * Z(PL[(b, v)]) + w * Z(PL[(p, v)])) - LV[b][i] for i, v in enumerate(('lag0', 'priv')))
        for w in (0.1, 0.2, 0.3, 0.4):
            s0, sl = S[w]
            c1 = '' if w == 0.2 else f5(wms(sl - S[0.2][1])); c0 = '' if w == 0.2 else f5(wms(s0 - S[0.2][0]))
            P_(f'| {b} + {p} | {w:.1f} | {f5(wms(s0))} | {f5(wms(sl))} | {f5(wms(sl - s0))} | {c1} | {c0} |')
if 'W16' in bases and 'W8' in bases:
    P_(); P_('| cross-composition (private-like view) | delta |')
    P_('|---|---|')
    mix = lambda b, p, w: N((1 - w) * Z(PL[(b, 'priv')]) + w * Z(PL[(p, 'priv')]))
    ref = mix('W8', 'Pd', 0.2)
    for b, p, w in (('W16', 'Pd', 0.2), ('W16', 'Px8', 0.2), ('W16', 'Px8', 0.3), ('W16', 'Px8', 0.4), ('W16', 'Pw', 0.4), ('W8', 'Px8', 0.4)):
        if p in pops: P_(f'| [{b} + {w:.1f} {p}] - [W8 + 0.2 Pd = the TO half of W8P20] | {f5(wms(mix(b, p, w) - ref))} |')
    P_(f'| W16 - W8 | {f5(wms(LV["W16"][1] - LV["W8"][1]))} (lag 0: {f5(wms(LV["W16"][0] - LV["W8"][0]))}) |')
    P_(f'| W16 - [W8 + 0.2 Pd] | {f5(wms(LV["W16"][1] - ref))} |')
open(OUT / 't6_lag.md', 'w').write('\n'.join(out) + '\n')
