"""Step 5 control: constant weights, different combiner over the four pool arrays (R4 = build_r4 form, ZM = raw z mean, RK = rank average, SM(T) = mean of per-feed softmax probabilities)."""
from lib2 import *
import json, sys
comp = sys.argv[1]; y = labels().astype(np.float32); f, M, off = meta('val'); g = groups(); q = pools4('val', None, None, tag=comp); n = len(M)
KEYS = ['Bft', 'Bto', 'Pft', 'Pto']
def amat(w, wf): return np.array([wf * (1 - w), (1 - wf) * (1 - w), wf * w, (1 - wf) * w], np.float64)
def tens(arrs, tag):
    p = CACHE / f'E5_{comp}_{tag}.npy'
    if p.exists(): return np.load(p)
    E = np.empty((n, len(WP), len(WF)))
    for i, w in enumerate(WP):
        for j, wf in enumerate(WF):
            a = amat(w, wf); s = a[0] * arrs[0] + a[1] * arrs[1] + a[2] * arrs[2] + a[3] * arrs[3]; E[:, i, j] = ndcg_fast(s.astype(np.float64), y, off, M)
    np.save(p, E); return E
T = {'R4': tensor(q, y, tag=comp), 'ZM': tens([q[k].astype(np.float64) for k in KEYS], 'ZM'), 'RK': tens([rankfeed(q[k], off, M).astype(np.float64) for k in KEYS], 'RK')}
for t in (0.5, 1.0, 2.0): T[f'SM{t}'] = tens([softmaxfeed(q[k], off, M, t) for k in KEYS], f'SM{t}')
SMS = ['SM0.5', 'SM1.0', 'SM2.0']; i2, i4, j5 = IP[0.2], IP[0.4], IF[0.5]
def cs(d): m, se = cluster_se(d, g); return (float(m), float(se))
fm = lambda t: f'{t[0]:+.5f}+-{t[1]:.5f}'
plain = {k: T[k][:, i2, j5] for k in T}; mr = {k: np.where(M >= 50, T[k][:, i4, j5], T[k][:, i2, j5]) for k in T}
res = {'comp': comp, 'fixed_plain': {}, 'fixed_mrule': {}, 'nested': {}}
print(f'=== {comp}: combiner control (constant weights). R4 PLAIN {plain["R4"].mean():.6f}, R4 MRULE {mr["R4"].mean():.6f}')
print('(i) at the plain weights (0.2, 0.5) and (iii) at the M-rule weights: combiner minus R4 (nothing fitted for ZM / RK / fixed-T SM)')
for k in ['ZM', 'RK'] + SMS:
    res['fixed_plain'][k] = cs(plain[k] - plain['R4']); res['fixed_mrule'][k] = cs(mr[k] - mr['R4']); print(f'  {k:6s} plain {fm(res["fixed_plain"][k])} | M-rule {fm(res["fixed_mrule"][k])}')
# nested: SM with T chosen on training folds at fixed weights; CONST2 (+T) per combiner
nest = {k: np.zeros(n) for k in ('SMn_plain', 'SMn_mrule', 'R4', 'ZM', 'RK', 'SM')}; picks = {k: [] for k in ('SMn_plain', 'SMn_mrule', 'R4', 'ZM', 'RK', 'SM')}
for s in range(5):
    fo = folds(g, s, 5)
    for c in range(5):
        tr = fo != c; te = ~tr
        b = max(SMS, key=lambda k: plain[k][tr].mean()); nest['SMn_plain'][te] += plain[b][te] / 5; picks['SMn_plain'].append(b)
        b = max(SMS, key=lambda k: mr[k][tr].mean()); nest['SMn_mrule'][te] += mr[b][te] / 5; picks['SMn_mrule'].append(b)
        for k in ('R4', 'ZM', 'RK'):
            m = T[k][tr].mean(0); i, j = np.unravel_index(m.argmax(), m.shape); nest[k][te] += T[k][te, i, j] / 5; picks[k].append((float(WP[i]), float(WF[j])))
        best = None
        for k in SMS:
            m = T[k][tr].mean(0); i, j = np.unravel_index(m.argmax(), m.shape)
            if best is None or m[i, j] > best[0]: best = (m[i, j], k, i, j)
        nest['SM'][te] += T[best[1]][te, best[2], best[3]] / 5; picks['SM'].append((best[1], float(WP[best[2]]), float(WF[best[3]])))
from collections import Counter
print('nested SM (T chosen on training folds) at fixed weights: plain', fm(cs(nest['SMn_plain'] - plain['R4'])), dict(Counter(picks['SMn_plain'])), '| M-rule', fm(cs(nest['SMn_mrule'] - mr['R4'])), dict(Counter(picks['SMn_mrule'])))
res['nested']['SMn_plain_vs_R4plain'] = cs(nest['SMn_plain'] - plain['R4']); res['nested']['SMn_mrule_vs_R4mrule'] = cs(nest['SMn_mrule'] - mr['R4'])
print('(ii) constant (w_pop, w_ft) [and T] chosen nested per combiner (5 x 5 user-grouped folds): nested combiner minus nested R4-CONST2 | minus R4 PLAIN | minus R4 MRULE | in-sample best constant minus R4 PLAIN')
for k in ('R4', 'ZM', 'RK', 'SM'):
    ins = max(T[kk].mean(0).max() for kk in (SMS if k == 'SM' else [k])) - plain['R4'].mean()
    res['nested'][k] = dict(vs_R4const2=cs(nest[k] - nest['R4']), vs_plain=cs(nest[k] - plain['R4']), vs_mrule=cs(nest[k] - mr['R4']), insample_vs_plain=float(ins), picks={str(a): b for a, b in Counter(picks[k]).most_common(4)})
    print(f'  {k:3s} {fm(res["nested"][k]["vs_R4const2"])} | {fm(res["nested"][k]["vs_plain"])} | {fm(res["nested"][k]["vs_mrule"])} | {ins:+.5f} | top picks {Counter(picks[k]).most_common(3)}')
json.dump(res, open(HERE / f'out_step5_{comp}.json', 'w'), indent=1)
