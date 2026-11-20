import json, numpy as np
from scipy import stats
exec(open('mixture.py').read().split("out = {}")[0])  # reuse data + functions
res = {}
for infl in (1.0, 1.2, 1.4):
    for use_best in (False, True):
        dd, ss = [], []
        for d in S if not use_best else [x for x in cl]:
            v, s = f(d['val']), f(d['se'])
            if use_best:
                c = []
                if v is not None and s: c.append((v / s, v, s))
                if f(d['best']) is not None and f(d['best_se']): c.append((f(d['best']) / f(d['best_se']), f(d['best']), f(d['best_se'])))
                if not c: continue
                _, v, s = max(c)
            dd.append(v); ss.append(s * infl)
        dd, ss = np.array(dd), np.array(ss); zz = dd / ss
        K = np.stack([dens(dd, ss, zz, dj) for dj in dgrid], axis=1)
        NEGx = np.stack([neg_comp(t, K) for t in taus], axis=1)
        C0x = dens(dd, ss, zz, 0.0)
        r = fit_three(0.0003, C0=C0x, NEG=NEGx, d=dd, s=ss, z=zz)
        key = f'infl{infl}_best{int(use_best)}'
        res[key] = dict(n=len(dd), pi_pos=r['pi_pos'], ci=r['ci'], exp=float(r['post'].sum()), pi_neg=r['pi_neg'], tau=r['tau'])
        print(key, 'n', len(dd), 'pi_pos', r['pi_pos'], 'CI', r['ci'], 'count CI', [round(x*len(dd)) for x in r['ci']], 'pi_neg', round(r['pi_neg'],2))
json.dump(res, open(W + 'work/sens.json', 'w'), indent=1)
