"""D1b: SE of the null mean of the M-rule-shape contrast over the 44 same-recipe subsets (per-feed average over subsets, user-clustered SE)."""
from lib2 import *
import json
y = labels().astype(np.float32); f, M, off = meta('val'); g = groups(); big = M >= 50; PH = json.load(open(HERE / 'out_posthoc.json')); out = {}
for S in ('FD16', 'W16'):
    ZF = np.stack([member_z('val', m, 0) for m in MEM[S]]); ZT = np.stack([member_z('val', m, 1) for m in MEM[S]]); acc = {k: np.zeros(len(M)) for k in ('T', 'T25', 'c20_25', 'c40_25')}
    subs = [r['subset'] for r in PH['D1_null'][S]]
    for c in subs:
        ib = np.array([i for i in range(16) if i not in c]); ip = np.array(c)
        q = {'Bft': Z('val', ZF[ib].mean(0)), 'Bto': Z('val', ZT[ib].mean(0)), 'Pft': Z('val', ZF[ip].mean(0)), 'Pto': Z('val', ZT[ip].mean(0))}
        e2 = ev_w(q, 0.2, 0.5, y); e4 = ev_w(q, 0.4, 0.5, y); e25 = ev_w(q, 0.25, 0.5, y); mr = np.where(big, e4, e2)
        acc['T'] += (mr - e2) / len(subs); acc['T25'] += (mr - e25) / len(subs); acc['c20_25'] += (e2 - e25) / len(subs); acc['c40_25'] += (e4 - e25) / len(subs)
    o = {}
    for k, d in acc.items():
        o[k] = dict(all=cluster_se(d, g), short=cluster_se(d[~big], g[~big]), long=cluster_se(d[big], g[big]))
    out[S] = {k: {kk: [float(v[0]), float(v[1])] for kk, v in o[k].items()} for k in o}
    print(f'{S}: mean over 44 subsets, per-feed averaged (user-clustered SE): ' + ' | '.join(f'{k}: all {o[k]["all"][0]:+.5f}+-{o[k]["all"][1]:.5f}, short {o[k]["short"][0]:+.5f}+-{o[k]["short"][1]:.5f}, long {o[k]["long"][0]:+.5f}+-{o[k]["long"][1]:.5f}' for k in o))
json.dump(out, open(HERE / 'out_d1b.json', 'w'), indent=1)
