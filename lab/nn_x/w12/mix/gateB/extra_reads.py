"""Fixed-comparator reads for the report (nothing fitted): pmix-weighted M-rule minus plain, cell contrasts of the M-rule family on both compositions and on the pooled tensor."""
from lib2 import *
import json, pmix
W = pmix.weights(); f, M, off = meta('val'); g = groups(); j5 = IF[0.5]; big = M >= 50; out = {}
for comp in ('FD16P4', 'W16P4'):
    E = np.load(CACHE / f'E_{comp}.npy'); pl_ = E[:, IP[0.2], j5]; mr = np.where(big, E[:, IP[0.4], j5], pl_); d = mr - pl_
    o = dict(val=cluster_se(d, g), pub=pmix.wmean_se(d, W['public']), priv=pmix.wmean_se(d, W['private']), long=cluster_se((E[:, IP[0.4], j5] - pl_)[big], g[big]), short=cluster_se((E[:, IP[0.4], j5] - pl_)[~big], g[~big]),
             c4=cluster_se(E[:, IP[0.4], j5] - pl_, g), c4_vs_mr=cluster_se(E[:, IP[0.4], j5] - mr, g), ft7=cluster_se(E[:, IP[0.2], IF[0.7]] - pl_, g), ft3=cluster_se(E[:, IP[0.2], IF[0.3]] - pl_, g),
             mr_ft6=cluster_se(np.where(big, E[:, IP[0.4], IF[0.6]], E[:, IP[0.2], IF[0.6]]) - mr, g), mr_ft4=cluster_se(np.where(big, E[:, IP[0.4], IF[0.4]], E[:, IP[0.2], IF[0.4]]) - mr, g))
    out[comp] = {k: [float(v[0]), float(v[1])] for k, v in o.items()}
    print(comp, ' | '.join(f'{k} {v[0]:+.5f}+-{v[1]:.5f}' for k, v in o.items()))
json.dump(out, open(HERE / 'out_extra.json', 'w'), indent=1)
