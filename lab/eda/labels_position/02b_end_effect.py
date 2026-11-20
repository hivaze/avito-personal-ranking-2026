"""Rate vs distance from feed end (maxpos - pos), and rate vs pos excluding the end region; NDCG of simple position rules on train."""
from common import *
import json
nb.set_num_threads(40)
F = feeds('train'); off = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
pos = np.asarray(cand('train', 'pos')).astype(np.int32); target = np.asarray(cand('train', 'target'))
z = np.load(f'{OUT}/02_perfeed_train.npz'); maxpos = z['maxpos']
dend = np.repeat(maxpos, M) - pos
# within-feed rank of pos (0..M-1) and rank from end
@nb.njit(parallel=True, cache=True)
def ranks(pos, off, M):
    r = np.empty(len(pos), np.int32)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        idx = np.argsort(pos[o:o + m])
        for k in range(m):
            r[o + idx[k]] = k
    return r
rk = ranks(pos, off, M)
rkend = np.repeat(M, M) - 1 - rk
res = {}
c = np.bincount(np.minimum(dend, 299), minlength=300); p = np.bincount(np.minimum(dend, 299), weights=target, minlength=300)
res['dend_count'] = c.tolist(); res['dend_npos'] = p.tolist()
c = np.bincount(rkend, minlength=300); p = np.bincount(rkend, weights=target, minlength=300)
res['rkend_count'] = c.tolist(); res['rkend_npos'] = p.tolist()
c = np.bincount(rk, minlength=300); p = np.bincount(rk, weights=target, minlength=300)
res['rk_count'] = c.tolist(); res['rk_npos'] = p.tolist()
# rate vs pos for candidates far from end (rkend>=20)
far = rkend >= 20
c = np.bincount(pos[far], minlength=300); p = np.bincount(pos[far], weights=target[far], minlength=300)
res['pos_far_count'] = c.tolist(); res['pos_far_npos'] = p.tolist()
# share of positives that are the last item; share of feeds whose last item is positive
last = rkend == 0
res['P_last_pos'] = float(target[last].mean()); res['P_first_pos'] = float(target[rk == 0].mean())
res['share_pos_last'] = float(target[last].sum() / target.sum())
res['share_pos_first'] = float(target[rk == 0].sum() / target.sum())
print({k: v for k, v in res.items() if not isinstance(v, list)})
# NDCG of rules on train
nd = {}
nd['pos_asc'] = ndcg_pf(-pos.astype(float), target, off, M)
nd['pos_desc'] = ndcg_pf(pos.astype(float), target, off, M)
nd['last_first_then_pos'] = ndcg_pf(np.where(rkend == 0, 1e6, -pos.astype(float)), target, off, M)
nd['last2_first'] = ndcg_pf(np.where(rkend <= 1, 1e6 - rkend, -pos.astype(float)), target, off, M)
# U-shape: min(rk, rkend*alpha)
for a in [1, 2, 4]:
    nd[f'U_{a}'] = ndcg_pf(-np.minimum(rk, a * rkend).astype(float) + 1e-3 * (-rk), target, off, M)
res['ndcg_rules'] = {k: float(v.mean()) for k, v in nd.items()}
# by M bin
Mb = np.digitize(M, [10, 20, 50, 100, 200])
res['ndcg_rules_byM'] = {k: [float(v[Mb == b].mean()) for b in range(6)] for k, v in nd.items()}
print(res['ndcg_rules']); print(res['ndcg_rules_byM'])
json.dump(res, open(f'{OUT}/02b_end_effect.json', 'w'))
