"""decide7 - PRE-DECLARED in CURRENT_STATUS 19:3x UTC 2026-11-12 (before any E result): pools with the TRUNC E members.
E = w5_e10_<arm>_s951-954 (arm from argv, default trunc), each with ft (n2cpv3) + to + holdout preds copied into preds/.
Candidates vs G3 (F10+F7x+F7x70): T1 = G3 + E (24); T2 = F10 + F7x + E (E replaces F7x70, 17).
Qualify iff vs G3 (paired): H > +1 SE (holdout, train-only pool); V >= 0 (final 0.5z(ft)+0.5z(to) val);
no val slice (M<50, M>=50, Tue, Thu) < -2 SE; pmix private-mix >= -1 SE; presub_gate PASS (run afterwards).
Pick = qualifying candidate with the highest H; none -> G3 stays."""
import sys, json; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
ARM = sys.argv[1] if len(sys.argv) > 1 else 'trunc'
P = Path('<repo>/preds'); H = P / 'holdout'
F10 = [f'v23_gru_pre2s_s{s}' for s in range(501, 507)]; F7x = [f'r6_cd_m02_r{s}' for s in range(101, 108)]
F7x70 = [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]; E = [f'w5_e10_{ARM}_s{s}' for s in range(951, 955)]
G3 = F10 + F7x + F7x70
C = {'G3': G3, 'T1=G3+E': G3 + E, 'T2=F10+F7x+E': F10 + F7x + E}
for k, ms in C.items():
    miss = [m for m in ms if not all(p.exists() for p in [P/f'nn_{m}_n2cpv3_val_oof.npy', P/f'nn_{m}_trainonly_val.npy', H/f'nn_{m}_holdout.npy', P/f'nn_{m}_n2cpv3_test.npy', P/f'nn_{m}_trainonly_test.npy'])]
    assert not miss, f'{k}: missing {miss}'
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT/'val/cand_target.npy')
Mh, oh = np.load(H/'holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(H/'holdout_y.npy')
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]; zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]
ev = lambda s: ndcg_per_feed(s, yv, ov, Mv); eh = lambda s: ndcg_per_feed(s, yh, oh, Mh); se = lambda d: d.std() / np.sqrt(len(d))
cache = {}
def ld(m):
    if m not in cache:
        cache[m] = (zv(np.load(P/f'nn_{m}_n2cpv3_val_oof.npy')), zv(np.load(P/f'nn_{m}_trainonly_val.npy')), zh(np.load(H/f'nn_{m}_holdout.npy')))
    return cache[m]
def pool(ms):
    ft = zv(np.mean([ld(m)[0] for m in ms], 0)); to = zv(np.mean([ld(m)[1] for m in ms], 0)); ho = zh(np.mean([ld(m)[2] for m in ms], 0))
    return ev(0.5 * ft + 0.5 * to), eh(ho)
R = {k: pool(ms) for k, ms in C.items()}
v0, h0 = R['G3']
tue = (fv['timestamp'].to_numpy() + 10800) // 86400 < 20720
SL = [('M<50', Mv < 50), ('M>=50', Mv >= 50), ('Tue', tue), ('Thu', ~tue)]
W = pmix.weights(); out = {}
for k, (v, h) in R.items():
    dv, dh = v - v0, h - h0
    zs = {n: dv[m].mean() / max(se(dv[m]), 1e-9) for n, m in SL}
    pm, pse = pmix.wmean_se(dv, W['private'])
    q = k != 'G3' and dh.mean() > se(dh) and dv.mean() >= 0 and min(zs.values()) >= -2 and pm >= -pse
    out[k] = dict(H=float(dh.mean()), q=bool(q))
    print(f'{k:16s} n={len(C[k]):2d} val {v.mean():.5f} hold {h.mean():.5f} | V {dv.mean():+.5f}+-{se(dv):.5f} H {dh.mean():+.5f}+-{se(dh):.5f} '
          f'pmix {pm:+.5f}+-{pse:.5f} | ' + ' '.join(f'{n} {z:+.1f}' for n, z in zs.items()) + (' | QUALIFIES' if q else ''))
qs = [k for k in out if out[k]['q']]
pick = max(qs, key=lambda k: out[k]['H']) if qs else None
print('PICK (pending presub_gate PASS):', pick or 'none -> G3 stays')
json.dump({'pick': pick, 'members': C.get(pick, [])}, open('/tmp/scratchpad/decide7_pick.json', 'w'))
