"""why/where shared loaders: per-feed NDCG of 4-pools (member-equal mean of per-feed rank-z, as arm_read8_moe.py) on val and holdout.
CPU, read-only. Per-feed arrays cached in out/perfeed.npz."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *  # noqa: F401,F403  (Meta, ndcg_fast, feed_rank_z, se, fmt, paths, ROOT)
import itertools

MOE_OUT = ROOT / 'lab/nn_x/w12/moe/out'
HERE = ROOT / 'lab/nn_x/w12/moe/why/where'
_po = paths


def paths2(v, suf=''):
    if (MOE_OUT / 'preds' / f'nn_{v}{suf}_trainonly_val.npy').exists():
        Wm = MOE_OUT / 'preds'
        return Wm / f'nn_{v}{suf}_trainonly_val.npy', Wm / f'holdout/nn_{v}{suf}_holdout.npy', Wm / f'nn_{v}{suf}_trainonly_test.npy'
    return _po(v, suf)


ARMS = {'A': 'kd05w192moeffn', 'B': 'kd05w192moevo', 'C': 'kd05w192b3ff1'}
S4 = (951, 952, 953, 954); S8 = tuple(range(951, 959))
nm = lambda a, s: f'w5_e10_{a}_s{s}'


def perfeed(m, force=False):
    """dict: key -> (val per-feed NDCG, holdout per-feed NDCG) for pools: ref4 (kd05w192 951-954), ref4b (955-958),
    refN (mean over the 70 4-subsets of the 8 kd05w192 seeds, per feed), A/B/C (4-pools 951-954), plus members."""
    cf = HERE / 'out/perfeed.npz'
    if cf.exists() and not force:
        d = np.load(cf); return {k: d[k] for k in d.files}
    Z = {}
    for v in [nm('kd05w192', s) for s in S8] + [nm(a, s) for a in ARMS.values() for s in S4]:
        pv, ph, _ = paths2(v)
        Z[v] = (m.z('val', np.load(pv).astype(np.float32)), m.z('hold', np.load(ph).astype(np.float32)))
    ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
    pool = lambda vs: (ev(np.mean([Z[v][0] for v in vs], 0)), eh(np.mean([Z[v][1] for v in vs], 0)))
    out = {}
    for k, vs in (('ref4', [nm('kd05w192', s) for s in S4]), ('ref4b', [nm('kd05w192', s) for s in S8[4:]])):
        out[k + '_v'], out[k + '_h'] = pool(vs)
    for k, a in ARMS.items():
        out[k + '_v'], out[k + '_h'] = pool([nm(a, s) for s in S4])
    rv = np.zeros(len(m.Mv)); rh = np.zeros(len(m.Mh)); subs = list(itertools.combinations(S8, 4))
    for sub in subs:
        a, b = pool([nm('kd05w192', s) for s in sub]); rv += a / len(subs); rh += b / len(subs)
    out['refN_v'], out['refN_h'] = rv, rh
    for v in Z:  # members
        out['m:' + v + '_v'] = ev(Z[v][0]); out['m:' + v + '_h'] = eh(Z[v][1])
    np.savez(cf, **out)
    return out
