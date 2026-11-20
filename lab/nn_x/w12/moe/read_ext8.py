"""EXTENSION RULE read (RULE_predeclared.txt, 2026-11-19T15:15:19Z). CPU only, read-only.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python lab/nn_x/w12/moe/read_ext8.py > lab/nn_x/w12/moe/logs/read_ext8.txt
PRIMARY: @last checkpoints, arm 8-pool s951-958 vs reference 8-pool kd05w192 s951-958, train-only pools (member-equal per-feed rank-z),
paired per feed on val and holdout. BUILD iff val delta >= +0.0003 AND holdout delta >= -0.0003; otherwise CLOSE.
SECONDARY (no decision): selected checkpoints, seed halves (951-954 / 955-958 against the same reference half), val M<50 / M>=50, member means."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
MO = ROOT / 'lab/nn_x/w12/moe/out/preds'
def P(v, suf):
    if (MO / f'nn_{v}{suf}_trainonly_val.npy').exists(): return MO / f'nn_{v}{suf}_trainonly_val.npy', MO / f'holdout/nn_{v}{suf}_holdout.npy'
    a, b, _ = paths(v, suf); return a, b
m = Meta(); C = {}
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
def Z(v, suf):
    if (v, suf) not in C:
        pv, ph = P(v, suf); C[(v, suf)] = (m.z('val', np.load(pv).astype(np.float32)), m.z('hold', np.load(ph).astype(np.float32)))
    return C[(v, suf)]
def pool(vs, suf):
    zs = [Z(v, suf) for v in vs]; return ev(np.mean([z[0] for z in zs], 0)), eh(np.mean([z[1] for z in zs], 0))
def member(vs, suf):
    zs = [Z(v, suf) for v in vs]; return np.mean([ev(z[0]) for z in zs], 0), np.mean([eh(z[1]) for z in zs], 0)
n = lambda a, s: f'w5_e10_{a}_s{s}'
ALL, H1, H2 = list(range(951, 959)), list(range(951, 955)), list(range(955, 959))
big = m.Mv >= 50
out = {}
for lab, a in (('A kd05w192moeffn', 'kd05w192moeffn'), ('C kd05w192b3ff1', 'kd05w192b3ff1')):
    print(f'== {lab} vs kd05w192 (train-only pools, paired per feed)')
    for suf, tag in (('@last', '@last [PRIMARY]'), ('', 'selected')):
        av, ah = pool([n(a, s) for s in ALL], suf); rv, rh = pool([n('kd05w192', s) for s in ALL], suf)
        dv, dh = av - rv, ah - rh
        mav, mah = member([n(a, s) for s in ALL], suf); mrv, mrh = member([n('kd05w192', s) for s in ALL], suf)
        print(f'  {tag:16s} 8-pool val {av.mean():.5f} vs {rv.mean():.5f}: {dv.mean():+.5f} +- {se(dv):.5f} (z {dv.mean()/se(dv):+.2f}) | '
              f'holdout {ah.mean():.5f} vs {rh.mean():.5f}: {dh.mean():+.5f} +- {se(dh):.5f} (z {dh.mean()/se(dh):+.2f})')
        print(f'  {"":16s} val M>=50 {dv[big].mean():+.5f} +- {se(dv[big]):.5f} | M<50 {dv[~big].mean():+.5f} +- {se(dv[~big]):.5f} | '
              f'member mean val {(mav-mrv).mean():+.5f} holdout {(mah-mrh).mean():+.5f}')
        if suf == '@last': out[lab] = (dv.mean(), dh.mean())
        for hn, H in (('s951-954', H1), ('s955-958', H2)):
            hv, hh = pool([n(a, s) for s in H], suf); qv, qh = pool([n('kd05w192', s) for s in H], suf)
            print(f'  {"":16s} half {hn}: val {(hv-qv).mean():+.5f} +- {se(hv-qv):.5f} | holdout {(hh-qh).mean():+.5f} +- {se(hh-qh):.5f}')
    dv, dh = out[lab]; ok = dv >= 0.0003 and dh >= -0.0003
    print(f'  RULE: val {dv:+.5f} (bar >= +0.0003), holdout {dh:+.5f} (bar >= -0.0003) -> {"BUILD an all-data try file" if ok else "CLOSE"}\n')
