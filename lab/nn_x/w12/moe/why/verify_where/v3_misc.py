"""verify_where V3: (1) W3 trimmed-mean SE (trimmed-sd SE vs winsorised SE vs bootstrap); (2) the +0.5 correlation of (X - ref4) with
(ref4b - ref4) for a control X with no relation to the arms (t25 4-pool) - is +0.5 just the construction?; (3) @last vs selected
checkpoint for arms and references (R0 with @last); (4) seed-null for the lagged-holdout DiD (35 splits) and the 'luck' of ref4 in the
lagged / stale views. CPU only, read-only. Output out/v3_misc.txt"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *  # noqa
import itertools
HERE = ROOT / 'lab/nn_x/w12/moe/why/verify_where'; WH = ROOT / 'lab/nn_x/w12/moe/why/where'
F = dict(np.load(HERE / 'out/v1_perfeed.npz'))
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
S8 = tuple(range(951, 959)); S4 = S8[:4]; ARMS = {'A': 'kd05w192moeffn', 'B': 'kd05w192moevo', 'C': 'kd05w192b3ff1'}
nm = lambda a, s: f'w5_e10_{a}_s{s}'
L = []
# (1) trimmed means
L.append('== (1) 1%-trimmed mean of per-feed val deltas: SE from trimmed sd (analyst) vs winsorised sd/(1-2g) vs bootstrap (2000) ==')
rng = np.random.default_rng(0)
def trim_stats(d, g=0.01):
    n = len(d); k = max(1, n // 100); o = np.sort(d); t = o[k:-k]
    se_t = t.std(ddof=1) / np.sqrt(len(t))
    w = np.clip(d, o[k], o[-k - 1]); se_w = w.std(ddof=1) / ((1 - 2 * k / n) * np.sqrt(n))
    bs = []
    for _ in range(400):
        x = np.sort(d[rng.integers(0, n, n)]); bs.append(x[k:-k].mean())
    return t.mean(), se_t, se_w, np.std(bs, ddof=1)
for k in ARMS:
    d = F[f'{k}v'] - F['r4v']; t, a, b, c = trim_stats(d)
    L.append(f'  {k} - ref4: raw {fmt(d)} | trimmed {t:+.5f}  SE(trimmed sd) {a:.5f}  SE(winsorised) {b:.5f}  SE(bootstrap) {c:.5f}')
d = F['r4bv'] - F['r4v']; t, a, b, c = trim_stats(d)
L.append(f'  null ref4b - ref4: raw {fmt(d)} | trimmed {t:+.5f}  SE(trimmed sd) {a:.5f}  SE(winsorised) {b:.5f}  SE(bootstrap) {c:.5f}')
d = np.mean([F[f'{k}v'] for k in ARMS], 0) - F['rNv']; t, a, b, c = trim_stats(d)
L.append(f'  mean(A,B,C) - refN: raw {fmt(d)} | trimmed {t:+.5f}  SE(trimmed sd) {a:.5f}  SE(winsorised) {b:.5f}  SE(bootstrap) {c:.5f}')
# (2) correlation construction
L.append('== (2) corr(X - ref4, ref4b - ref4) on val per feed: arms vs control X that shares nothing with the arms ==')
nl = F['r4bv'] - F['r4v']
for lab, x in [(k, F[f'{k}v']) for k in ARMS] + [('t25 4-pool s951-954', F['t4v']), ('t25 4-pool s955-958', F['t4bv']), ('t25N', F['tNv'])]:
    L.append(f'  {lab:22s}: corr {np.corrcoef(x - F["r4v"], nl)[0, 1]:+.3f}')
# (3) @last
L.append('== (3) @last checkpoints: arm 4-pool (@last) minus reference pools (@last) ==')
MO = ROOT / 'lab/nn_x/w12/moe/out/preds'
def pth(v, suf):
    if (MO / f'nn_{v}{suf}_trainonly_val.npy').exists(): return MO / f'nn_{v}{suf}_trainonly_val.npy', MO / f'holdout/nn_{v}{suf}_holdout.npy'
    return W / f'nn_{v}{suf}_trainonly_val.npy', W / f'holdout/nn_{v}{suf}_holdout.npy'
ZL = {}
for v in [nm('kd05w192', s) for s in S8] + [nm(a, s) for a in ARMS.values() for s in S4]:
    pv, ph = pth(v, '@last'); ZL[v] = (m.z('val', np.load(pv).astype(np.float32)), m.z('hold', np.load(ph).astype(np.float32)))
pl_ = lambda vs: (ev(np.mean([ZL[v][0] for v in vs], 0)), eh(np.mean([ZL[v][1] for v in vs], 0)))
subs = list(itertools.combinations(S8, 4)); PS = {s: pl_([nm('kd05w192', x) for x in s]) for s in subs}
rNv = np.mean([PS[s][0] for s in subs], 0); rNh = np.mean([PS[s][1] for s in subs], 0); vals = np.array([PS[s][0].mean() for s in subs])
L.append(f'  @last ref4 val rank {int((vals > vals[subs.index(S4)]).sum()) + 1}/70 (ref4 - mean {vals[subs.index(S4)] - vals.mean():+.5f})')
for k, a in ARMS.items():
    x = pl_([nm(a, s) for s in S4])
    L.append(f'  {k}@last: vs ref4@last val {fmt(x[0] - PS[S4][0])} hold {fmt(x[1] - PS[S4][1])} | vs refN@last val {fmt(x[0] - rNv)} hold {fmt(x[1] - rNh)}')
# (4) lag DiD seed null
L.append('== (4) lagged holdout: seed-null of the DiD (priv_v3 - lag0) over the 35 disjoint 4-vs-4 reference splits ==')
LAG = WH / 'out/lag'
ZG = {}
for view in ('lag0', 'priv_v3'):
    for s in S8: ZG[(view, s)] = m.z('hold', np.load(LAG / f'pred_{view}_{nm("kd05w192", s)}.npy'))
pc = {}
def pg(view, s):
    if (view, s) not in pc: pc[(view, s)] = eh(np.mean([ZG[(view, x)] for x in s], 0))
    return pc[(view, s)]
did, zd = [], []
for s in subs:
    if 951 not in s: continue
    o = tuple(x for x in S8 if x not in s)
    d = (pg('priv_v3', s) - pg('priv_v3', o)) - (pg('lag0', s) - pg('lag0', o)); did.append(d.mean()); zd.append(d.mean() / se(d))
L.append(f'  SD of DiD over 35 splits {np.std(did, ddof=1):.5f}; z sd {np.std(zd, ddof=1):.2f}; typical per-feed SE {np.mean([abs(a / b) for a, b in zip(did, zd)]):.5f}')
ref4_l = [pg(v, S4).mean() - np.mean([pg(v, s).mean() for s in subs]) for v in ('lag0', 'priv_v3')]
L.append(f'  ref4 minus mean of 70 subsets: lag0 {ref4_l[0]:+.5f} priv_v3 {ref4_l[1]:+.5f}')
txt = '\n'.join(L); print(txt); open(HERE / 'out/v3_misc.txt', 'w').write(txt + '\n')
