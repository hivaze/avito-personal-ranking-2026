"""verify_where V2: is the cold-positive deficit (W1) a property of the arms, or of the reference refN?
T = mean(delta | no positive seen in training) - mean(delta | all positives seen). Checks:
 (a) arms vs alternative references: t25N (mean of 70 4-subsets of kd05t25w192 s951-958, another teacher), and the 16-student mean.
 (b) a different recipe as a control: t25 4-pool / t25N vs refN (does any non-kd05w192 recipe show the same T?).
 (c) null splits (35 disjoint 4-vs-4 per recipe): SD of T and the val-holdout correlation of T (are the two periods independent replications?).
 (d) the mix-prediction recomputed with SE that includes the 'all seen' / 'some' classes, and the residual gap.
CPU only, read-only. Uses out/v1_perfeed.npz (v1_seed.py) + the analyst's segment files (read-only). Output out/v2_cold.txt"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *  # noqa
import itertools
HERE = ROOT / 'lab/nn_x/w12/moe/why/verify_where'; WH = ROOT / 'lab/nn_x/w12/moe/why/where'
F = dict(np.load(HERE / 'out/v1_perfeed.npz'))
sv = dict(np.load(WH / 'out/seg_val.npz')); sh = dict(np.load(WH / 'out/seg_hold.npz'))
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
S8 = tuple(range(951, 959)); S4 = S8[:4]; ARMS = 'ABC'
SEG = {'v': (sv['pos_seen'] <= 0, sv['pos_seen'] >= 1), 'h': (sh['pos_seen'] <= 0, sh['pos_seen'] >= 1)}
T = lambda d, sp: d[SEG[sp][0]].mean() - d[SEG[sp][1]].mean()
Tse = lambda d, sp: np.sqrt(se(d[SEG[sp][0]]) ** 2 + se(d[SEG[sp][1]]) ** 2)
L = []
# 16-student matched reference: per-feed mean of the two 70-subset means (approximation of 200 random 4-subsets of the 16)
ref = {'refN': (F['rNv'], F['rNh']), 't25N': (F['tNv'], F['tNh']), 'mid16': ((F['rNv'] + F['tNv']) / 2, (F['rNh'] + F['tNh']) / 2)}
L.append('== (a) arms vs alternative references: all | cold | warm | T (per-feed SE) ==')
for rk, (rv, rh) in ref.items():
    for k in ARMS:
        row = []
        for sp, x, r in (('v', F[f'{k}v'], rv), ('h', F[f'{k}h'], rh)):
            d = x - r; row.append(f'{sp}: all {fmt(d)} cold {fmt(d[SEG[sp][0]])} warm {fmt(d[SEG[sp][1]])} T {T(d, sp):+.5f}+-{Tse(d, sp):.5f}')
        L.append(f'  {k} - {rk:5s} ' + ' || '.join(row))
    for sp, rr in (('v', rv), ('h', rh)):
        d = np.mean([F[f'{k}{sp}'] for k in ARMS], 0) - rr
        L.append(f'  mean(A,B,C) - {rk} {sp}: all {fmt(d)} T {T(d, sp):+.5f}+-{Tse(d, sp):.5f}')
L.append('== (b) control recipe kd05t25w192 (same arch, other teacher) vs refN ==')
for lab, xv, xh in (('t25 4-pool s951-954', F['t4v'], F['t4h']), ('t25 4-pool s955-958', F['t4bv'], F['t4bh']), ('t25N (70-subset mean)', F['tNv'], F['tNh'])):
    row = []
    for sp, x, r in (('v', xv, F['rNv']), ('h', xh, F['rNh'])):
        d = x - r; row.append(f'{sp}: all {fmt(d)} cold {fmt(d[SEG[sp][0]])} warm {fmt(d[SEG[sp][1]])} T {T(d, sp):+.5f}+-{Tse(d, sp):.5f}')
    L.append(f'  {lab:22s} - refN ' + ' || '.join(row))
# (c) null splits for both recipes: need per-subset pools -> recompute from member z
MO = ROOT / 'lab/nn_x/w12/moe/out/preds'
Z = {}
for arm in ('kd05w192', 'kd05t25w192'):
    for s in S8:
        v = f'w5_e10_{arm}_s{s}'; Z[v] = (m.z('val', np.load(W / f'nn_{v}_trainonly_val.npy').astype(np.float32)), m.z('hold', np.load(W / f'holdout/nn_{v}_holdout.npy').astype(np.float32)))
L.append('== (c) same-recipe null: 35 disjoint 4-vs-4 splits per recipe; T in each split; per-feed z of the overall delta ==')
for arm in ('kd05w192', 'kd05t25w192'):
    pc = {}
    def pool(s):
        if s not in pc: vs = [f'w5_e10_{arm}_{"s"}{x}' for x in s]; pc[s] = (ev(np.mean([Z[v][0] for v in vs], 0)), eh(np.mean([Z[v][1] for v in vs], 0)))
        return pc[s]
    tv, th, zv, zh, dv_, dh_ = [], [], [], [], [], []
    for s in itertools.combinations(S8, 4):
        if 951 not in s: continue
        o = tuple(x for x in S8 if x not in s); a, b = pool(s), pool(o)
        d0, d1 = a[0] - b[0], a[1] - b[1]
        tv.append(T(d0, 'v')); th.append(T(d1, 'h')); zv.append(d0.mean() / se(d0)); zh.append(d1.mean() / se(d1)); dv_.append(d0.mean()); dh_.append(d1.mean())
    tv, th = np.array(tv), np.array(th)
    L.append(f'  {arm}: SD of T val {tv.std(ddof=1):.5f} hold {th.std(ddof=1):.5f}; corr(T_val, T_hold) over 35 splits {np.corrcoef(tv, th)[0, 1]:+.2f}; '
             f'z sd of overall delta val {np.std(zv, ddof=1):.2f} hold {np.std(zh, ddof=1):.2f}; corr(val delta, hold delta) {np.corrcoef(dv_, dh_)[0, 1]:+.2f}')
    # refN-like: how far is the 70-subset mean's T from each 4-pool's T (the reference's own noise in T)
    allT = [(T(pool(s)[0] - F['rNv'], 'v'), T(pool(s)[1] - F['rNh'], 'h')) for s in itertools.combinations(S8, 4)] if arm == 'kd05w192' else None
    if allT:
        a = np.array(allT); L.append(f'    kd05w192 4-subsets minus refN: SD of T val {a[:, 0].std(ddof=1):.5f} hold {a[:, 1].std(ddof=1):.5f} (spread of a 4-pool around refN; arm T values compare to this)')
# (d) mix prediction with the residual gap
L.append('== (d) mix prediction (class = positives seen all/some/none), arm = mean(A,B,C) - refN ==')
dv = np.mean([F[f'{k}v'] for k in ARMS], 0) - F['rNv']; dh = np.mean([F[f'{k}h'] for k in ARMS], 0) - F['rNh']
cls = lambda s: np.where(s['pos_seen'] >= 1, 0, np.where(s['pos_seen'] > 0, 1, 2))
cv, ch = cls(sv), cls(sh)
for sub, sel in (('all val', np.ones(len(dv), bool)), ('val M>=50', sv['M'] >= 50)):
    pv = np.bincount(cv[sel], minlength=3) / sel.sum(); hd = np.array([dh[ch == i].mean() for i in range(3)]); hs = np.array([se(dh[ch == i]) for i in range(3)])
    pred = (pv * hd).sum(); pse = np.sqrt(((pv * hs) ** 2).sum()); obs = dv[sel]
    gap_raw = obs.mean() - dh.mean(); gap_raw_se = np.sqrt(se(obs) ** 2 + se(dh) ** 2)
    gap_res = obs.mean() - pred; gap_res_se = np.sqrt(se(obs) ** 2 + pse ** 2)
    L.append(f'  {sub}: observed {fmt(obs)}; holdout {fmt(dh)}; raw gap {gap_raw:+.5f}+-{gap_raw_se:.5f} (z {gap_raw / gap_raw_se:+.2f}); '
             f'mix-predicted {pred:+.5f}+-{pse:.5f}; residual gap {gap_res:+.5f}+-{gap_res_se:.5f} (z {gap_res / gap_res_se:+.2f}); share of raw gap explained by mix {(pred - dh.mean()) / gap_raw:.0%}')
    # with the screen's ref4 instead
    dv4 = np.mean([F[f'{k}v'] for k in ARMS], 0) - F['r4v']; dh4 = np.mean([F[f'{k}h'] for k in ARMS], 0) - F['r4h']
    hd4 = np.array([dh4[ch == i].mean() for i in range(3)]); pred4 = (pv * hd4).sum()
    L.append(f'      same vs ref4: observed {fmt(dv4[sel])}; holdout {fmt(dh4)}; mix-predicted {pred4:+.5f}; residual {dv4[sel].mean() - pred4:+.5f}')
txt = '\n'.join(L); print(txt); open(HERE / 'out/v2_cold.txt', 'w').write(txt + '\n')
