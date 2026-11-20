"""verify_noise V1 (2026-11-19, adversarial check of why/noise N1-N3). Independent recompute from RAW prediction files (no use of the
noise cache), CPU only, read-only.
 (a) every member: val / holdout NDCG of the stored preds vs the json (selected checkpoint) values -> catches @last vs selected mix-ups;
 (b) pools Ra, Rb, A, B, C, all 70 reference 4-subsets; Ra rank, Ra-Rb rank among the 35 even splits; arms vs Ra / Rb / Rbar70;
 (c) z-cache agreement with the noise analyst's cache (spot check);
 (d) ndcg_fast vs src/metrics.ndcg_per_feed on one pool;
 (e) member-level two-way table (recipe x seed) for the 8-seed recipes: is there a shared seed effect (data order) across recipes?
"""
import os, sys, json, itertools, time
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import Meta, ndcg_fast, ROOT  # noqa
import numpy as np
from pathlib import Path

OUT = ROOT / 'lab/nn_x/w12/moe/why/verify_noise'
SCR = Path('/tmp/scratchpad/why/verify_noise'); SCR.mkdir(parents=True, exist_ok=True)
MOE = ROOT / 'lab/nn_x/w12/moe/out'
W5 = ROOT / 'nn_x/w5'
nm = lambda a, s: f'w5_e10_{a}_s{s}'


def P_(v, suf=''):
    if (MOE / 'preds' / f'nn_{v}{suf}_trainonly_val.npy').exists():
        return MOE / 'preds' / f'nn_{v}{suf}_trainonly_val.npy', MOE / 'preds/holdout' / f'nn_{v}{suf}_holdout.npy', MOE / 'logs' / f'{v}.json'
    return W5 / 'preds' / f'nn_{v}{suf}_trainonly_val.npy', W5 / 'preds/holdout' / f'nn_{v}{suf}_holdout.npy', W5 / 'logs' / f'{v}.json'


m = Meta()
out = []
P = lambda *a: (print(*a, flush=True), out.append(' '.join(str(x) for x in a)))
se = lambda d: float(np.std(d, ddof=1) / np.sqrt(len(d)))
fmt = lambda d: f'{np.mean(d):+.5f}+-{se(d):.5f}'
Zc = {}


def Z(v):
    if v not in Zc:
        f = SCR / f'z_{v}.npz'
        if f.exists():
            d = np.load(f); Zc[v] = (d['v'], d['h'])
        else:
            pv, ph, _ = P_(v)
            Zc[v] = (m.z('val', np.load(pv).astype(np.float32)).astype(np.float32), m.z('hold', np.load(ph).astype(np.float32)).astype(np.float32))
            np.savez(f, v=Zc[v][0], h=Zc[v][1])
    return Zc[v]


ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv)
eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
pool = lambda vs: (ev(np.mean([Z(v)[0] for v in vs], 0)), eh(np.mean([Z(v)[1] for v in vs], 0)))

REF = [nm('kd05w192', s) for s in range(951, 959)]
ARMS = {'A': 'kd05w192moeffn', 'B': 'kd05w192moevo', 'C': 'kd05w192b3ff1'}
AM = {k: [nm(a, s) for s in range(951, 955)] for k, a in ARMS.items()}

# (a) stored preds vs json
P('== (a) member NDCG of the stored (selected) preds vs the json record (val_ndcg = selected; holdout from history at best_step)')
t = time.time()
bad = 0
for v in REF + sum(AM.values(), []):
    pv, ph, pj = P_(v); j = json.load(open(pj))
    raw_v = np.load(pv).astype(np.float32); raw_h = np.load(ph).astype(np.float32)
    nv, nh = ev(raw_v).mean(), eh(raw_h).mean()
    hist = {h['step']: h for h in j['history']}
    hb = hist.get(j['best_step'], {}).get('holdout_ndcg', float('nan'))
    last = max(hist); hl = hist[last].get('holdout_ndcg')
    pvl, phl, _ = P_(v, '@last') if False else (None, None, None)
    rl_h = eh(np.load(str(ph).replace('_holdout.npy', '@last_holdout.npy')).astype(np.float32)).mean()
    flag = '' if abs(nv - j['val_ndcg']) < 2e-5 and abs(nh - hb) < 2e-5 else '  <-- MISMATCH'
    bad += bool(flag)
    P(f'  {v:34s} best_step {j["best_step"]:6d} | val stored {nv:.5f} json {j["val_ndcg"]:.5f} | hold stored {nh:.5f} hist@best {hb:.5f} | hold @last file {rl_h:.5f} hist@{last} {hl:.5f}{flag}')
P(f'  mismatches: {bad}  ({time.time() - t:.0f}s)')

# (d) ndcg_fast vs the project metric on one pool
sys.path.insert(0, str(ROOT / 'src'))
from metrics import ndcg_per_feed  # noqa
zra = np.mean([Z(v)[0] for v in REF[:4]], 0)
a1 = ev(zra); a2 = np.asarray(ndcg_per_feed(zra, m.yv, m.ov, m.Mv)) if True else None
try:
    P(f'== (d) ndcg_fast vs src/metrics.ndcg_per_feed on Ra val: {a1.mean():.6f} vs {np.mean(a2):.6f} (max |diff| per feed {np.max(np.abs(a1 - a2)):.2e})')
except Exception as e:  # noqa
    P(f'== (d) metrics comparison failed: {e}')

# (c) cache agreement
NC = Path('/tmp/scratchpad/why/noise_cache')
mx = 0
for v in REF + sum(AM.values(), []):
    f = NC / f'z_{v}.npz'
    if f.exists():
        d = np.load(f); mx = max(mx, float(np.max(np.abs(d['v'] - Z(v)[0]))), float(np.max(np.abs(d['h'] - Z(v)[1]))))
P(f'== (c) max |z diff| own recompute vs the noise cache over 20 members: {mx:.2e}')

# (b) pools
P('')
P('== (b) reference 4-pools and arms (independent recompute)')
subs = list(itertools.combinations(range(8), 4))
pv, ph = {}, {}
for c in subs:
    pv[c], ph[c] = pool([REF[i] for i in c])
mv = np.array([pv[c].mean() for c in subs]); mh = np.array([ph[c].mean() for c in subs])
cA, cB = (0, 1, 2, 3), (4, 5, 6, 7)
rk = lambda arr, x: int((arr < x).sum()) + 1
P(f'  val  : 70-subset mean {mv.mean():.5f} sd {mv.std(ddof=1):.5f}; Ra {pv[cA].mean():.5f} rank {rk(mv, pv[cA].mean())}/70; Rb {pv[cB].mean():.5f} rank {rk(mv, pv[cB].mean())}/70')
P(f'  hold : 70-subset mean {mh.mean():.5f} sd {mh.std(ddof=1):.5f}; Ra {ph[cA].mean():.5f} rank {rk(mh, ph[cA].mean())}/70; Rb {ph[cB].mean():.5f} rank {rk(mh, ph[cB].mean())}/70')
D35v = np.array([pv[c].mean() - pv[tuple(sorted(set(range(8)) - set(c)))].mean() for c in subs if 0 in c])
P(f'  Ra-Rb val {fmt(pv[cA] - pv[cB])} holdout {fmt(ph[cA] - ph[cB])}; |Ra-Rb| rank among 35 even splits: {int((np.abs(D35v) >= abs(D35v[0])).sum())}/35 (1 = largest)')
Rbv = np.mean([pv[c] for c in subs], 0); Rbh = np.mean([ph[c] for c in subs], 0)
P(f'  Ra - Rbar70 val {fmt(pv[cA] - Rbv)} holdout {fmt(ph[cA] - Rbh)}')
AP = {}
for k in 'ABC':
    av, ah = pool(AM[k]); AP[k] = (av, ah)
    P(f'  {k}: vs Ra val {fmt(av - pv[cA])} hold {fmt(ah - ph[cA])} | vs Rb val {fmt(av - pv[cB])} hold {fmt(ah - ph[cB])} | vs Rbar70 val {fmt(av - Rbv)} hold {fmt(ah - Rbh)}')

# (e) seed effect shared across recipes? member val/holdout NDCG table for 8-seed recipes
P('')
P('== (e) member NDCG by seed for 8-seed w192 recipes: is there a seed (init + data order) effect shared across recipes?')
REC = ['kd05w192', 'kd05t25w192', 'kd08w192', 'kwwd0', 'kwwd01', 'kwwd05', 'kwaux0', 'kd05w192ps']
TV = np.zeros((len(REC), 8)); TH = np.zeros((len(REC), 8))
for i, r in enumerate(REC):
    for j, s in enumerate(range(951, 959)):
        zv, zh = Z(nm(r, s)); TV[i, j] = ev(zv).mean(); TH[i, j] = eh(zh).mean()
for nmx, T in (('val', TV), ('holdout', TH)):
    R = T - T.mean(1, keepdims=True)                   # remove recipe means
    seed_eff = R.mean(0)                               # shared seed effect estimate
    resid = R - seed_eff[None, :]
    n_r, n_s = T.shape
    ms_seed = n_r * np.sum(seed_eff ** 2) / (n_s - 1); ms_res = np.sum(resid ** 2) / ((n_r - 1) * (n_s - 1))
    from scipy.stats import f as Fd
    Fv = ms_seed / ms_res; pF = 1 - Fd.cdf(Fv, n_s - 1, (n_r - 1) * (n_s - 1))
    # leave-ref-out: seed effect from the 7 non-reference recipes, compared with the reference's own seed pattern
    se_other = R[1:].mean(0)
    P(f'  [{nmx}] seed effects (mean over recipes, x1e4): ' + ' '.join(f'{s}:{x * 1e4:+.1f}' for s, x in zip(range(951, 959), seed_eff))
      + f' | F(seed) {Fv:.2f} p {pF:.3f} | a-half minus b-half shared seed effect {seed_eff[:4].mean() - seed_eff[4:].mean():+.5f}'
      + f' | 7 non-ref recipes: a-b {se_other[:4].mean() - se_other[4:].mean():+.5f}; corr(ref seed pattern, other recipes) {np.corrcoef(R[0], se_other)[0, 1]:+.2f}')
json.dump(dict(TV=TV.tolist(), TH=TH.tolist(), rec=REC), open(OUT / 'v1_member_table.json', 'w'))
open(OUT / 'v1_recompute.txt', 'w').write('\n'.join(out) + '\n')
