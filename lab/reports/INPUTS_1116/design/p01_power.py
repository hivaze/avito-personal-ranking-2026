"""DESIGN p01: noise of with/without-input contrasts from STORED members (CPU only, read-only).
Outputs out/p01_power.txt. For k members per arm it gives, on val (plain and private-mix weights) and on the holdout:
  (a) the null SE of a member-mean contrast from the per-feed seed variance: SE_k = sqrt(2*mean_f(s_f^2)/(k*n));
  (b) empirical A/A: random disjoint k-vs-k splits of one family: sd of the member-mean difference and of the pooled-z difference,
      next to the mean paired per-feed SE of the same differences (so the nominal SE is checked against the realised spread);
  (c) seed pairing: per-feed SE of (arm A seed s) - (arm B seed s) against (arm A seed s) - (arm B seed s') for stored arm pairs;
  (d) fine-tune level: paired SE of OOF val of two fine-tune variants of the same checkpoints.
"""
import os, sys, itertools
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition'); sys.path.insert(0, '<repo>/src')
import numpy as np
import pc_common as pc
import pmix
from pathlib import Path
OUT = Path('<repo>/reports/INPUTS_1116/design/out'); F = open(OUT/'p01_power.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); F.write(s + '\n'); F.flush()
m = pc.Meta(); rng = np.random.default_rng(1116)
W = pmix.weights(); wpriv = np.asarray(W['private'], np.float64); wpub = np.asarray(W['public'], np.float64)
fam = {
 'HL base d128 (w5_e10_base)': [f'w5_e10_base_s{s}' for s in range(951, 957)],
 'HL pop1 d128 (abl_e10_pop1)': [f'abl_e10_pop1_s{s}' for s in range(951, 959)],
 'HL xdrop_pop d128': [f'abl_e10_xdrop_pop_s{s}' for s in range(951, 955)],
 'HL pop2 d128': [f'abl_e10_pop2_s{s}' for s in range(951, 955)],
 'KD w192 (mean11)': [f'w5_e10_kd05w192_s{s}' for s in range(951, 959)],
 'KD t25w192 (mean25)': [f'w5_e10_kd05t25w192_s{s}' for s in range(951, 959)],
}
def load(v, suf=''):
    pv, ph, _ = pc.paths(v, suf)
    zv = m.z('val', np.load(pv)); zh = m.z('hold', np.load(ph))
    return zv, zh
Z = {}; N = {}
for g, vs in fam.items():
    for v in vs:
        zv, zh = load(v)
        Z[v] = dict(val=zv, hold=zh)
        N[v] = dict(val=pc.ndcg_fast(zv, m.y['val'], m.off['val'], m.M['val']), hold=pc.ndcg_fast(zh, m.y['hold'], m.off['hold'], m.M['hold']))
    P(f'{g}: n={len(vs)} member mean val {np.mean([N[v]["val"].mean() for v in vs]):.5f} hold {np.mean([N[v]["hold"].mean() for v in vs]):.5f}')
fam['KD wide 16 (both teachers)'] = fam['KD w192 (mean11)'] + fam['KD t25w192 (mean25)']
def wse(d, w=None):
    if w is None: return d.mean(), d.std() / np.sqrt(len(d))
    mu, s = pmix.wmean_se(d, w); return mu, s
def pool_ndcg(vs, sp):
    z = np.mean([Z[v][sp] for v in vs], axis=0).astype(np.float32)
    return pc.ndcg_fast(z, m.y[sp], m.off[sp], m.M[sp])
P('\n== (a) null SE of a k-vs-k MEMBER-MEAN contrast from per-feed seed variance: sqrt(2*mean(s_f^2)/(k*n)) ==')
P('family | split | k=4 | k=6 | k=8 | k=12 | k=16   (val-priv = private-mix weights, effective n lower)')
for g, vs in fam.items():
    for sp, w in (('val', None), ('val', wpriv), ('val', wpub), ('hold', None)):
        A = np.stack([N[v][sp] for v in vs]); s2 = A.var(axis=0, ddof=1)
        if w is None: ms2 = s2.mean(); n = A.shape[1]
        else: ms2 = np.average(s2 * w, weights=w) / 1.0; ms2 = np.sum(w * w * s2) / np.sum(w) ** 2 * len(w); n = len(w)
        tag = sp if w is None else ('val-priv' if w is wpriv else 'val-pub')
        P(f'{g} | {tag} | ' + ' | '.join(f'{np.sqrt(2 * ms2 / (k * n)):.5f}' for k in (4, 6, 8, 12, 16)))
P('\n== (b) empirical A/A: random disjoint k-vs-k splits; sd of difference vs mean nominal paired SE ==')
def aa(vs, k, sp, w=None, nrep=200):
    dm, dp, sem, sep = [], [], [], []
    seen = set()
    for _ in range(nrep * 5):
        perm = rng.permutation(len(vs)); a = tuple(sorted(perm[:k])); b = tuple(sorted(perm[k:2 * k]))
        key = (min(a, b), max(a, b))
        if key in seen: continue
        seen.add(key)
        va = [vs[i] for i in a]; vb = [vs[i] for i in b]
        d1 = np.mean([N[v][sp] for v in va], axis=0) - np.mean([N[v][sp] for v in vb], axis=0)
        d2 = pool_ndcg(va, sp) - pool_ndcg(vb, sp)
        mu1, s1 = wse(d1, w); mu2, s2 = wse(d2, w)
        dm.append(mu1); sem.append(s1); dp.append(mu2); sep.append(s2)
        if len(seen) >= nrep: break
    return len(seen), np.std(dm), np.mean(sem), np.std(dp), np.mean(sep)
P('family | k | split | n splits | member-mean: A/A sd, nominal SE | pool: A/A sd, nominal SE')
for g, k in (('HL base d128 (w5_e10_base)', 3), ('HL pop1 d128 (abl_e10_pop1)', 4), ('KD w192 (mean11)', 4), ('KD t25w192 (mean25)', 4), ('KD wide 16 (both teachers)', 8), ('KD wide 16 (both teachers)', 4)):
    for sp, w, tag in (('val', None, 'val'), ('val', wpriv, 'val-priv'), ('hold', None, 'hold')):
        n, a1, b1, a2, b2 = aa(fam[g], k, sp, w, nrep=120 if k == 8 else 60)
        P(f'{g} | {k} | {tag} | {n} | {a1:.5f}, {b1:.5f} | {a2:.5f}, {b2:.5f}')
P('\n== (c) seed pairing: per-feed SE of member(A,s) - member(B,s) [same seed] vs member(A,s) - member(B,s2) [other seed] ==')
pairs = [('abl_e10_pop1', 'abl_e10_xdrop_pop', range(951, 955)), ('abl_e10_pop2', 'abl_e10_xdrop_pop', range(951, 955)),
         ('w5_e10_base', 'abl_e10_xdrop_pop', range(951, 955)), ('w5_e10_kd05w192', 'w5_e10_kd05t25w192', range(951, 959))]
for a, b, seeds in pairs:
    for sp in ('val', 'hold'):
        same = [np.std(N[f'{a}_s{s}'][sp] - N[f'{b}_s{s}'][sp]) / np.sqrt(len(N[f'{a}_s{s}'][sp])) for s in seeds]
        oth = [np.std(N[f'{a}_s{s}'][sp] - N[f'{b}_s{t}'][sp]) / np.sqrt(len(N[f'{a}_s{s}'][sp])) for s in seeds for t in seeds if s != t]
        ca = [np.corrcoef(N[f'{a}_s{s}'][sp], N[f'{b}_s{s}'][sp])[0, 1] for s in seeds]
        co = [np.corrcoef(N[f'{a}_s{s}'][sp], N[f'{b}_s{t}'][sp])[0, 1] for s in seeds for t in seeds if s != t]
        P(f'{a} - {b} | {sp} | single-pair SE same seed {np.mean(same):.5f}, other seed {np.mean(oth):.5f} | per-feed corr same {np.mean(ca):.4f} other {np.mean(co):.4f}')
P('\n== (c2) stored arm contrasts with k members per arm (member mean and pool; val, val-priv, hold) ==')
def contrast(va, vb, lab):
    for sp, w, tag in (('val', None, 'val'), ('val', wpriv, 'val-priv'), ('hold', None, 'hold')):
        d1 = np.mean([N[v][sp] for v in va], axis=0) - np.mean([N[v][sp] for v in vb], axis=0)
        d2 = pool_ndcg(va, sp) - pool_ndcg(vb, sp)
        mu1, s1 = wse(d1, w); mu2, s2 = wse(d2, w)
        P(f'{lab} | {tag} | member {mu1:+.5f} +- {s1:.5f} | pool {mu2:+.5f} +- {s2:.5f}')
contrast(fam['HL pop1 d128 (abl_e10_pop1)'][:4], fam['HL xdrop_pop d128'], 'pop1 - xdrop_pop (ONE added column xp_imp_c50), 4v4 HL')
contrast(fam['HL pop2 d128'], fam['HL xdrop_pop d128'], 'pop2 - xdrop_pop (ONE added column xp_ctr_c), 4v4 HL')
contrast(fam['HL base d128 (w5_e10_base)'][:4], fam['HL xdrop_pop d128'], 'base - xdrop_pop (8 popularity inputs), 4v4 HL')
contrast(fam['KD w192 (mean11)'], fam['KD t25w192 (mean25)'], 'KD w192 - t25w192 (teacher identity; A/A-like), 8v8')
contrast(fam['KD wide 16 (both teachers)'][:4] + fam['KD wide 16 (both teachers)'][8:12], fam['KD wide 16 (both teachers)'][4:8] + fam['KD wide 16 (both teachers)'][12:], 'KD wide seeds 951-954 - seeds 955-958 (both teachers), 8v8 A/A')
P('\n== (d) fine-tune level: OOF val of the same checkpoints under two fine-tune variants (paired) ==')
AB = pc.AB; Wd = pc.W
def oof(v, tag):
    d = Wd if v.startswith('w5_') else AB
    p = d/f'nn_{v}_{tag}_val_oof.npy'
    return np.load(p) if p.exists() else None
for vs, t1, t2, lab in ((fam['HL pop1 d128 (abl_e10_pop1)'], 'n2cpv3s3', 'n2cpv3', 'pop1 HL: n2cpv3s3 - n2cpv3'), (fam['HL pop1 d128 (abl_e10_pop1)'], 'n2cpv3np', 'n2cpv3', 'pop1 HL: n2cpv3np - n2cpv3')):
    ds = []; A1 = []; A2 = []
    for v in vs:
        a, b = oof(v, t1), oof(v, t2)
        if a is None or b is None: continue
        na = pc.ndcg_fast(m.z('val', a), m.y['val'], m.off['val'], m.M['val']); nb_ = pc.ndcg_fast(m.z('val', b), m.y['val'], m.off['val'], m.M['val'])
        ds.append(na - nb_); A1.append(m.z('val', a)); A2.append(m.z('val', b))
    if not ds: P(lab, 'no files'); continue
    one = np.mean([d.std() / np.sqrt(len(d)) for d in ds]); dm = np.mean(ds, axis=0)
    pa = pc.ndcg_fast(np.mean(A1, axis=0).astype(np.float32), m.y['val'], m.off['val'], m.M['val']); pb = pc.ndcg_fast(np.mean(A2, axis=0).astype(np.float32), m.y['val'], m.off['val'], m.M['val'])
    dp = pa - pb
    P(f'{lab} | n={len(ds)} | single member paired SE {one:.5f} | member-mean {dm.mean():+.5f} +- {dm.std() / np.sqrt(len(dm)):.5f} | pool {dp.mean():+.5f} +- {dp.std() / np.sqrt(len(dp)):.5f} | pool val-priv {pmix.wmean_se(dp, wpriv)[0]:+.5f} +- {pmix.wmean_se(dp, wpriv)[1]:.5f}')
# FT vs TO for the 16 wide students (how much the FT stage moves a member, and the paired SE of an FT-level contrast between seeds halves)
vs = fam['KD wide 16 (both teachers)']
ft = {v: oof(v, 'n2cpv3s3') for v in vs}
nft = {v: pc.ndcg_fast(m.z('val', ft[v]), m.y['val'], m.off['val'], m.M['val']) for v in vs}
d = np.mean([nft[v] - N[v]['val'] for v in vs], axis=0)
P(f'16 wide students: FT OOF - TO, member mean {d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f}; FT member mean level {np.mean([nft[v].mean() for v in vs]):.5f}')
A = np.stack([nft[v] for v in vs]); s2 = A.var(axis=0, ddof=1).mean()
P('FT OOF seed variance -> null SE of an UNPAIRED k-vs-k member-mean FT contrast: ' + ' | '.join(f'k={k}: {np.sqrt(2 * s2 / (k * A.shape[1])):.5f}' for k in (8, 16)))
F.close()
