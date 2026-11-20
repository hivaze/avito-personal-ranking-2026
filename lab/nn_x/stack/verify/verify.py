"""Adversarial re-evaluation of the 4 accepted stacking/weighting variants (own code; CPU only).
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 POLARS_MAX_THREADS=12 NUMBA_NUM_THREADS=12 CUDA_VISIBLE_DEVICES= python verify.py
Reads preds only; writes only into nn_x/stack/verify/."""
import os, sys, json, time
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS']: os.environ.setdefault(_k, '12')
sys.path.insert(0, '<repo>/src')
import numpy as np, numba as nb, polars as pl
from sklearn.model_selection import GroupKFold
from common import feed_meta, FLAT
from metrics import ndcg_per_feed

R = '<repo>'; P = f'{R}/preds'; V = f'{R}/nn_x/stack/verify'; C = f'{V}/cache'; os.makedirs(C, exist_ok=True)
t0 = time.time()
def log(*a): print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)

FAM = {
    'F5cd': ['v15_cd_all_e10', 'v15_cd_all_s81', 'v15_cd_all_s82', 'v15_cd_allprew_s83', 'v15_cd_allprew_s84', 'v15_cd_allprew_s85'],
    'F6': ['v13_lr1_mix1'] + [f'v14_s{s}' for s in range(92, 97)],
    'F7': [f'v17_pre2s_s{s}' for s in range(101, 108)],
    'F7x': [f'r6_cd_m02_r{s}' for s in range(101, 108)],
    'F7x70': [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)],
    'F9': [f'v22_pre2s_lr1_s{s}' for s in range(401, 407)],
    'F10': [f'v23_gru_pre2s_s{s}' for s in range(501, 507)],
    'F12': [f'v26_d192_pre2s_s{s}' for s in range(701, 708)],
    # members as frozen in hfit families.json at build time (checked below against the file list)
    'V27': ['v27_gru_m02_s511', 'v27_gru_m02_s515'],
    'V28': ['v28_lr1_m02_s411', 'v28_lr1_m02_s413', 'v28_lr1_m02_s415'],
    'V29': ['v29_d192_m02_s711', 'v29_d192_m02_s712'],
}
hf = json.load(open(f'{R}/nn_x/stack/hfit/cache/families.json'))
assert all(hf[k] == FAM[k] for k in FAM) and set(hf) == set(FAM), 'family lists differ from hfit families.json'
PF7x = ['F5cd', 'F6', 'F7x', 'F9', 'F10', 'F12']; G3 = ['F10', 'F7x', 'F7x70']; M2 = ['F7x', 'F7x70']
U45 = PF7x + ['F7x70']; P11 = ['F5cd', 'F6', 'F7', 'F7x', 'F7x70', 'F9', 'F10', 'F12', 'V27', 'V28', 'V29']


@nb.njit(parallel=True)
def fz(off, M, x):
    out = np.empty(len(x))
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m].astype(np.float64)
        mu = v.mean(); sd = v.std() + 1e-9
        for k in range(m):
            out[o + k] = (v[k] - mu) / sd
    return out


@nb.njit(parallel=True)
def feed_spearman(off, M, a, b):
    out = np.ones(len(M))
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        if m < 3: continue
        ra = np.argsort(np.argsort(a[o:o + m])).astype(np.float64)
        rb = np.argsort(np.argsort(b[o:o + m])).astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        out[f] = (ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return out


def Z(off, M, x): return fz(off.astype(np.int64), M.astype(np.int64), np.ascontiguousarray(x, dtype=np.float64))


# ---------------- meta
fv, Mv, ov = feed_meta('val'); Mv = Mv.astype(np.int64); ov = ov.astype(np.int64)
yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64)
ft_, Mt, ot = feed_meta('test'); Mt = Mt.astype(np.int64); ot = ot.astype(np.int64)
Mh, oh = np.load(f'{P}/holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
yh = np.load(f'{P}/holdout/holdout_y.npy').astype(np.float64)
uv = fv['user_id'].to_numpy().astype(np.int64)
gv = np.where(uv == 0, -1 - np.arange(len(Mv)), uv)  # cluster ids (user 0 singletons)
# holdout reconstruction (independent re-derivation of the same deterministic recipe)
tr = pl.read_parquet(f'{R}/data/flat/train/feeds.parquet', columns=['timestamp', 'M', 'user_id'])
pr = np.nan_to_num(np.load(f'{R}/eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0)
trM = tr['M'].to_numpy(); pr[trM < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((pr > 0.9) & (trM >= 50))[0], 30000, replace=False))
assert (trM[hold] == Mh).all()
uh = tr['user_id'].to_numpy()[hold].astype(np.int64)
assert (uh == np.load(f'{R}/nn_x/stack/hfit/cache/hold_user.npy')).all()
assert (uh == np.load(f'{R}/nn_x/stack/nnstack/data/hold_user.npy')).all()
gh = np.where(uh == 0, -1 - np.arange(len(Mh)), uh)
# holdout feeds are TRAIN-period: confirm no overlap in time with val
log('holdout ts max', tr['timestamp'].to_numpy()[hold].max(), 'val ts min', fv['timestamp'].to_numpy().min(),
    'holdout users also in val:', np.isin(np.unique(uh[uh != 0]), uv).mean().round(3))

# slices
fidv = np.repeat(np.arange(len(Mv)), Mv)
day = (fv['timestamp'].to_numpy() + 10800) // 86400
luo = np.nan_to_num(np.asarray(np.load(f'{R}/nn_x/r6/feat_cache/val/it_hist_all_luo.npy', mmap_mode='r'), np.float64), nan=0.0)
cold = luo == 0
npos = np.bincount(fidv, yv, len(Mv)); ncold = np.bincount(fidv, yv * cold, len(Mv))
SL = {'M<50': Mv < 50, 'M>=50': Mv >= 50, 'Tue': day < 20720, 'Thu': day >= 20720,
      'coldpos': (npos > 0) & (ncold == npos), 'warmpos': (npos > 0) & (ncold == 0), 'user0': uv == 0}

# ---------------- family z sums (cached)
def famsum(f, split, half):
    fn = f'{C}/{split}_{half}_{f}.npy'
    if os.path.exists(fn): return np.load(fn)
    if split == 'hold':
        acc = sum(Z(oh, Mh, np.nan_to_num(np.load(f'{P}/holdout/nn_{m}_holdout.npy').astype(np.float64))) for m in FAM[f])
    else:
        off, M = (ov, Mv) if split == 'val' else (ot, Mt)
        suf = {('val', 'ft'): 'n2cpv3_val_oof', ('val', 'to'): 'trainonly_val', ('test', 'ft'): 'n2cpv3_test', ('test', 'to'): 'trainonly_test'}[(split, half)]
        acc = sum(Z(off, M, np.nan_to_num(np.load(f'{P}/nn_{m}_{suf}.npy').astype(np.float64))) for m in FAM[f])
    np.save(fn, acc); return acc


FS = {}
for f in FAM:
    for split, halves in [('val', ('ft', 'to')), ('test', ('ft', 'to')), ('hold', ('to',))]:
        for h in halves: FS[(f, split, h)] = famsum(f, split, h)
    log('fam', f, 'done')
NF = {f: len(FAM[f]) for f in FAM}
META = {'val': (ov, Mv), 'test': (ot, Mt), 'hold': (oh, Mh)}


def half_pool(fams, split, h, w=None):
    """member-equal (w None) or family-weighted (w dict, family means) z-mean of one half."""
    if w is None: return sum(FS[(f, split, h)] for f in fams) / sum(NF[f] for f in fams)
    return sum(w[f] * FS[(f, split, h)] / NF[f] for f in fams)


def spec(fams, split):
    off, M = META[split]
    if split == 'hold': return half_pool(fams, 'hold', 'to')
    return 0.5 * Z(off, M, half_pool(fams, split, 'ft')) + 0.5 * Z(off, M, half_pool(fams, split, 'to'))


def nd(s, split='val'):
    if split == 'val': return ndcg_per_feed(s, yv, ov, Mv)
    return ndcg_per_feed(s, yh, oh, Mh)


def se_pair(d, groups):
    n = len(d); iid = d.std(ddof=1) / np.sqrt(n)
    _, inv = np.unique(groups, return_inverse=True); g = np.bincount(inv, d - d.mean()); G = len(g)
    cl = np.sqrt(G / (G - 1) * (g ** 2).sum()) / n
    return d.mean(), iid, cl


def report(name, new, ref, split='val'):
    a, b = (nd(new, split), nd(ref, split))
    d = a - b
    m, iid, cl = se_pair(d, gv if split == 'val' else gh); se = max(iid, cl)
    out = {'new': a.mean(), 'ref': b.mean(), 'd': m, 'se_iid': iid, 'se_cl': cl, 't': m / se}
    s = f'{name:44s} {split:4s} new {a.mean():.5f} ref {b.mean():.5f} d {m:+.6f} se(iid {iid:.6f}, cl {cl:.6f}) t {m / se:+.2f}'
    if split == 'val':
        out['slices'] = {}
        for k, msk in SL.items():
            dm, di, dc = se_pair(d[msk], gv[msk]); sk = max(di, dc); out['slices'][k] = (dm, sk)
        s += '\n' + ' ' * 50 + ' '.join(f'{k} {v[0]:+.5f}+-{v[1]:.5f}({v[0] / v[1]:+.1f})' for k, v in out['slices'].items())
        out['minslice_t'] = min(v[0] / v[1] for k, v in out['slices'].items() if k != 'user0')
    log(s); return out


RES = {}
# ---------------- reference pools + parity with frozen arrays
for name, fams in [('PF7x', PF7x), ('G3', G3), ('M2', M2)]:
    for split in ('val', 'test'):
        off, M = META[split]
        for h in ('ft', 'to'):
            mine = half_pool(fams, split, h); frozen = np.load(f'{P}/ens_{name}_{h}_{split}.npy').astype(np.float64)
            log(f'parity ens_{name}_{h}_{split}: max|diff| {np.abs(mine - frozen).max():.2e}')
sp = {k: spec(f, 'val') for k, f in [('PF7x', PF7x), ('G3', G3), ('M2', M2), ('U45', U45), ('P11', P11)]}
sph = {k: spec(f, 'hold') for k, f in [('PF7x', PF7x), ('G3', G3), ('M2', M2), ('U45', U45), ('P11', P11)]}
for k in sp: log(f'{k} val {nd(sp[k]).mean():.5f} hold {nd(sph[k], "hold").mean():.5f}')

# ---------------- fold audit (S2)
fold_mine = np.zeros(len(Mv), np.int8)
for k, (_, ho) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(Mv)), groups=gv)): fold_mine[ho] = k
fold_s2 = np.load(f'{R}/nn_x/stack/nnstack/data/val_fold.npy')
log('S2 val_fold == my GroupKFold(user, u0 singletons):', bool((fold_mine == fold_s2).all()), np.bincount(fold_mine))
# users never split across folds
us = pl.DataFrame({'u': uv, 'f': fold_s2}).filter(pl.col('u') != 0).group_by('u').agg(pl.col('f').n_unique())
log('users with feeds in >1 fold:', int((us['f'] > 1).sum()))
# empirical alignment with the FT fold models: raw ft OOF per-fold mean logit differs by fold model; compare to shuffled
for m in ['r6_cd_m02_r101', 'v23_gru_pre2s_s501']:
    x = np.load(f'{P}/nn_{m}_n2cpv3_val_oof.npy').astype(np.float64); fm = np.bincount(fidv, x, len(Mv)) / Mv
    def between(fl): return np.var([fm[fl == k].mean() for k in range(5)])
    rnd = [between(np.random.default_rng(s).permutation(fold_s2)) for s in range(20)]
    alt = np.zeros(len(Mv), np.int8)  # GroupKFold WITHOUT u0 singletons, to see the check discriminates
    for k, (_, ho) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(Mv)), groups=uv)): alt[ho] = k
    log(f'fold alignment {m}: between-fold var of feed-mean ft logit: S2 folds {between(fold_s2):.3e}  no-u0-singleton folds {between(alt):.3e}  random max {max(rnd):.3e}')

# ---------------- S2 inputs parity (stacker inputs are the label-free family z's of the claimed member files)
D2 = f'{R}/nn_x/stack/nnstack/data'
for f in G3:
    for h in ('ft', 'to'):
        mine = Z(ov, Mv, FS[(f, 'val', h)] / NF[f]); theirs = np.load(f'{D2}/val_fam_{f}_{h}.npy')
        log(f'S2 input val_fam_{f}_{h} max|diff| {np.abs(mine - theirs).max():.2e}')
log('S2 base val_pool_G3_spec max|diff| vs my G3 spec z', np.abs(Z(ov, Mv, sp['G3']) - np.load(f'{D2}/val_pool_G3_spec.npy')).max())

np.save(f'{C}/fold_s2.npy', fold_s2)
import pickle
pickle.dump({'sp': sp, 'sph': sph}, open(f'{C}/refs.pkl', 'wb'))
log('stage 1 done')
