"""STUDY C context table: private-like (adversarially weighted) val deltas for the pools / PoPs (no new candidates) (plan.txt in this dir, written before any number).
Usage: python agg.py  -> agg_out.txt / agg_out.json in this dir. Read-only on preds/; writes only here."""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[_k] = '12'
import sys, re, glob, json, time
sys.path.insert(0, '<repo>/src')
from pathlib import Path
import numpy as np, polars as pl
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path('<repo>'); P = ROOT / 'preds'; H = P / 'holdout'; OUT = ROOT / 'nn_x/stack/agg'
T0 = time.time()
LOG = open(OUT / 'agg4_out.txt', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

FAM = {
    'F5cd': ['v15_cd_all_e10', 'v15_cd_all_s81', 'v15_cd_all_s82', 'v15_cd_allprew_s83', 'v15_cd_allprew_s84', 'v15_cd_allprew_s85'],
    'F6': ['v13_lr1_mix1', 'v14_s92', 'v14_s93', 'v14_s94', 'v14_s95', 'v14_s96'],
    'F9': [f'v22_pre2s_lr1_s{s}' for s in range(401, 407)],
    'F10': [f'v23_gru_pre2s_s{s}' for s in range(501, 507)],
    'F12': [f'v26_d192_pre2s_s{s}' for s in range(701, 708)],
    'F7x': [f'r6_cd_m02_r{s}' for s in range(101, 108)],
    'F7x70': [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)],
}
NEWFAM = {'v27_gru_m02': 'N27', 'v28_lr1_m02': 'N28', 'v29_d192_m02': 'N29', 'v30_f7_m02': 'N30'}
have = lambda m: all(p.exists() for p in [P / f'nn_{m}_n2cpv3_val_oof.npy', P / f'nn_{m}_n2cpv3_test.npy', P / f'nn_{m}_trainonly_val.npy',
                                           P / f'nn_{m}_trainonly_test.npy', H / f'nn_{m}_holdout.npy'])
cand_new = sorted({re.sub(r'^nn_|_trainonly_val\.npy$', '', Path(f).name) for pre in NEWFAM for f in glob.glob(str(P / f'nn_{pre}_s*_trainonly_val.npy'))})
cand_new = [m for m in cand_new if '@' not in m and have(m)]

def family(m):
    for f, ms in FAM.items():
        if m in ms: return f
    for pre, f in NEWFAM.items():
        if m.startswith(pre): return f
    raise KeyError(m)

def trunk(m):  # F7x and F7x70 of the same root share a trunk; everything else is its own trunk
    r = re.match(r'^r6_cd_m02_(?:p70k_)?r(\d+)$', m)
    return f'T{r.group(1)}' if r else m

# ---------------- splits ----------------
class Split:
    def __init__(self, name):
        self.name = name
        if name == 'val':
            fv, M, off = feed_meta('val')
            self.M, self.off = M.astype(np.int64), off.astype(np.int64)
            self.y = np.load(FLAT / 'val/cand_target.npy').astype(np.float64)
            u = fv['user_id'].to_numpy(); ts = fv['timestamp'].to_numpy()
        else:
            M, off = np.load(H / 'holdout_meta.npy'); self.M, self.off = M.astype(np.int64), off.astype(np.int64)
            self.y = np.load(H / 'holdout_y.npy').astype(np.float64)
            tr = pl.read_parquet(FLAT / 'train/feeds.parquet', columns=['timestamp', 'M', 'user_id'])
            prr = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0)
            trM = tr['M'].to_numpy(); prr[trM < 50] = 0.0
            cand = np.where((prr > 0.9) & (trM >= 50))[0]
            hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
            assert (trM[hold] == self.M).all()
            u = tr['user_id'].to_numpy()[hold]; ts = tr['timestamp'].to_numpy()[hold]
        self.n = len(self.M); self.fid = np.repeat(np.arange(self.n), self.M)
        uu = np.where(u == 0, -1 - np.arange(self.n), u)  # user 0 = singletons
        self.cl = np.unique(uu, return_inverse=True)[1]; self.G = self.cl.max() + 1
        if name == 'val':
            tue = (ts + 10800) // 86400 < 20720
            cold = np.nan_to_num(np.load(ROOT / 'nn_x/r6/feat_cache/val/it_hist_all_luo.npy').astype(np.float64), nan=0.0) == 0
            npos = np.bincount(self.fid, self.y, self.n); ncold = np.bincount(self.fid, self.y * cold, self.n)
            self.slices = [('M<50', self.M < 50), ('M>=50', self.M >= 50), ('Tue', tue), ('Thu', ~tue),
                           ('coldpos', (npos > 0) & (ncold == npos)), ('warmpos', (npos > 0) & (ncold == 0))]
        else:
            dow = (((ts + 10800) // 86400) + 3) % 7
            self.slices = [('late20%', ts >= np.quantile(ts, 0.8)), ('wkday', dow < 5), ('wkend', dow >= 5)]

    def rz(self, x):
        return feed_rank_z(self.off, self.M, np.asarray(x, np.float32))
    def z(self, x):
        return self.rz(x)[:, 1].astype(np.float64)
    def ev(self, s):
        return ndcg_per_feed(np.asarray(s, np.float64), self.y, self.off, self.M)
    def se(self, d, mask=None):
        """max(iid SE, user-clustered SE) of the mean of d (optionally on a mask)."""
        if mask is not None:
            d = d[mask]; cl = self.cl[mask]
        else:
            cl = self.cl
        n = len(d); mu = d.mean()
        iid = d.std() / np.sqrt(n)
        s = np.bincount(cl, d - mu); g = (s != 0).sum()
        clu = np.sqrt((s ** 2).sum()) / n * np.sqrt(g / max(g - 1, 1))
        return max(iid, clu), iid, clu
    def segmax(self, x):
        return np.maximum.reduceat(x, self.off)
    def segsum(self, x):
        return np.add.reduceat(x, self.off)

V = Split('val'); HO = Split('hold')
pr(f'[{time.time()-T0:.0f}s] splits: val {V.n} feeds {len(V.y)} rows, holdout {HO.n} feeds {len(HO.y)} rows')

# ---------------- member z / rank cache ----------------
ZC = {}  # (split, half, m) -> (z float32, rank float32)
def get(split, half, m):
    k = (split, half, m)
    if k not in ZC:
        if split == 'val':
            f = P / (f'nn_{m}_n2cpv3_val_oof.npy' if half == 'ft' else f'nn_{m}_trainonly_val.npy'); S = V
        else:
            assert half == 'to'; f = H / f'nn_{m}_holdout.npy'; S = HO
        rz = S.rz(np.load(f)); ZC[k] = (rz[:, 1].copy(), rz[:, 0].copy())
    return ZC[k]

# ---------------- member gate for NEW ----------------
NEW = []
pr('NEW member gate (a): single to val >= 0.39467 and holdout >= 0.31799')
for m in cand_new:
    sv = V.ev(get('val', 'to', m)[0]).mean(); sh = HO.ev(get('hold', 'to', m)[0]).mean()
    ok = sv >= 0.39467 and sh >= 0.31799
    pr(f'  {m:24s} to val {sv:.5f} hold {sh:.5f} {"PASS" if ok else "fail"}'); NEW += [m] if ok else []
pr(f'NEW eligible: {len(NEW)} {NEW}')

fam = lambda *fs: [m for f in fs for m in FAM[f]]
SETS = {
    'PF7x': fam('F5cd', 'F6', 'F7x', 'F9', 'F10', 'F12'),
    'G3': fam('F10', 'F7x', 'F7x70'),
    'M2': fam('F7x', 'F7x70'),
    'ALLg': fam('F5cd', 'F6', 'F7x', 'F9', 'F10', 'F12', 'F7x70') + NEW,
}
if NEW:
    SETS['N1=G3+NEW'] = fam('F10', 'F7x', 'F7x70') + NEW
for k, ms in SETS.items():
    assert len(set(ms)) == len(ms)
    for m in ms: assert have(m), m

# ---------------- aggregators over a list of member arrays (one half) ----------------
def A_mean(S, ms, split, half):
    return np.mean([get(split, half, m)[0] for m in ms], 0, dtype=np.float64)

def A_fameq(S, ms, split, half):
    fs = sorted({family(m) for m in ms})
    return np.mean([np.mean([get(split, half, m)[0] for m in ms if family(m) == f], 0, dtype=np.float64) for f in fs], 0)

def A_trunk(S, ms, split, half):
    ts = sorted({trunk(m) for m in ms})
    return np.mean([np.mean([get(split, half, m)[0] for m in ms if trunk(m) == t], 0, dtype=np.float64) for t in ts], 0)

def A_trim(p):
    def f(S, ms, split, half):
        X = np.stack([get(split, half, m)[0] for m in ms], 1)  # (N, K)
        X.sort(axis=1); K = X.shape[1]; c = int(p * K)
        return X[:, c:K - c].mean(1, dtype=np.float64)
    return f

def A_median(S, ms, split, half):
    X = np.stack([get(split, half, m)[0] for m in ms], 1)
    return np.median(X, 1).astype(np.float64)

def A_smx(T):
    def f(S, ms, split, half):
        acc = np.zeros(len(S.y))
        for m in ms:
            x = get(split, half, m)[0].astype(np.float64) / T
            e = np.exp(x - np.repeat(S.segmax(x), S.M))
            acc += e / np.repeat(S.segsum(e), S.M)
        return np.log(acc / len(ms))
    return f

def A_rankmean(S, ms, split, half):
    r = np.mean([get(split, half, m)[1] for m in ms], 0, dtype=np.float64)
    return r + 1e-6 * A_mean(S, ms, split, half)  # z tie-break (1e-6*|z| < rank granularity 1/(K*(M-1)))

AGG = {'mean': A_mean, 'fam-eq': A_fameq, 'trim10': A_trim(0.10), 'trim20': A_trim(0.20), 'median': A_median,
       'smx_T0.5': A_smx(0.5), 'smx_T1': A_smx(1.0), 'smx_T2': A_smx(2.0), 'rankmean': A_rankmean, 'trunk-dedup': A_trunk}

def final_val(agg, ms):
    a = V.z(AGG[agg](V, ms, 'val', 'ft')); b = V.z(AGG[agg](V, ms, 'val', 'to'))
    return 0.5 * a + 0.5 * b, b
def final_hold(agg, ms):
    return AGG[agg](HO, ms, 'hold', 'to')

# ---------------- reporting ----------------
RES = []
def report(setname, var, sv, sto, sh, ref):
    """sv: final val score; sto: to-only val score (or None); sh: holdout score (or None); ref: dict of reference per-feed ndcg arrays."""
    ev = V.ev(sv); dv = ev - ref['v']; se_v = V.se(dv)[0]
    row = dict(set=setname, var=var, val=ev.mean(), dv=dv.mean(), se_v=se_v, z_v=dv.mean() / max(se_v, 1e-12))
    sl = {}
    for n, msk in V.slices:
        d = dv[msk].mean(); s = V.se(dv, msk)[0]; sl[n] = (d, d / max(s, 1e-12))
    row['slices'] = sl
    if sto is not None:
        dt = V.ev(sto) - ref['vto']; row['dvto'] = dt.mean(); row['se_vto'] = V.se(dt)[0]
    if sh is not None:
        eh = HO.ev(sh); dh = eh - ref['h']; s = HO.se(dh)[0]
        row.update(hold=eh.mean(), dh=dh.mean(), se_h=s, z_h=dh.mean() / max(s, 1e-12))
        row['hslices'] = {n: (dh[msk].mean(), dh[msk].mean() / max(HO.se(dh, msk)[0], 1e-12)) for n, msk in HO.slices}
    # acceptance (lead's rule + plan's sign guard)
    ok2 = row['z_v'] > 2
    ok3 = (row['z_h'] >= -1) if sh is not None else None
    ok4 = min(z for _, z in sl.values()) >= -2
    row['accept'] = bool(ok2 and (ok3 is not False) and ok4 and (sh is None or row['dh'] > 0)) and not setname.startswith('vs')
    row['crit'] = f"ii={'Y' if ok2 else 'N'} iii={'-' if ok3 is None else ('Y' if ok3 else 'N')} iv={'Y' if ok4 else 'N'}"
    RES.append(row)
    h = f"hold {row['hold']:.5f} dH {row['dh']:+.5f}+-{row['se_h']:.5f} (z {row['z_h']:+.1f})" if sh is not None else 'hold   n/a'
    t = f"dVto {row['dvto']:+.5f}+-{row['se_vto']:.5f}" if sto is not None else ''
    pr(f"  {setname:10s} {var:16s} val {row['val']:.5f} dV {row['dv']:+.5f}+-{se_v:.5f} (z {row['z_v']:+.1f}) {t} | {h} | "
       + ' '.join(f'{n} {d:+.5f}({z:+.1f})' for n, (d, z) in sl.items())
       + ('' if sh is None else ' | H: ' + ' '.join(f'{n} {d:+.5f}({z:+.1f})' for n, (d, z) in row['hslices'].items()))
       + f" | {row['crit']} {'ACCEPT' if row['accept'] else ''}")

def ref_of(ms):
    sv, sto = final_val('mean', ms); sh = final_hold('mean', ms)
    return dict(v=V.ev(sv), vto=V.ev(sto), h=HO.ev(sh)), (sv, sto, sh)

REFS, REFSC = {}, {}
for sname, ms in SETS.items():
    REFS[sname], REFSC[sname] = ref_of(ms)
    pr(f'[{time.time()-T0:.0f}s] REF {sname}: n={len(ms)} families={sorted({family(m) for m in ms})} trunks={len({trunk(m) for m in ms})} '
       f'val {REFS[sname]["v"].mean():.5f} val_to {REFS[sname]["vto"].mean():.5f} hold {REFS[sname]["h"].mean():.5f}')


Wp = np.load(ROOT / 'nn_x/pubdiag/shift/p2_adv_weights.npz')['private_intrinsic+hour']
def wse(d):
    w = Wp; mu = (w * d).sum() / w.sum(); s = np.bincount(V.cl, w * (d - mu))
    return mu, np.sqrt((s ** 2).sum()) / w.sum()
SETS['NEW'] = list(NEW); REFS['NEW'], REFSC['NEW'] = ref_of(NEW)
def pop_scores(pop):
    return (np.mean([V.z(REFSC[s][0]) for s in pop], 0), np.mean([V.z(REFSC[s][1]) for s in pop], 0), np.mean([HO.z(REFSC[s][2]) for s in pop], 0))
C = {k: REFSC[k] for k in ('PF7x', 'G3', 'M2', 'N1=G3+NEW', 'ALLg')}
C['PoP3'] = pop_scores(('PF7x', 'G3', 'M2')); C['PoP2M'] = pop_scores(('PF7x', 'M2'))
C['P5a'] = pop_scores(('PF7x', 'G3', 'M2', 'NEW')); C['P5b'] = pop_scores(('PF7x', 'M2', 'NEW'))
E = {k: (V.ev(a), V.ev(b), HO.ev(c)) for k, (a, b, c) in C.items()}
for base in ('PF7x', 'G3'):
    pr(f'\n== vs {base}: V, Vp(private-like adv weights), Vto, H, D=Vto-H, E0=Vp, E1=Vp-D, S=min(E0,E1), S_V=min(V,V-D) ==')
    v0, t0, h0 = E[base]
    for k, (v, t, h) in E.items():
        dv, dt, dh = v - v0, t - t0, h - h0; D = dt.mean() - dh.mean(); vp = wse(dv)
        pr(f'  {k:10s} V {dv.mean():+.5f}+-{V.se(dv)[0]:.5f} Vp {vp[0]:+.5f}+-{vp[1]:.5f} Vto {dt.mean():+.5f} H {dh.mean():+.5f}+-{HO.se(dh)[0]:.5f} '
           f'D {D:+.5f} E0 {vp[0]:+.5f} E1 {vp[0]-D:+.5f} S {min(vp[0], vp[0]-D):+.5f} S_V {min(dv.mean(), dv.mean()-D):+.5f}')
pr(f'done {time.time()-T0:.0f}s')
