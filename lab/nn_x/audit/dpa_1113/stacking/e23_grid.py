"""E2 (d) per-family weights grid and E3 (a) feed-conditional weights, nested user-grouped 5-fold on val
(the exact FT folds) + holdout read with full-val fitted weights (TO form). Per-feed NDCG matrices over the grid."""
from lib import *
import time, json, itertools
import pmix

T0 = time.time()
V = S('val'); HO = S('hold')
log = open(OUT / 'e23_grid.txt', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
wpriv = pmix.weights()['private']

def fam_means(sp, half):
    Z = sp.Z(half)
    return [np.asarray(Z[FAMIDX == i], np.float64).mean(0) for i in range(3)]

step = 0.05
A = [(i * step, j * step, round(1 - (i + j) * step, 10)) for i in range(21) for j in range(21) if i + j <= 20]
A = np.array(A)
ME = np.array([6 / 20, 7 / 20, 7 / 20])
iME = int(np.argmin(np.abs(A - ME).sum(1))); assert np.abs(A[iME] - ME).max() < 1e-9
WFT = np.array([0.3, 0.4, 0.5, 0.6, 0.7]); iW5 = 2
pr(f'grid: {len(A)} family-weight points x {len(WFT)} w_ft; ME index {iME} {A[iME]}')

Fv = {h: fam_means(V, h) for h in ('ft', 'to')}
Fh = fam_means(HO, 'to')
# val per-feed NDCG matrix [feeds, nA, nW]
NV = np.zeros((V.n, len(A), len(WFT)), np.float32)
for ia, a in enumerate(A):
    zf = V.z((a[0] * Fv['ft'][0] + a[1] * Fv['ft'][1] + a[2] * Fv['ft'][2]).astype(np.float32))
    zt = V.z((a[0] * Fv['to'][0] + a[1] * Fv['to'][1] + a[2] * Fv['to'][2]).astype(np.float32))
    for iw, w in enumerate(WFT):
        NV[:, ia, iw] = V.ev(w * zf.astype(np.float64) + (1 - w) * zt.astype(np.float64))
NH = np.zeros((HO.n, len(A)), np.float32)
for ia, a in enumerate(A):
    NH[:, ia] = HO.ev(a[0] * Fh[0] + a[1] * Fh[1] + a[2] * Fh[2])
base_v = NV[:, iME, iW5].astype(np.float64); base_h = NH[:, iME].astype(np.float64)
pr(f'[{time.time()-T0:.0f}s] ME val {base_v.mean():.5f} (G3 0.40032)  ME holdout {base_h.mean():.5f} (0.32379)')
np.save(CACHE / 'grid_NV.npy', NV); np.save(CACHE / 'grid_NH.npy', NH); np.save(CACHE / 'grid_A.npy', A)

# ---------------- surfaces (context) ----------------
mv = NV.mean(0); mh = NH.mean(0)
ia, iw = np.unravel_index(np.argmax(mv), mv.shape)
pr(f'SURFACE val: best a={A[ia]} w_ft={WFT[iw]} val {mv[ia, iw]:.5f} (in-sample, +{mv[ia, iw] - mv[iME, iW5]:.5f} vs ME); '
   f'best at w_ft .5: a={A[np.argmax(mv[:, iW5])]} {mv[:, iW5].max():.5f}')
pr(f'SURFACE val w_ft curve at ME: ' + ' '.join(f'{w}:{mv[iME, k]:.5f}' for k, w in enumerate(WFT)))
ih = int(np.argmax(mh))
pr(f'SURFACE holdout: best a={A[ih]} {mh[ih]:.5f} (in-sample +{mh[ih] - mh[iME]:.5f}); holdout at val-best a: {mh[ia]:.5f}; '
   f'val(w.5) at holdout-best a: {mv[ih, iW5]:.5f}')
pr(f'surface corr (val w.5 vs holdout over the 231 a): {np.corrcoef(mv[:, iW5], mh)[0, 1]:.3f}')
for nm, a in [('F10 only', (1, 0, 0)), ('F7x only', (0, 1, 0)), ('F7x70 only', (0, 0, 1)), ('fam-equal', (1/3, 1/3, 1/3)),
              ('no F10', (0, .5, .5)), ('no F7x70', (6/13, 7/13, 0)), ('no F7x', (6/13, 0, 7/13))]:
    k = int(np.argmin(np.abs(A - np.array(a)).sum(1)))
    pr(f'   {nm:11s} a~{A[k]}: val {mv[k, iW5]:.5f} ({mv[k, iW5] - mv[iME, iW5]:+.5f}) holdout {mh[k]:.5f} ({mh[k] - mh[iME]:+.5f})')

# ---------------- nested fits ----------------
fold = V.fold
def report(tag, oof_v, hold_s=None, extra=''):
    dv = oof_v - base_v
    mu, se = se_cl(dv, V.cl); pm, pse = wmean_se(dv, wpriv)
    sl = []
    for nm, m in [('M<50', V.M < 50), ('M>=50', V.M >= 50), ('Tue', V.tue), ('Thu', ~V.tue)]:
        a, b = se_cl(dv, V.cl, m); sl.append(f'{nm} {a:+.5f}({a / b:+.1f})')
    s = f'{tag:34s} VAL {mu:+.5f} +- {se:.5f} (z {mu / se:+.2f}) pmix {pm:+.5f} +- {pse:.5f}'
    out = dict(val=mu, val_se=se, pmix=pm, pmix_se=pse)
    if hold_s is not None:
        dh = hold_s - base_h; hm, hse = se_cl(dh, HO.cl); s += f' | HOLD {hm:+.5f} +- {hse:.5f} (z {hm / hse:+.2f})'
        out.update(hold=hm, hold_se=hse)
    else:
        s += ' | HOLD n/a (w_ft only)'
    pr(s + ' | ' + ' '.join(sl) + extra); return out

RES = {}
def fit(Msub, allowed):
    """argmax over allowed (ia, iw) pairs of the mean of NV rows Msub."""
    m = NV[Msub].mean(0)
    best = max(allowed, key=lambda t: m[t]); return best

ALL_A = [(ia, iw) for ia in range(len(A)) for iw in range(len(WFT))]
COARSE = [ia for ia in range(len(A)) if np.allclose(np.round(A[ia] * 10), A[ia] * 10)]  # step 0.1
PSETS = {
    'a1_wft': [(iME, iw) for iw in range(len(WFT))],
    'a2_fam': [(ia, iW5) for ia in range(len(A))],
    'a2c_fam01': [(ia, iW5) for ia in COARSE],
    'a3_both': ALL_A,
}

def bucketize(sp, var, cuts=None):
    if var == 'none': return np.zeros(sp.n, int), None
    if var == 'M': return np.digitize(sp.M, [50, 100]), None
    if var == 'M2': return (sp.M >= 50).astype(int), None
    if var == 'u0': return sp.u0.astype(int), None
    if var == 'cold':
        c = cuts if cuts is not None else np.quantile(V.cshare, [1 / 3, 2 / 3]); return np.digitize(sp.cshare, c), c
    if var == 'H':
        c = cuts if cuts is not None else np.quantile(V.H, [1 / 3, 2 / 3]); return np.digitize(sp.H, c), c
    if var == 'day': return (~sp.tue).astype(int) if sp.name == 'val' else None, None
    if var == 'Mxcold':
        c = cuts if cuts is not None else np.quantile(V.cshare, [1 / 3, 2 / 3])
        return (sp.M >= 50).astype(int) * 3 + np.digitize(sp.cshare, c), c
    raise KeyError(var)

for var in ['none', 'M', 'M2', 'cold', 'H', 'u0', 'Mxcold', 'day']:
    bv, cuts = bucketize(V, var)
    bh = bucketize(HO, var, cuts)[0] if var != 'day' else None
    for pname, allowed in PSETS.items():
        if var == 'Mxcold' and pname not in ('a2_fam', 'a2c_fam01', 'a1_wft'): continue
        oof = np.zeros(V.n); chosen = {}
        for k in range(5):
            tr = fold != k
            for b in np.unique(bv):
                ia, iw = fit(tr & (bv == b), allowed)
                te = (fold == k) & (bv == b)
                oof[te] = NV[te, ia, iw]
                chosen.setdefault(int(b), []).append((ia, iw))
        # full-val fit per bucket -> holdout
        full = {int(b): fit(bv == b, allowed) for b in np.unique(bv)}
        hold = None
        if bh is not None and pname != 'a1_wft':
            hold = base_h.copy()
            for b in np.unique(bh):
                if int(b) in full:
                    hold[bh == b] = NH[bh == b, full[int(b)][0]]
        fs = ' ; '.join(f'b{b}: a={tuple(np.round(A[full[b][0]], 2))} w={WFT[full[b][1]]} n={int((bv == b).sum())}'
                        + (f'/h{int((bh == b).sum())}' if bh is not None else '') for b in full)
        stab = ' fold-choices ' + ' '.join(f'b{b}:' + ','.join(f'{tuple(np.round(A[ia], 2))}/{WFT[iw]}' for ia, iw in v) for b, v in chosen.items()) if var == 'none' else ''
        RES[f'{var}|{pname}'] = report(f'{var:7s} {pname}', oof, hold, '\n      full-val fit: ' + fs + stab)
        RES[f'{var}|{pname}']['oof'] = None
        np.save(CACHE / f'oof_grid_{var}_{pname}.npy', oof)
        if hold is not None: np.save(CACHE / f'hold_grid_{var}_{pname}.npy', hold)

# conditional vs unconditional (matched control) paired deltas
pr('\n== conditional minus unconditional (same parameter set), paired ==')
for var in ['M', 'M2', 'cold', 'H', 'u0', 'Mxcold', 'day']:
    for pname in PSETS:
        f = CACHE / f'oof_grid_{var}_{pname}.npy'
        if not f.exists(): continue
        d = np.load(f) - np.load(CACHE / f'oof_grid_none_{pname}.npy'); mu, se = se_cl(d, V.cl)
        s = f'  {var:7s} {pname:10s} VAL {mu:+.5f} +- {se:.5f} (z {mu / se:+.2f})'
        fh = CACHE / f'hold_grid_{var}_{pname}.npy'
        if fh.exists():
            dh = np.load(fh) - np.load(CACHE / f'hold_grid_none_{pname}.npy'); hm, hse = se_cl(dh, HO.cl)
            s += f' | HOLD {hm:+.5f} +- {hse:.5f} (z {hm / hse:+.2f})'
        pr(s)
json.dump(RES, open(OUT / 'e23_grid.json', 'w'), indent=1)
pr(f'[{time.time()-T0:.0f}s] done')
