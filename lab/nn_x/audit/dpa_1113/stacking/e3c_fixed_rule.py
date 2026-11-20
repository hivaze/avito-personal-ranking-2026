"""E3c: the EXACT M-conditional w_ft rule fitted on G3 (full val: w_ft 0.3 for M<50, 0.6 for M>=50; e23_grid 'M2 a1_wft')
applied FROZEN (no refit) to member sets that are not in G3. Paired val delta vs the same set at w_ft 0.5."""
from lib import *
import re, glob
import pmix
V = S('val'); wpriv = pmix.weights()['private']
log = open(OUT / 'e3c_fixed_rule.txt', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
have = lambda m: (P / f'nn_{m}_n2cpv3_val_oof.npy').exists() and (P / f'nn_{m}_trainonly_val.npy').exists()
FAMS = {
    'F5cd': ['v15_cd_all_e10', 'v15_cd_all_s81', 'v15_cd_all_s82', 'v15_cd_allprew_s83', 'v15_cd_allprew_s84', 'v15_cd_allprew_s85'],
    'F6': ['v13_lr1_mix1', 'v14_s92', 'v14_s93', 'v14_s94', 'v14_s95', 'v14_s96'],
    'F9': [f'v22_pre2s_lr1_s{s}' for s in range(401, 407)],
    'F12': [f'v26_d192_pre2s_s{s}' for s in range(701, 708)],
    'F7(v17)': [f'v17_pre2s_s{s}' for s in (101, 102, 103, 105, 106, 107)],
}
for pre in ['v27_gru_m02', 'v28_lr1_m02', 'v29_d192_m02', 'v30_f7_m02', 'abl_e10_pop1']:
    ms = sorted({re.sub(r'^nn_|_trainonly_val\.npy$', '', Path(f).name) for f in glob.glob(str(P / f'nn_{pre}_s*_trainonly_val.npy'))})
    FAMS[pre] = [m for m in ms if '@' not in m and 'np' not in m.split('_')[-1] and have(m)]
w = np.where(np.repeat(V.M, V.M) < 50, 0.3, 0.6)
D = []
for nm, ms in list(FAMS.items()) + [('G3 (fit set, in-sample)', MEMBERS)]:
    ms = [m for m in ms if have(m)]
    if len(ms) < 3: pr(nm, 'skipped', len(ms)); continue
    zf = V.z(np.mean([V.z(np.load(P / f'nn_{m}_n2cpv3_val_oof.npy').astype(np.float32)) for m in ms], 0).astype(np.float32)).astype(np.float64)
    zt = V.z(np.mean([V.z(np.load(P / f'nn_{m}_trainonly_val.npy').astype(np.float32)) for m in ms], 0).astype(np.float32)).astype(np.float64)
    d = V.ev(w * zf + (1 - w) * zt) - V.ev(0.5 * zf + 0.5 * zt)
    mu, se = se_cl(d, V.cl); pm, pse = wmean_se(d, wpriv)
    a, b = se_cl(d, V.cl, V.M < 50); c, e = se_cl(d, V.cl, V.M >= 50)
    pr(f'{nm:24s} n={len(ms):2d} rule-vs-w.5 {mu:+.5f} +- {se:.5f} (z {mu / se:+.2f}) pmix {pm:+.5f} | M<50 {a:+.5f} (z {a / b:+.2f}) M>=50 {c:+.5f} (z {c / e:+.2f})')
    if not nm.startswith('G3'): D.append(d)
D = np.array(D)
mu, se = se_cl(D.mean(0), V.cl)
pr(f'mean over {len(D)} fresh sets (per-feed average of their deltas, user-clustered SE): {mu:+.5f} +- {se:.5f} (z {mu / se:+.2f}); '
   f'sets with delta > 0: {(D.mean(1) > 0).sum()}/{len(D)}')
