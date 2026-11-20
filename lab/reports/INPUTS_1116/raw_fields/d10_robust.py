"""INVENTORY D, POST HOC robustness of the flagged forms to the shape of the score control: control R = 11 columns
[s, s^2, s^3, s*lm, s^2*lm, s*lm^2, hinge(s+1), hinge(s), hinge(s-1), hinge(s-2), within-feed percentile rank of s], lm = log(M/50).
A column correlated with the score could otherwise absorb a misspecified score transform. CPU only, read-only."""
import sys
sys.path.insert(0, '<repo>/reports/INPUTS_1116/raw_fields')
from d03_lib import *
from d03_lib import _z
L = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); L.append(s)
V = load_val(); Hd = load_hold()
@nb.njit(parallel=True, cache=True)
def pct(off, M, x):
    out = np.empty(len(x))
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]; r = np.argsort(np.argsort(x[o:o + m]))
        for k in range(m): out[o + k] = r[k] / max(m - 1, 1)
    return out
def rich(S, s):
    z = _z(S.off, S.M, np.asarray(s, np.float64)); lm = np.log(np.repeat(S.M, S.M) / 50.0); h = lambda c: np.maximum(z - c, 0)
    return np.column_stack([z, z * z, z ** 3, z * lm, z * z * lm, z * lm * lm, h(-1), h(0), h(1), h(2), pct(S.off, S.M, z)])
V.ctrl['R'] = rich(V, V.score['TO']); V.score['R'] = V.score['TO']; Hd.ctrl['R'] = rich(Hd, Hd.score['H']); Hd.score['R'] = Hd.score['H']
X = lambda n, sp: np.load(OUT / f'x_{n}_{sp}.npy').astype(np.float64)
ld = lambda sp, n: np.load(F / sp / f'{n}.npy', mmap_mode='r')
fl = {'val': (np.asarray(ld('val', 'cand_f1')) < 0.018).astype(np.float64), 'hold': (np.asarray(ld('train', 'cand_f1')[Hd.rows]) < 0.018).astype(np.float64)}
for n in ('V2b_uc_ctr', 'C3_uc_coec', 'V2a_ui_reshown', 'C2_click_sim', 'V8a_f2_home', 'V8b_f1_tie', 'D4_f1_floor', 'V3_it_coec', 'V5a_it_s1share'):
    xv = fl['val'] if n == 'D4_f1_floor' else X(n, 'val')
    P(fmt(n, 'val', 'R', lr_test(V, xv, 'R', cv=False)))
    if n not in ('V3_it_coec', 'V5a_it_s1share'):
        xh = fl['hold'] if n == 'D4_f1_floor' else X(n, 'hold')
        P(fmt(n, 'hold', 'R', lr_test(Hd, xh, 'R', cv=False)))
P(f'rich control fit: val ll {V._base["R"][1]:.1f} (4-column control: see d04a) | hold ll {Hd._base["R"][1]:.1f}')
(OUT / 'd10_robust.txt').write_text('\n'.join(L) + '\n')
