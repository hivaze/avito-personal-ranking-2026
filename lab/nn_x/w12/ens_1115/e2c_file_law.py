"""e2 part D: two-pool law AT FILE LEVEL (final val = 0.5 z(FT) + 0.5 z(TO), per branch 0.8 z(base) + 0.2 z(pop)), all-data members:
  F(nb, np) = a - cb / nb - cp / np,   base = nb of the 16 full-data wide students, pop = np of the 12 full-data wide pop1 students (random subsets).
Marginal value of one more student: cb / nb^2 (base) against cp / np^2 (popularity-light).  Also on val reweighted to the public / private mix.  Output out/file_law.json."""
from ens_common import *
import time, pmix
K = int(sys.argv[1]) if len(sys.argv) > 1 else 40
m = Meta(); st = Store(m); WTS = pmix.weights(); rng = np.random.default_rng(1117); t0 = time.time(); Z = lambda x: m.z('val', x)
BF = st.mat('ft_val', FD16).astype(np.float64); BT = st.mat('to_val', FD16).astype(np.float64); PF = st.mat('ft_val', FDP12).astype(np.float64); PT = st.mat('to_val', FDP12).astype(np.float64)
pts = []
for nb_ in (1, 2, 3, 4, 6, 8, 12, 16):
    for np_ in (1, 2, 3, 4, 6, 8, 12):
        acc = np.zeros(len(m.Mv)); k = 1 if (nb_ == 16 and np_ == 12) else K; lv = []
        for _ in range(k):
            b = rng.choice(16, nb_, replace=False); p = rng.choice(12, np_, replace=False)
            e = m.ev(0.5 * Z(0.8 * Z(BF[b].mean(0)) + 0.2 * Z(PF[p].mean(0))) + 0.5 * Z(0.8 * Z(BT[b].mean(0)) + 0.2 * Z(PT[p].mean(0)))); acc += e; lv.append(e.mean())
        pts.append((nb_, np_, acc / k, float(np.std(lv))))
    print(nb_, f'{time.time() - t0:.0f}s', flush=True)
A = np.array([[1.0, -1.0 / nb_, -1.0 / np_] for nb_, np_, _, _ in pts]); G = np.linalg.pinv(A); Y = np.stack([p[2] for p in pts]); pf = G @ Y; out = {}
for nm, w in (('val', None), ('public', WTS['public']), ('private', WTS['private'])):
    co = []; s_ = []
    for i in range(3):
        a, b = pmix.wmean_se(pf[i], w); co.append(float(a)); s_.append(float(b))
    obs = Y.mean(1) if w is None else np.array([np.average(y_, weights=w) for y_ in Y]); fitv = A @ np.array(co)
    def con(b1, p1, b2, p2):
        v = np.array([0.0, -(1 / b2 - 1 / b1), -(1 / p2 - 1 / p1)]); d = v @ pf; a, b = pmix.wmean_se(d, w); return float(a), float(b)
    fc = {k_: con(*v_) for k_, v_ in {'16+4 -> 16+12': (16, 4, 16, 12), '16+12 -> 32+12': (16, 12, 32, 12), '16+12 -> 16+24': (16, 12, 16, 24), '16+12 -> 32+24': (16, 12, 32, 24), '16+12 -> 64+48': (16, 12, 64, 48),
                                      '16+12 -> inf': (16, 12, 1e9, 1e9), '16+4 -> 32+12': (16, 4, 32, 12), '16+4 -> inf': (16, 4, 1e9, 1e9), '8+4 -> 16+4': (8, 4, 16, 4)}.items()}
    out[nm] = dict(a=co[0], cb=co[1], cp=co[2], se=s_, max_resid=float(np.abs(fitv - obs).max()), forecast=fc, marginal_base_at16=co[1] / 256, marginal_pop_at12=co[2] / 144, marginal_pop_at4=co[2] / 16)
    print(f'{nm}: a {co[0]:.5f} cb {co[1]:.5f}+-{s_[1]:.5f} cp {co[2]:.5f}+-{s_[2]:.5f} max resid {np.abs(fitv - obs).max():.5f} | one more student: base at 16 {co[1] / 256:+.6f}, pop at 12 {co[2] / 144:+.6f}, pop at 4 {co[2] / 16:+.6f} | '
          + ' | '.join(f'{k_} {v_[0]:+.5f}+-{v_[1]:.5f}' for k_, v_ in fc.items()), flush=True)
out['obs'] = {f'{nb_}+{np_}': (float(y_.mean()), sd) for (nb_, np_, y_, sd) in pts}
json.dump(out, open(OUT / 'file_law.json', 'w'), indent=0)
print('observed grid (final val):'); print('nb\\np ' + ' '.join(f'{p:>8d}' for p in (1, 2, 3, 4, 6, 8, 12)))
for nb_ in (1, 2, 3, 4, 6, 8, 12, 16): print(f'{nb_:5d} ' + ' '.join(f'{out["obs"][f"{nb_}+{p}"][0]:.5f}' for p in (1, 2, 3, 4, 6, 8, 12)))
print('draw sd at 16 + 4:', out['obs']['16+4'][1], ' at 8 + 4:', out['obs']['8+4'][1], f'done {time.time() - t0:.0f}s')
