"""B. Pool level: FT pool vs TO pool, blend-weight curve, nested user-half optimum of w_ft, slices. Students (W8, N8, S16) vs C3."""
from lib import *
day = (tv.astype(np.int64) + 3 * 3600) // 86400; dd = np.unique(day)
SL = {'all': np.ones(len(Mv), bool), 'M<50': Mv < 50, 'M>=50': Mv >= 50, 'Tue': day == dd[0], 'Thu': day == dd[2], 'user0': uv == 0, 'known': uv != 0,
      'H<=20': Hv <= 20, 'H21-100': (Hv > 20) & (Hv <= 100), 'H101-255': (Hv > 100) & (Hv < 256), 'H256': Hv >= 256}
POOLS = {'W8': W8, 'N8': N8, 'S16': W8 + N8, 'kd05w192': GROUPS['S:kd05w192'], 'kd05t25w192': GROUPS['S:kd05t25w192'], 'kd08': GROUPS['S:kd08'],
         'C3': C3M, 'G3': GROUPS['H:gru'] + GROUPS['H:m02'] + GROUPS['H:m02p70k'], 'm02x7': GROUPS['H:m02'], 'pop1x8': GROUPS['H:pop1']}
ws = np.round(np.arange(0, 1.01, 0.1), 2)
# user-half split (two fixed hashes); user 0 feeds split by feed index
def halves(seed):
    rng = np.random.default_rng(seed); us = np.unique(uv); h = dict(zip(us, rng.integers(0, 2, len(us))))
    a = np.array([h[u] for u in uv]); u0 = uv == 0; a[u0] = rng.integers(0, 2, u0.sum()); return a == 0
res = {}
print(f"{'pool':12s} n   FT       TO       0.5/0.5  | FT-TO (paired)        blend-TO              | " + ' '.join(f'w{w:.1f}' .rjust(8) for w in ws))
for k, ms in POOLS.items():
    pf, pt = pool(ms, 'ft', 'val'), pool(ms, 'to', 'val'); zf, zt_ = zv(pf), zv(pt)
    e = {w: ev(w * zf + (1 - w) * zt_) for w in ws}; res[k] = e
    print(f"{k:12s} {len(ms):2d} {e[1.0].mean():.5f} {e[0.0].mean():.5f} {e[0.5].mean():.5f} | {f(e[1.0]-e[0.0])} {f(e[0.5]-e[0.0])} | " + ' '.join(f'{e[w].mean()-e[0.0].mean():+.5f}' for w in ws))
print('\n== slices: FT pool - TO pool | 0.5/0.5 - TO pool')
for k in ('W8', 'N8', 'S16', 'C3', 'G3'):
    e = res[k]; print(f' {k:5s} ' + ' | '.join(f'{s}: {f((e[1.0]-e[0.0])[m])} / {f((e[0.5]-e[0.0])[m])}' for s, m in SL.items()))
print('\n== nested w_ft (choose w on one user half, apply to the other; 20 random user-half splits; both directions)')
for k in ('W8', 'N8', 'S16', 'C3', 'G3'):
    e = res[k]; E = np.array([e[w] for w in ws]); gains = []; picks = []
    for sdd in range(20):
        a = halves(sdd)
        for tr, te in ((a, ~a), (~a, a)):
            wi = int(np.argmax(E[:, tr].mean(1))); picks.append(ws[wi]); gains.append((E[wi, te] - E[5, te]).mean())  # vs w 0.5
    full = ws[int(np.argmax(E.mean(1)))]
    d_best = E[int(np.argmax(E.mean(1)))] - E[5]
    print(f' {k:5s} full-val argmax w_ft {full:.1f} (in-sample gain vs 0.5 {f(d_best)}); nested picks mean {np.mean(picks):.2f} sd {np.std(picks):.2f} (min {np.min(picks):.1f} max {np.max(picks):.1f}); nested gain vs 0.5: {np.mean(gains):+.5f} (sd over splits {np.std(gains):.5f})')
# fixed alternatives, paired vs 0.5
print('\n== fixed-weight alternatives vs 0.5/0.5 (paired per feed), W8 / S16 / C3')
for k in ('W8', 'S16', 'C3'):
    e = res[k]; print(f' {k:4s} ' + ' | '.join(f'w{w:.1f}: {f(e[w]-e[0.5])}' for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.7, 1.0)))
# cross branches: students TO + C3 FT etc.
print('\n== cross-branch files (val): 0.5 z(FT of X) + 0.5 z(TO of Y)')
PF = {k: zv(pool(POOLS[k], 'ft', 'val')) for k in ('W8', 'S16', 'C3')}; PT = {k: zv(pool(POOLS[k], 'to', 'val')) for k in ('W8', 'S16', 'C3')}
base = ev(0.5 * PF['W8'] + 0.5 * PT['W8'])
for a in PF:
    for b in PT:
        e = ev(0.5 * PF[a] + 0.5 * PT[b]); print(f'  FT {a:4s} + TO {b:4s}: {e.mean():.5f}  vs W8 file {f(e-base)}')
e = ev(PT['W8']); print(f'  TO W8 alone: {e.mean():.5f} vs W8 file {f(e-base)}')
for wc in (0.25, 0.5):
    e = ev(0.5 * zv((1 - wc) * PF['W8'] + wc * PF['C3']) + 0.5 * PT['W8']); print(f'  FT = {1-wc:.2f} W8 + {wc:.2f} C3 ; TO W8: {e.mean():.5f} vs W8 file {f(e-base)}')
    e = ev(0.5 * zv((1 - wc) * PF['W8'] + wc * PF['C3']) + 0.5 * zv((1 - wc) * PT['W8'] + wc * PT['C3'])); print(f'  both branches {1-wc:.2f} W8 + {wc:.2f} C3: {e.mean():.5f} vs W8 file {f(e-base)}')
