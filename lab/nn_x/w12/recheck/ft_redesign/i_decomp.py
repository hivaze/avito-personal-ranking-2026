"""I. What is the FT shift made of? Pool level, s = z(FT pool) - z(TO pool) per candidate; cold = it_hist_all_luo == 0.
 (i) between-group part: within each feed, the mean of s over its cold candidates / over its warm candidates (a feed-specific cold offset);
 (ii) within-group part: s minus (i) (re-ranking inside the cold and inside the warm candidates).
 NDCG of z_TO + part, vs TO pool: all / coldpos / warmpos / public-mix / private-mix. Plus a ONE-parameter global cold offset delta (nested by user halves)."""
from lib import *
import pmix
luo = np.load(FEATS / 'val/it_hist_all_luo.npy'); cold = (np.nan_to_num(np.asarray(luo, np.float64), nan=0.0) == 0).astype(np.float64)
luot = np.load(FEATS / 'test/it_hist_all_luo.npy'); coldt = (np.nan_to_num(np.asarray(luot, np.float64), nan=0.0) == 0).astype(np.float64)
npos = np.add.reduceat(yv, ov); ncold = np.add.reduceat(yv * cold, ov)
SL = {'all': np.ones(len(Mv), bool), 'coldpos': (npos > 0) & (ncold == npos), 'warmpos': (npos > 0) & (ncold == 0)}
Wt = pmix.weights(); wpub, wpri = np.asarray(Wt['public'], float), np.asarray(Wt['private'], float)
def wm(d, w): m = (d * w).sum() / w.sum(); s = np.sqrt(((w * (d - m)) ** 2).sum()) / w.sum(); return f'{m:+.5f}+-{s:.5f}'
def split(s, c, M, off):
    nc = np.add.reduceat(c, off); nw = M - nc
    mc = np.add.reduceat(s * c, off) / np.maximum(nc, 1); mw = np.add.reduceat(s * (1 - c), off) / np.maximum(nw, 1)
    part = np.repeat(mc, M) * c + np.repeat(mw, M) * (1 - c); return part, s - part, mc - mw, (nc > 0) & (nw > 0)
print(f'cold share of candidates: val {cold.mean():.3f} test {coldt.mean():.3f} (public {coldt[np.repeat(np.arange(len(Mt)) < 8562, Mt)].mean():.3f}, private {coldt[np.repeat(np.arange(len(Mt)) >= 8562, Mt)].mean():.3f})')
for k, ms in (('W8', W8), ('S16', W8 + N8), ('C3', C3M), ('G3', GROUPS['H:gru'] + GROUPS['H:m02'] + GROUPS['H:m02p70k'])):
    a, b = zv(pool(ms, 'to', 'val')), zv(pool(ms, 'ft', 'val')); s = b - a; e0 = ev(a)
    p1, p2, gap, both = split(s, cold, Mv, ov)
    at, bt = zt(pool(ms, 'to', 'test')), zt(pool(ms, 'ft', 'test')); st_ = bt - at; q1, q2, gapt, botht = split(st_, coldt, Mt, ot)
    print(f'\n{k}: cold-minus-warm mean shift per feed (z units): val {gap[both].mean():+.4f}+-{gap[both].std()/np.sqrt(both.sum()):.4f} | test {gapt[botht].mean():+.4f}; variance share of the between-group part: val {p1.var()/s.var():.3f} test {q1.var()/st_.var():.3f}')
    for nm, x in (('full shift (= FT pool)', s), ('(i) cold/warm offset only', p1), ('(ii) within-group only', p2), ('0.5 full (= file)', 0.5 * s), ('0.5 (i)', 0.5 * p1), ('0.5 (ii)', 0.5 * p2), ('(i) + 0.5 (ii)', p1 + 0.5 * p2), ('0.5 (i) + 0.25 (ii)', 0.5 * p1 + 0.25 * p2)):
        d = ev(a + x) - e0
        print(f'  {nm:26s} ' + ' | '.join(f'{sn} {f(d[m])}' for sn, m in SL.items()) + f' | public mix {wm(d, wpub)} | private mix {wm(d, wpri)}')
    # one-parameter global cold offset, nested by user halves (20 splits x 2 directions)
    grid = np.round(np.arange(0, 0.31, 0.03), 2); E = np.array([ev(a + dlt * cold) - e0 for dlt in grid])
    rng = np.random.default_rng(0); us = np.unique(uv); gains = []; picks = []
    for r in range(20):
        h = dict(zip(us, rng.integers(0, 2, len(us)))); A = np.array([h[u] for u in uv]) == 0
        for tr, te in ((A, ~A), (~A, A)):
            i = int(np.argmax(E[:, tr].mean(1))); picks.append(grid[i]); gains.append(E[i, te].mean())
    i = int(np.argmax(E.mean(1)))
    print(f'  global cold offset z_TO + delta*cold: in-sample best delta {grid[i]:.2f} gain {f(E[i])} (coldpos {f(E[i][SL["coldpos"]])}, warmpos {f(E[i][SL["warmpos"]])}); nested picks mean {np.mean(picks):.3f}, nested gain {np.mean(gains):+.5f} (sd over splits {np.std(gains):.5f})')
    j = int(np.argmin(np.abs(grid - gap[both].mean())))
    print(f'  offset at the FT-implied delta {grid[j]:.2f}: {f(E[j])} | public mix {wm(E[j], wpub)} | private mix {wm(E[j], wpri)}; curve: ' + ' '.join(f'{g:.2f}:{E[q].mean():+.5f}' for q, g in enumerate(grid)))
