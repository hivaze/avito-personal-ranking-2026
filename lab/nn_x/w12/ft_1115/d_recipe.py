"""ft_1115 step D.
 D1  recipe-level curve of final NDCG against w_ft: four DISJOINT 8-member wide pools (held-out students s951-954, s955-958; full-data students s951-954, s955-958),
     each built from member files exactly like the stored branches (member-equal mean of per-feed rank-z; file = w z(FT) + (1 - w) z(TO)); the mean of the four per-feed
     deltas is the least selected read of 'what w_ft does to an 8-wide-student file' (the pools share the FT shift, so the SE shrinks less than 1/2).
 D2  fine-tune weight of the popularity-light members only: file = 0.5 z[(1-a) z(B_ft) + a z(P_ft)] + 0.5 z[(1-b) z(B_to) + b z(P_to)], grid a, b in {0, 0.1, 0.2, 0.3, 0.4};
     the scored form is a = b = 0.2. Nested pick of (a, b) by user-grouped folds. W8P20 and FD16P20.
 D3  how much of the val-to-public gap of the FT half is the feed mix: paired SE of (pmix-public - plain val) for the same per-feed delta.
Output: out/d_recipe.txt"""
from lib import *
out = []
def w_(s=''):
    print(s, flush=True); out.append(s)
GRID = [0.0, 0.15, 0.25, 0.35, 0.5, 0.6, 0.65, 0.75, 0.85, 1.0]; I5 = GRID.index(0.5)
wide = lambda pre, seeds: [f'w5_{pre}_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in seeds]
POOLS = {'W8 (held-out, s951-954)': wide('e10', S4), 'W8b (held-out, s955-958)': wide('e10', range(955, 959)),
         'FD8 (full data, s951-954)': wide('fd10', S4), 'FD8b (full data, s955-958)': wide('fd10', range(955, 959))}
w_('## D1. Four disjoint 8-member wide pools: delta of the file against w_ft 0.5 (paired per feed)')
Dall = []
for name, ms_ in POOLS.items():
    F = zv(mpool(ms_, 'n2cpv3s3', 'val')); T = zv(mpool(ms_, 'trainonly', 'val'))
    e = np.stack([ev(w * F + (1 - w) * T) for w in GRID]); Dall.append(e - e[I5])
    w_(f'{name}: file {e[I5].mean():.5f} FT {e[-1].mean():.5f} TO {e[0].mean():.5f} | ' + ' '.join(f'{w:.2f}:{(e[i] - e[I5]).mean():+.5f}' for i, w in enumerate(GRID)))
Dall = np.array(Dall); Dm = Dall.mean(0)
w_('mean of the four pools (per feed), by view:')
w_('w_ft | plain val | pmix-public | pmix-private | pools with d > 0 (of 4): val / pmix-public / pmix-private')
for i, w in enumerate(GRID):
    npos = [(Dall[:, i].mean(1) > 0).sum(), sum(np.average(Dall[p, i], weights=WPUB) > 0 for p in range(4)), sum(np.average(Dall[p, i], weights=WPRIV) > 0 for p in range(4))]
    w_(f'{w:.2f} | {f(Dm[i])} | {f(Dm[i], WPUB)} | {f(Dm[i], WPRIV)} | {npos[0]} / {npos[1]} / {npos[2]}')
C = np.corrcoef(np.array([Dall[p, GRID.index(0.75)] for p in range(4)])); w_(f'per-feed correlation of d(0.75) between pools: mean off-diagonal {C[np.triu_indices(4, 1)].mean():.3f}; of d(0.0): {np.corrcoef(np.array([Dall[p, 0] for p in range(4)]))[np.triu_indices(4, 1)].mean():.3f}')
np.save(OUT / 'd_recipe_mean_delta.npy', Dm)
# quadratic fit of the mean curve (least squares over the grid) with a user-group bootstrap of the vertex
g = user_groups(); ug, gi = np.unique(g, return_inverse=True); ng = len(ug); rng = np.random.default_rng(7)
X = np.stack([np.array(GRID) - 0.5, (np.array(GRID) - 0.5) ** 2], 1)
def vertex(m):
    p, q = np.linalg.lstsq(X, m, rcond=None)[0]
    return p, q, (0.5 - p / (2 * q) if q < 0 else (1.0 if p > 0 else 0.0))
w_('quadratic fit d(w) = p (w - 0.5) + q (w - 0.5)^2 of the four-pool mean curve; vertex = fitted optimum (clipped to [0, 1]); 2000 user-group bootstrap draws')
for view, wt in (('plain val', np.ones(len(Mv))), ('pmix-public', WPUB), ('pmix-private', WPRIV)):
    num = np.stack([np.bincount(gi, weights=Dm[i] * wt, minlength=ng) for i in range(len(GRID))]); den = np.bincount(gi, weights=wt, minlength=ng)
    p0, q0, v0 = vertex(num.sum(1) / den.sum()); vs = []; ps = []
    for b in range(2000):
        c = np.bincount(rng.integers(0, ng, ng), minlength=ng).astype(np.float64); p, q, v = vertex((num @ c) / (den @ c)); vs.append(np.clip(v, 0, 1)); ps.append(p)
    vs = np.array(vs); ps = np.array(ps)
    w_(f'  {view}: slope at 0.5 p = {p0:+.5f} (bootstrap sd {ps.std():.5f}, P(p > 0) {(ps > 0).mean():.2f}) | curvature q = {q0:+.5f} | vertex {np.clip(v0, 0, 1):.2f} (bootstrap 16-50-84%: {np.percentile(vs, 16):.2f} {np.percentile(vs, 50):.2f} {np.percentile(vs, 84):.2f}; P(vertex > 0.6) {(vs > 0.6).mean():.2f}; P(vertex < 0.4) {(vs < 0.4).mean():.2f})'
       f' | fitted gain of the vertex over 0.5: {(-p0 * p0 / (4 * q0) if q0 < 0 else float("nan")):+.5f}')
w_()
# ------------------------------------------------------------------ D2
w_('## D2. Popularity-light weight per branch (a in the FT branch, b in the train-only branch); scored form a = b = 0.2; delta against it, paired per feed')
AG = [0.0, 0.1, 0.2, 0.3, 0.4]
for name, base, pop in (('W8P20', MEM['W8'], MEM['P1d']), ('FD16P20', MEM['FD16'], MEM['FDP'])):
    Bf, Bt = zv(mpool(base, 'n2cpv3s3', 'val')), zv(mpool(base, 'trainonly', 'val')); Pf, Pt = zv(mpool(pop, 'n2cpv3s3', 'val')), zv(mpool(pop, 'trainonly', 'val'))
    E2 = np.zeros((len(AG), len(AG), len(Mv)))
    for i, a in enumerate(AG):
        zf = zv((1 - a) * Bf + a * Pf)
        for j, b in enumerate(AG):
            E2[i, j] = ev(0.5 * zf + 0.5 * zv((1 - b) * Bt + b * Pt))
    ref = E2[2, 2]; w_(f'{name}: scored form val {ref.mean():.5f} (stored branch file: {ev(0.5 * zv(stored(name + "kd", "val")[0]) + 0.5 * zv(stored(name + "kd", "val")[1])).mean():.5f})')
    for view, wt in (('plain val', None), ('pmix-public', WPUB), ('pmix-private', WPRIV)):
        w_(f'  {view}: rows a (FT branch) = {AG}, columns b (TO branch) = {AG}')
        for i, a in enumerate(AG):
            w_('    a ' + f'{a:.1f}: ' + ' | '.join(f(E2[i, j] - ref, wt) for j in range(len(AG))))
    # nested pick over the 25 cells
    rng2 = np.random.default_rng(3); Dc = (E2 - ref).reshape(25, -1); res = []; picks = []
    for r in range(40):
        fold = rng2.integers(0, 5, ng)[gi]; oos = np.zeros(len(Mv))
        for k in range(5):
            tr = fold != k; j = int(np.argmax(Dc[:, tr].mean(1))); picks.append(j); oos[~tr] = Dc[j, ~tr]
        res.append((oos.mean(), np.average(oos, weights=WPUB), np.average(oos, weights=WPRIV)))
    res = np.array(res); pk = np.bincount(picks, minlength=25).reshape(5, 5)
    w_(f'  nested pick over the 25 cells (5 user-grouped folds x 40 repeats, picked by plain val): out-of-sample val {res[:, 0].mean():+.5f} (sd {res[:, 0].std():.5f}) | pmix-public {res[:, 1].mean():+.5f} | pmix-private {res[:, 2].mean():+.5f}'
       f' | most picked cell a {AG[np.unravel_index(pk.argmax(), pk.shape)[0]]} b {AG[np.unravel_index(pk.argmax(), pk.shape)[1]]} ({pk.max() / pk.sum():.0%} of picks)')
w_()
# ------------------------------------------------------------------ D3
w_('## D3. Is the val-to-public gap of the FT half the feed mix? (FT half = file at w_ft 0.5 minus TO pool; same per-feed delta, weighted against unweighted)')
for name in ('W8', 'W16', 'W8P20', 'FD16P20', 'FD8P20', 'G3s3', 'C3'):
    z = np.load(OUT / f'b_perfeed_{name}.npz'); gr = list(np.round(z['grid'], 2)); d = z['e'][gr.index(0.5)] - z['e'][gr.index(0.0)]
    dp = (WPUB - 1) * d; dq = (WPRIV - 1) * d
    w_(f'{name}: val {f(d)} | pmix-public {f(d, WPUB)} | pmix-private {f(d, WPRIV)} | (pmix-public - val) {dp.mean():+.5f}+-{se(dp):.5f} | (pmix-private - val) {dq.mean():+.5f}+-{se(dq):.5f}')
open(OUT / 'd_recipe.txt', 'w').write('\n'.join(out) + '\n')
