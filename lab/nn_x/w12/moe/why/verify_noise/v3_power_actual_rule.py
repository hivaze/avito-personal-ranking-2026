"""verify_noise V3 (numpy only). The noise analyst's N4 power table models 'PASS = both >= 0 and pooled z >= 2' = the OLD arm_read8 'RULE:'
line, which RULE_predeclared.txt says is NOT this screen's decision. The screen rule (screen_rules.py decide):
  PROMOTE iff 4-pool val delta >= +0.0003 AND 4-pool holdout delta >= 0 AND perm8 p_perm <= 0.2
  (p_perm = share of the 16 exact within-seed swap patterns, identity included, whose pooled z >= the observed one); KILL iff val < 0.
Parametric Monte Carlo of the actual rule, including the 16-pattern swap test:
  member value = delta (arm only) + sigma_m * N(0,1) per split; pool value = member mean + composition term g(S);
  g: (i) iid per 4-set, sd sigma_g; (ii) pairwise model g(S) = mean of c_ij over the 6 pairs in S (c_ij iid) - sets that share members share terms.
  sigma_m and the null 4-vs-4 sd are the noise analyst's pooled estimates (n4_power.txt): member 0.00048 / 0.00027, 4v4 delta 0.00055 / 0.00028;
  sigma_g solves 2 sigma_m^2 / 4 + 2 sigma_g^2 = sd_4v4^2. val/holdout noise independent (analyst: corr +0.04). SE in z: 0.00057 / 0.00032.
Also: the conditional 'P(all three val < 0 | Ra - Rb)' by quadrature with alternative sigma_p values (sensitivity)."""
import numpy as np, itertools, json
from scipy.stats import norm
from pathlib import Path

OUT = Path('<repo>/lab/nn_x/w12/moe/why/verify_noise')
out = []
P = lambda *a: (print(*a, flush=True), out.append(' '.join(str(x) for x in a)))
SM = dict(v=0.00048, h=0.00027); S44 = dict(v=0.00055, h=0.00028); SE = dict(v=0.00057, h=0.00032)
SG = {k: np.sqrt((S44[k] ** 2 - 2 * SM[k] ** 2 / 4) / 2) for k in 'vh'}
P(f'sigma_g (composition, per 4-set): val {SG["v"]:.5f} holdout {SG["h"]:.5f}')
pats = np.array([[(k >> i) & 1 for i in range(4)] for k in range(16)], bool)    # flip indicators, row 0 = identity
pairs8 = list(itertools.combinations(range(8), 2)); pidx = {p: i for i, p in enumerate(pairs8)}
# members 0-3 = arm seeds, 4-7 = reference seeds; pattern s: X = {i if not s_i else 4+i}, Y = complement
X = np.array([[i + 4 if s[i] else i for i in range(4)] for s in pats]); Y = np.array([[i if s[i] else i + 4 for i in range(4)] for s in pats])
setsX = [tuple(sorted(x)) for x in X]; setsY = [tuple(sorted(y)) for y in Y]
allsets = sorted(set(setsX) | set(setsY)); sidx = {s: i for i, s in enumerate(allsets)}
PX = np.zeros((len(allsets), len(pairs8)))
for s in allsets:
    for p in itertools.combinations(s, 2):
        PX[sidx[s], pidx[p]] = 1 / 6


def sim(delta_v, delta_h, N, model, rng):
    zs = 0
    D = {}
    for k, d in (('v', delta_v), ('h', delta_h)):
        mem = rng.standard_normal((N, 8)) * SM[k]; mem[:, :4] += d
        if model == 'iid':
            g = rng.standard_normal((N, len(allsets))) * SG[k]
        else:
            c = rng.standard_normal((N, len(pairs8))) * SG[k] * np.sqrt(6)
            g = c @ PX.T
        mx = mem[:, X].mean(2) + g[:, [sidx[s] for s in setsX]]
        my = mem[:, Y].mean(2) + g[:, [sidx[s] for s in setsY]]
        D[k] = mx - my                                     # (N, 16) pattern deltas, column 0 = observed
        zs = zs + D[k] / SE[k]
    zp = zs / np.sqrt(2)
    p = (zp >= zp[:, [0]] - 1e-15).mean(1)
    dv, dh = D['v'][:, 0], D['h'][:, 0]
    promote = (dv >= 0.0003) & (dh >= 0) & (p <= 0.2)
    hold = (dv >= 0.0003) & (dh >= 0) & (p > 0.2)
    kill = dv < 0
    oldpass = (dv >= 0) & (dh >= 0) & (zp[:, 0] >= 2)
    return promote.mean(), hold.mean(), kill.mean(), oldpass.mean(), (p <= 0.2).mean(), np.sqrt((dv ** 2).mean() - dv.mean() ** 2)


rng = np.random.default_rng(1119)
res = {}
for model in ('iid', 'pairwise'):
    P(f'== model {model}: P(outcome) of the ACTUAL screen rule, 4 seeds, effect delta on both splits (200k draws); last col = old pooled-z rule')
    P(f'  {"delta":>8s} {"PROMOTE":>8s} {"HOLD":>6s} {"KILL":>6s} {"p<=0.2":>7s} {"sd(dv)":>8s} | old PASS')
    for d in (0.0, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002):
        pr, ho, ki, op, pp, sdv = sim(d, d, 200000, model, rng)
        res[f'{model}_{d}'] = dict(promote=pr, hold=ho, kill=ki, oldpass=op)
        P(f'  {d:+.5f} {pr:8.3f} {ho:6.3f} {ki:6.3f} {pp:7.3f} {sdv:8.5f} | {op:.3f}')
    grid = np.arange(0, 0.004, 0.0001); pg = np.array([sim(d, d, 30000, model, rng)[0] for d in grid])
    P(f'  delta for 50% PROMOTE {grid[np.argmax(pg >= 0.5)]:.4f}, for 80% PROMOTE {grid[np.argmax(pg >= 0.8)] if (pg >= 0.8).any() else float("nan"):.4f}; max P(PROMOTE) on grid {pg.max():.3f}')
    # val-only effect (holdout 0) and holdout-only effect
    for dv, dh in ((0.0005, 0.0), (0.001, 0.0), (0.0, 0.0005)):
        pr, ho, ki, op, pp, _ = sim(dv, dh, 100000, model, rng)
        P(f'  val {dv:+.4f} / holdout {dh:+.4f}: PROMOTE {pr:.3f} HOLD {ho:.3f} KILL {ki:.3f} | old PASS {op:.3f}')

P('')
P('== conditional P(all three val deltas < 0 | Ra - Rb = +0.00138) under the null, quadrature, sigma_p sensitivity')
for sp in (0.00030, 0.000368, 0.00039, 0.00045, 0.00055):
    e = np.linspace(0.00069 - 8 * sp / np.sqrt(2), 0.00069 + 8 * sp / np.sqrt(2), 20001)
    w = norm.pdf(e, 0.00069, sp / np.sqrt(2)); w /= w.sum()
    P(f'  sigma_p {sp:.6f}: P(all three < 0) {np.sum(w * norm.cdf(e / sp) ** 3):.3f}; P(B <= -0.00144) {norm.cdf((-0.00144 + 0.00069) / (sp * np.sqrt(1.5))):.3f}')
json.dump(res, open(OUT / 'v3_power_actual_rule.json', 'w'))
open(OUT / 'v3_power_actual_rule.txt', 'w').write('\n'.join(out) + '\n')
