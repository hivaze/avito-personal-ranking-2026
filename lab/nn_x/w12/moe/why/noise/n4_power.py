"""why/noise N4: noise calibration and power of the 4-seed screen.
(1) Within-recipe null: for every recipe with 8 seeds (reference kd05w192 + 7 historical w192 arms), the 35 disjoint 4-vs-4 splits give
    null pool deltas D (val, holdout) and their per-feed SEs -> seed sd of a 4-pool sigma_p, calibration of the per-feed z (sd of null z),
    corr(D_val, D_hold). Single-member seed sd from the 8 seeds.
(2) Power of the pre-declared rule at 4 seeds (PASS: both >= 0 and pooled z >= 2; EXTEND: both >= 0 and pooled z >= 1.6) for a true
    effect delta on both splits, by Monte Carlo with the estimated noise (unpaired: same-seed pairing buys ~nothing, see N3).
(3) 'All three arms negative on val' under the null with a SHARED reference: unconditional, and conditional on the observed reference draw
    (Ra - Rb = +0.00138 val / +0.00001 holdout), for the signs and for the observed mean of the three deltas."""
import itertools, json
from nz_common import *
from scipy.stats import norm

m = meta()
out = []
P = lambda *a: (print(*a, flush=True), out.append(' '.join(str(x) for x in a)))
REC = ['kd05w192', 'kd05t25w192', 'kd08w192', 'kwwd0', 'kwwd01', 'kwwd05', 'kwaux0', 'kd05w192ps']
splits = [c for c in itertools.combinations(range(8), 4) if 0 in c]
rows = []
P('== (1) within-recipe null: 35 disjoint 4-vs-4 splits per 8-seed recipe (pool deltas, per-feed SE)')
mem_sd = {'val': [], 'hold': []}
for r in REC:
    vs = [nm(r, s) for s in S8]
    pc = {}
    for c in itertools.combinations(range(8), 4):
        pc[c] = pool_ndcg([vs[i] for i in c])
    Dv, Dh, Sv, Sh = [], [], [], []
    for c in splits:
        cc = tuple(sorted(set(range(8)) - set(c)))
        dv = pc[c][0] - pc[cc][0]; dh = pc[c][1] - pc[cc][1]
        Dv.append(dv.mean()); Dh.append(dh.mean()); Sv.append(se(dv)); Sh.append(se(dh))
    Dv, Dh, Sv, Sh = map(np.array, (Dv, Dh, Sv, Sh))
    mv = np.array([member_ndcg(v)[0].mean() for v in vs]); mh = np.array([member_ndcg(v)[1].mean() for v in vs])
    mem_sd['val'].append(mv.std(ddof=1)); mem_sd['hold'].append(mh.std(ddof=1))
    rows.append(dict(rec=r, Dv=Dv, Dh=Dh, Sv=Sv, Sh=Sh))
    P(f'  {r:12s}: rms D val {np.sqrt((Dv ** 2).mean()):.5f} (mean per-feed SE {Sv.mean():.5f}, sd null z {np.sqrt(((Dv / Sv) ** 2).mean()):.2f})'
      f' | holdout {np.sqrt((Dh ** 2).mean()):.5f} (SE {Sh.mean():.5f}, sd null z {np.sqrt(((Dh / Sh) ** 2).mean()):.2f}) | corr(Dv,Dh) {np.corrcoef(Dv, Dh)[0, 1]:+.2f}'
      f' | member sd val {mv.std(ddof=1):.5f} hold {mh.std(ddof=1):.5f}')
Dv = np.concatenate([x['Dv'] for x in rows]); Dh = np.concatenate([x['Dh'] for x in rows])
Sv = np.concatenate([x['Sv'] for x in rows]); Sh = np.concatenate([x['Sh'] for x in rows])
sDv, sDh = np.sqrt((Dv ** 2).mean()), np.sqrt((Dh ** 2).mean())
# uncertainty of the pooled rms: each recipe ~ 7 effective df (8 seeds); use the spread of per-recipe rms
rv_ = np.array([np.sqrt((x['Dv'] ** 2).mean()) for x in rows]); rh_ = np.array([np.sqrt((x['Dh'] ** 2).mean()) for x in rows])
rc = np.mean([np.corrcoef(x['Dv'], x['Dh'])[0, 1] for x in rows])
zsd_v, zsd_h = np.sqrt(((Dv / Sv) ** 2).mean()), np.sqrt(((Dh / Sh) ** 2).mean())
sp_v, sp_h = sDv / np.sqrt(2), sDh / np.sqrt(2)
P(f'  POOLED over {len(REC)} recipes: sd of a null 4-vs-4 delta val {sDv:.5f} (recipe spread {rv_.std(ddof=1) / np.sqrt(len(rv_)):.5f}) holdout {sDh:.5f}'
  f' ({rh_.std(ddof=1) / np.sqrt(len(rh_)):.5f}) -> seed sd of one 4-pool sigma_p val {sp_v:.5f} holdout {sp_h:.5f}; per-feed SE val {Sv.mean():.5f} holdout {Sh.mean():.5f};'
  f' sd of null z val {zsd_v:.2f} holdout {zsd_h:.2f}; mean corr(Dv, Dh) {rc:+.2f}')
P(f'  single-member seed sd (8 seeds, mean over recipes): val {np.mean(mem_sd["val"]):.5f} holdout {np.mean(mem_sd["hold"]):.5f}')
# empirical null sign pattern from fake arm vs fake reference
P(f'  fake arm vs fake ref (all 8 recipes x 35 splits, both orientations): share val<0 & hold>0 {np.mean(np.r_[(Dv < 0) & (Dh > 0), (-Dv < 0) & (-Dh > 0)]):.2f};'
  f' share |z_val| >= 1.5 {np.mean(np.abs(Dv / Sv) >= 1.5):.2f} (normal 0.13); share |z_val| >= 2.5 {np.mean(np.abs(Dv / Sv) >= 2.5):.3f} (normal 0.012)')

# ---- (2) power of the rule
P('')
P('== (2) power of the pre-declared 4-seed rule (Monte Carlo, 200k draws; true effect delta on BOTH splits; unpaired noise from (1);'
  ' SEs used in the z = the screen\'s per-feed SEs 0.00057 / 0.00032)')
rng = np.random.default_rng(20261119)
N = 200000
SEv, SEh = 0.00057, 0.00032
cov = np.array([[sDv ** 2, rc * sDv * sDh], [rc * sDv * sDh, sDh ** 2]])
E = rng.multivariate_normal([0, 0], cov, N)
pw = {}
P(f'  {"delta":>8s} {"P(val>=0)":>10s} {"P(PASS)":>8s} {"P(EXTEND|PASS)":>15s} {"P(val z<=-1.5)":>15s} | 8 seeds (sigma/sqrt2, SE/sqrt2): P(PASS)')
for d in (0.0, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003):
    dv, dh = d + E[:, 0], d + E[:, 1]
    zp = (dv / SEv + dh / SEh) / np.sqrt(2)
    ps = np.mean((dv >= 0) & (dh >= 0) & (zp >= 2)); pe = np.mean((dv >= 0) & (dh >= 0) & (zp >= 1.6))
    dv8, dh8 = d + E[:, 0] / np.sqrt(2), d + E[:, 1] / np.sqrt(2)
    zp8 = (dv8 / (SEv / np.sqrt(2)) + dh8 / (SEh / np.sqrt(2))) / np.sqrt(2)
    ps8 = np.mean((dv8 >= 0) & (dh8 >= 0) & (zp8 >= 2))
    pw[d] = dict(pass4=float(ps), extend4=float(pe), pass8=float(ps8))
    P(f'  {d:+.5f} {np.mean(dv >= 0):10.2f} {ps:8.3f} {pe:15.3f} {np.mean(dv / SEv <= -1.5):15.3f} | {ps8:.3f}')
# MDE: delta with 80% P(PASS) at 4 / 8 seeds
for lab, sc in (('4 seeds', 1.0), ('8 seeds', 1 / np.sqrt(2))):
    grid = np.arange(0, 0.006, 0.00005)
    pp = []
    for d in grid:
        dv, dh = d + E[:50000, 0] * sc, d + E[:50000, 1] * sc
        zp = (dv / (SEv * sc) + dh / (SEh * sc)) / np.sqrt(2)
        pp.append(np.mean((dv >= 0) & (dh >= 0) & (zp >= 2)))
    pp = np.array(pp)
    P(f'  {lab}: delta for 50% P(PASS) {grid[np.argmax(pp >= 0.5)]:.5f}, for 80% P(PASS) {grid[np.argmax(pp >= 0.8)]:.5f} (both splits shifted by delta);'
      f' val-only 2-sided 5%/80% MDE {2.80 * sDv * sc:.5f}, holdout-only {2.80 * sDh * sc:.5f}')
# seeds needed for a +0.0003 effect (the size of the best recent architecture changes) at 80% via pooled z only
for d in (0.0002, 0.0003, 0.0005):
    k = 4
    while k < 400:
        sc = np.sqrt(4 / k)
        dv, dh = d + E[:50000, 0] * sc, d + E[:50000, 1] * sc
        zp = (dv / (SEv * sc) + dh / (SEh * sc)) / np.sqrt(2)
        if np.mean((dv >= 0) & (dh >= 0) & (zp >= 2)) >= 0.8:
            break
        k += 4
    P(f'  seeds per arm (and per reference) for 80% P(PASS) at delta {d:+.4f} on both splits (noise ~ 1/sqrt(seeds), pool-size gain ignored): ~{k}')

# ---- (3) all three negative with a shared reference
P('')
P('== (3) null with a SHARED reference: D_k = e_k - e_R, e ~ N(0, sigma_p^2) iid (pool seed noise from (1)); 1M draws')
N3 = 1000000
obs_v = np.array([-0.00086, -0.00144, -0.00069]); obs_h = np.array([0.00020, 0.00029, 0.00040])
cov1 = np.array([[sp_v ** 2, rc * sp_v * sp_h], [rc * sp_v * sp_h, sp_h ** 2]])
eR = rng.multivariate_normal([0, 0], cov1, N3); eK = rng.multivariate_normal([0, 0], cov1, (N3, 3))
Dk_v = eK[..., 0] - eR[:, [0]]; Dk_h = eK[..., 1] - eR[:, [1]]
allneg = (Dk_v < 0).all(1); allpos_h = (Dk_h > 0).all(1)
P(f'  unconditional: P(all three val < 0) {allneg.mean():.3f} (independent refs: 0.125); P(all three val<0 AND all three holdout>0) {(allneg & allpos_h).mean():.3f}'
  f' (independent: {0.125 ** 2:.4f}); P(mean val delta <= {obs_v.mean():+.5f}) {(Dk_v.mean(1) <= obs_v.mean()).mean():.3f};'
  f' P(mean holdout delta >= {obs_h.mean():+.5f}) {(Dk_h.mean(1) >= obs_h.mean()).mean():.3f}; corr(D_A, D_B) val {np.corrcoef(Dk_v[:, 0], Dk_v[:, 1])[0, 1]:+.2f}')
# conditional on the observed reference split Ra - Rb (an independent second reference of the same recipe)
gv, gh = 0.00138, 0.00001
eRb = rng.multivariate_normal([0, 0], cov1, N3)
# importance: keep draws with Ra - Rb close to the observed values (kernel on both splits)
w = np.exp(-0.5 * (((eR[:, 0] - eRb[:, 0]) - gv) / (0.05 * sDv)) ** 2 - 0.5 * (((eR[:, 1] - eRb[:, 1]) - gh) / (0.05 * sDh)) ** 2)
w /= w.sum()
ess = 1 / np.sum(w ** 2)
P(f'  conditional on the observed Ra - Rb (val +0.00138, holdout +0.00001; kernel-weighted, ESS {ess:.0f}): E[e_Ra] val {np.sum(w * eR[:, 0]):+.5f} holdout {np.sum(w * eR[:, 1]):+.5f}')
P(f'    P(all three val < 0 | Ra-Rb) {np.sum(w * allneg):.3f}; P(all three holdout > 0 | Ra-Rb) {np.sum(w * allpos_h):.3f}; P(both patterns) {np.sum(w * (allneg & allpos_h)):.3f}')
P(f'    P(mean val delta <= {obs_v.mean():+.5f} | Ra-Rb) {np.sum(w * (Dk_v.mean(1) <= obs_v.mean())):.3f}; P(mean holdout delta >= {obs_h.mean():+.5f} | Ra-Rb) {np.sum(w * (Dk_h.mean(1) >= obs_h.mean())):.3f}')
# arm-level z of the three arms against the best available null, the per-arm deltas vs Rbar70 (n1)
n1 = json.load(open(OUT / 'n1_shared_ref.json'))
dvb = np.array([n1[k]['rows']['Rbar70'][0] for k in 'ABC']); dhb = np.array([n1[k]['rows']['Rbar70'][2] for k in 'ABC'])
# Var(arm4 - Rbar70) under the null: arm pool sigma_p^2 + Var(Rbar70) ~ sigma_p^2 / 2 (Rbar70 ~ 8 seeds)
sd1v, sd1h = sp_v * np.sqrt(1.5), sp_h * np.sqrt(1.5)
P(f'  arms vs Rbar70 with seed-noise sd (sigma_p*sqrt(1.5) = val {sd1v:.5f}, holdout {sd1h:.5f}): z val ' + ' '.join(f'{k} {x / sd1v:+.2f}' for k, x in zip('ABC', dvb))
  + ' | z holdout ' + ' '.join(f'{k} {x / sd1h:+.2f}' for k, x in zip('ABC', dhb)))
sdm_v = sp_v * np.sqrt(1 / 3 + 1 / 2); sdm_h = sp_h * np.sqrt(1 / 3 + 1 / 2)
P(f'  mean of the three arms vs Rbar70: val {dvb.mean():+.5f} (null sd {sdm_v:.5f}, z {dvb.mean() / sdm_v:+.2f}) holdout {dhb.mean():+.5f} (null sd {sdm_h:.5f}, z {dhb.mean() / sdm_h:+.2f})')
json.dump(dict(sDv=sDv, sDh=sDh, sp_v=sp_v, sp_h=sp_h, rc=rc, zsd_v=zsd_v, zsd_h=zsd_h, power={str(k): v for k, v in pw.items()}),
          open(OUT / 'n4_power.json', 'w'))
open(OUT / 'n4_power.txt', 'w').write('\n'.join(out) + '\n')
