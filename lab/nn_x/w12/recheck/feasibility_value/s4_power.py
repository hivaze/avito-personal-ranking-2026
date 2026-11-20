"""Power / false-pass arithmetic of the screens that the proposals would be read with, and base-rate priors (normal model, CPU, seconds).
Inputs are MEASURED SEs quoted in the lens reports:
  pool_composition#2: per-feed SE of the asymptote delta, 4 seeds 0.00065 val / 0.00037 holdout; 2 seeds 0.0011 / 0.0006
  kd_reprice#1: run-to-run sd of one 2-seed Qhat 0.0006 val / 0.0003 holdout (seed noise, per arm)
  8-vs-8 pool delta (W8 - N8): 0.00049 val / 0.00028 holdout (per feed)
  CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/recheck/feasibility_value/s4_power.py"""
from math import erf, sqrt

Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
up = lambda thr, mu, sd: 1 - Phi((thr - mu) / sd)  # P(X >= thr), X ~ N(mu, sd)


def tot(per_feed, seed_per_arm):  # SE of an arm-minus-reference asymptote delta: per-feed part + seed noise of both arms
    return sqrt(per_feed ** 2 + 2 * seed_per_arm ** 2)


rules = {}
# (A) 2-seed Qhat extension rule (E_kd_predeclared 20:21Z item 3): Qhat holdout >= +0.0004 AND val >= 0
sh, sv = tot(0.0006, 0.0003), tot(0.0011, 0.0006)
rules['2-seed Qhat screen (hold >= +0.0004, val >= 0)'] = lambda d, sh=sh, sv=sv: up(0.0004, d, sh) * up(0.0, d, sv)
# (B) 4-seed asymptote "ceiling shift" rule: holdout >= +2 per-feed SE (0.00074) AND val >= 0 (the @last clause is ignored: optimistic)
sh4, sv4 = tot(0.00037, 0.0003 / sqrt(2)), tot(0.00065, 0.0006 / sqrt(2))
rules['4-seed asymptote rule (hold >= +0.00074, val >= 0)'] = lambda d, sh=sh4, sv=sv4: up(0.00074, d, sh) * up(0.0, d, sv)
# (C) 8-seed pool confirmation vs W8: holdout >= +0.0004 AND val >= 0 (seed noise of an 8-pool about 0.00012 hold / 0.00018 val: sd of random 8-subsets)
sh8, sv8 = sqrt(0.00028 ** 2 + 2 * 0.00012 ** 2), sqrt(0.00049 ** 2 + 2 * 0.00018 ** 2)
rules['8-seed pool confirmation (hold >= +0.0004, val >= 0)'] = lambda d, sh=sh8, sv=sv8: up(0.0004, d, sh) * up(0.0, d, sv)
# (D) adopted evidence rule: both signs >= 0 and (z_val + z_hold)/sqrt(2) >= 2, 8-vs-8 pools, per-feed SEs
def pooled(d, sh=0.00028, sv=0.00049, n=200000):
    import random
    random.seed(1); k = 0
    for _ in range(n):
        xh = random.gauss(d, sqrt(sh ** 2 + 2 * 0.00012 ** 2)); xv = random.gauss(d, sqrt(sv ** 2 + 2 * 0.00018 ** 2))
        k += (xh >= 0 and xv >= 0 and (xh / sh + xv / sv) / sqrt(2) >= 2)
    return k / n
rules['8-vs-8 pooled-z rule (both >= 0, z >= 2)'] = pooled
# (E) the same pooled rule with a 3-rotation holdout (89,794 eligible feeds; per-feed SE x sqrt(30000/89794)) and 12 runs per arm
rules['pooled-z rule, 3-rotation holdout, 12 runs/arm'] = lambda d: pooled(d, sh=0.00028 * sqrt(30000 / 89794))
print(f'SE used (arm minus reference): 2-seed Qhat hold {sh:.5f} val {sv:.5f} | 4-seed asymptote hold {sh4:.5f} val {sv4:.5f} | 8-pool hold {sh8:.5f} val {sv8:.5f}')
print(f'\n{"rule":52s} | P(pass) at true shift 0 | +0.0003 | +0.0005 | +0.0007 (width-sized) | +0.0010')
for nm, f in rules.items():
    print(f'{nm:52s} | {f(0.0):22.3f} | {f(0.0003):7.3f} | {f(0.0005):7.3f} | {f(0.0007):21.3f} | {f(0.001):7.3f}')

# base rates -> priors (Beta(a, b) with mean 0.025 and weight 20 = the 5-of-200 project base rate)
a0, b0 = 0.5, 19.5
print('\nprior P(an arm truly moves the pool by >= +0.0005), Beta(0.5, 19.5) updated with today\'s KD record:')
for nm, s, n in (('lever closed under hard labels, re-tested under KD (width 192 worked; drop0, e20, id, hl2, ff4, w256, b3 failed)', 1, 8),
                 ('KD target / teacher variant (kd_w 0.3 / 0.8 / 1.0, t25, gen 2 x2, cross-fit x2, pop1-view students at 50/50)', 0, 9),
                 ('all of today\'s KD arms together (KD itself + width of about 19)', 2, 19)):
    print(f'  {nm}: {s} of {n} -> {(a0 + s) / (a0 + b0 + n):.3f}')

# expected yield of a wave of K arms with prior p and true size d, screened by rule (B) then confirmed by rule (C)
pB0, pB1 = rules['4-seed asymptote rule (hold >= +0.00074, val >= 0)'](0.0), rules['4-seed asymptote rule (hold >= +0.00074, val >= 0)'](0.0005)
pC0, pC1 = rules['8-seed pool confirmation (hold >= +0.0004, val >= 0)'](0.0), rules['8-seed pool confirmation (hold >= +0.0004, val >= 0)'](0.0005)
for K, p in ((6, 0.04), (15, 0.04), (30, 0.03)):
    tp = K * p * pB1 * pC1; fp = K * (1 - p) * pB0 * pC0
    print(f'wave of {K} arms, prior {p}: screen passes true {K * p * pB1:.2f} / false {K * (1 - p) * pB0:.2f}; after the 8-seed confirmation '
          f'true {tp:.3f} / false {fp:.3f}; P(at least one confirmed real +0.0005) about {1 - (1 - p * pB1 * pC1) ** K:.3f}')
