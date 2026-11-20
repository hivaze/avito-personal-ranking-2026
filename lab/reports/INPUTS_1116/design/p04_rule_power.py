"""DESIGN p04: Monte Carlo power of the pre-declared rules (normal approximation; SEs from out/p01_power.txt part (a), stated below).
Stage A = hard-label d128 members, 8 with the bundle vs 8 without. Stage B = distilled wide students, 8 with vs the 16 stored.
Stage F = bundle 2 as fine-tune extras on 8 all-data wide members, paired with their stored production fine-tunes.
z statistics are member-mean differences over their SE; the rule also asks the pool-level deltas (8-pool minus 8-pool) to be >= 0 on val and holdout. val and holdout are different feeds -> independent; the private-mix read is a
reweighting of val (corr 0.7 assumed), the pool-level read correlates with the member-level read (0.8 assumed). Null sd of z = 1
(the record's A/A null of the pooled-z rule is 0.80, so the false-pass rates printed here are upper bounds)."""
import numpy as np
from pathlib import Path
rng = np.random.default_rng(1116); N = 400000
out = open(Path(__file__).resolve().parent/'out/p04_rule_power.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n')
POOL_RATIO = dict(v=0.00036 / 0.00052, h=0.00029 / 0.00042)   # member-mean SE / pool SE (8 + 8): pool z mean = member z mean x this ratio
def draw(mu_v, mu_p, mu_h, n=N):
    e = rng.standard_normal((n, 6))
    zv = mu_v + e[:, 0]; zp = mu_p + 0.7 * e[:, 0] + np.sqrt(1 - 0.49) * e[:, 1]; zh = mu_h + e[:, 2]
    # pool-level z of the same contrast: same true effect, larger SE, noise correlated 0.8 with the member-level noise
    pv = mu_v * POOL_RATIO['v'] + 0.8 * e[:, 0] + 0.6 * e[:, 3]; ph = mu_h * POOL_RATIO['h'] + 0.8 * e[:, 2] + 0.6 * e[:, 4]
    return zv, zp, zh, pv, ph
def rule(zv, zp, zh, pv, ph):
    pooled = (zv + zh) / np.sqrt(2)
    ok_side = (zv >= 1.0) & (zh >= 1.0) & (zp >= -1.0) & (pv >= 0) & (ph >= 0)
    return pooled, ok_side
def stage(se_v, se_p, se_h, d_v, d_p, d_h, rep_factor):
    """returns P(pass at first read), P(pass after the replicate step), P(closed: pooled z < 1.6 or a side clause fails outside the band)."""
    zv, zp, zh, pv, ph = draw(d_v / se_v, d_p / se_p, d_h / se_h)
    pooled, side = rule(zv, zp, zh, pv, ph)
    p1 = (pooled >= 2.0) & side
    band = (pooled >= 1.6) & (pooled < 2.0) & (zv >= 0) & (zh >= 0)
    # replicate: +50% members per arm -> SE shrinks by rep_factor; combined estimate = weighted mean of old and new
    w_old = 1 / 1.5; zv2n, zp2n, zh2n, pv2n, ph2n = draw(d_v / (se_v * np.sqrt(2)) , d_p / (se_p * np.sqrt(2)), d_h / (se_h * np.sqrt(2)))  # 4 new members: SE x sqrt(2)
    comb = lambda zo, zn, se: (w_old * zo * se + (1 - w_old) * zn * se * np.sqrt(2)) / (se * rep_factor)
    zv2, zp2, zh2 = comb(zv, zv2n, se_v), comb(zp, zp2n, se_p), comb(zh, zh2n, se_h)
    pv2 = (w_old * pv + (1 - w_old) * pv2n * np.sqrt(2)) / rep_factor; ph2 = (w_old * ph + (1 - w_old) * ph2n * np.sqrt(2)) / rep_factor
    pooled2, side2 = rule(zv2, zp2, zh2, pv2, ph2)
    p2 = band & (pooled2 >= 2.0) & side2
    return p1.mean(), (p1 | p2).mean(), band.mean()
def mde50(f, lo=0.0, hi=0.01):
    for _ in range(40):
        mid = (lo + hi) / 2
        if f(mid) >= 0.5: hi = mid
        else: lo = mid
    return hi
def mde(f, lo=0.0, hi=0.01):
    for _ in range(40):
        mid = (lo + hi) / 2
        if f(mid) >= 0.8: hi = mid
        else: lo = mid
    return hi
for INFL in (1.0, 1.17):
    P(f'\n######## SE inflation x{INFL} (1.0 = null SE from the seed variance; 1.17 = what stored contrasts between different input views ran above it: 0.00059 vs 0.00051 val, 0.00036 vs 0.00030 holdout, out/p01_power.txt c2 vs a) ########')
    P('== Stage A: hard labels, 8 + 8 members. SE member mean: val 0.00036, private mix 0.00052 [out/p01_power.txt (a), k = 8]; holdout clause on the 15,225 holdout feeds of Sep 12-19 (the feeds where the lag view exists): 0.00021 x sqrt(30000 / 15225) = 0.00029 ==')
    P('true member-level effect on val | effect at the private mix | effect on the holdout | P(PASS first read) | P(PASS incl. replicate) | P(replicate band)')
    for d in (0.0, 0.0003, 0.0005, 0.001, 0.002):
        for rp, rh, lab in ((1.0, 1.0, 'same on all three'), (0.5, 0.7, 'private x0.5, holdout x0.7'), (0.5, 0.4, 'private x0.5, holdout x0.4')):
            a, b, c = stage(0.00036 * INFL, 0.00052 * INFL, 0.00029 * INFL, d, d * rp, d * rh, np.sqrt(8 / 12))
            P(f'{d:+.4f} | {lab} | {a:.3f} | {b:.3f} | {c:.3f}')
    P('\n== Stage B: distilled wide students, 8 with the bundle vs the 16 stored. SE member mean: val 0.00026, private mix 0.00037 [0.866 x the k = 8 values of (a)], holdout subset 0.00021 ==')
    P('hard-label effect | attenuation by the blind teacher | P(PASS first read) | P(PASS incl. replicate)')
    for d in (0.0, 0.0005, 0.001, 0.002):
        for att in (1.0, 0.5):
            a, b, c = stage(0.00026 * INFL, 0.00037 * INFL, 0.00021 * INFL, d * att, d * att * 0.5, d * att * 0.7, np.sqrt(8 / 12))
            P(f'{d:+.4f} | {att} | {a:.3f} | {b:.3f}   (private x0.5, holdout x0.7)')
    P('\n== smallest true effect with power 0.8 (first read, effect equal on the witnesses used) ==')
    P(f'Stage A, val alone at 2 SE: {2.84 * 0.00036 * INFL:.5f}; holdout subset alone at 2 SE: {2.84 * 0.00029 * INFL:.5f}; private mix alone at 2 SE: {2.84 * 0.00052 * INFL:.5f}')
    P(f'Stage A, full rule, same effect on val and holdout: {mde(lambda d: stage(0.00036 * INFL, 0.00052 * INFL, 0.00029 * INFL, d, d, d, 1)[0]):.5f}')
    P(f'Stage A, full rule, holdout effect x0.7, private x0.5: {mde(lambda d: stage(0.00036 * INFL, 0.00052 * INFL, 0.00029 * INFL, d, d * 0.5, d * 0.7, 1)[0]):.5f}')
    P(f'Stage A, full rule, holdout effect x0.4, private x0.5: {mde(lambda d: stage(0.00036 * INFL, 0.00052 * INFL, 0.00029 * INFL, d, d * 0.5, d * 0.4, 1)[0]):.5f}')
    P(f'Stage A, power 0.5 instead of 0.8, holdout x0.7, private x0.5: {mde50(lambda d: stage(0.00036 * INFL, 0.00052 * INFL, 0.00029 * INFL, d, d * 0.5, d * 0.7, 1)[0]):.5f}')
    P(f'Stage B, full rule, holdout effect x0.7, private x0.5 (effect = what the student keeps): {mde(lambda d: stage(0.00026 * INFL, 0.00037 * INFL, 0.00021 * INFL, d, d * 0.5, d * 0.7, 1)[0]):.5f}')
P('\n== Stage F: fine-tune extras, 8 all-data wide members, arm vs stored control, FILE form 0.5 z(FT) + 0.5 z(TO) ==')
P('record (ledger 2026-11-16T02:38Z, Q4 MIXFT, same layout): file-level SE val 0.00022, private mix 0.00035, val M<50 0.00040')
P('rule F: private-mix z >= +1.5 AND val z >= +1.0 AND val M<50 z >= -1.0. corr(val, private) 0.7; M<50 treated as independent of the two')
for d in (0.0, 0.0002, 0.0003, 0.0005, 0.001):
    e = rng.standard_normal((N, 3)); zv = d / 0.00022 + e[:, 0]; zp = d / 0.00035 + 0.7 * e[:, 0] + np.sqrt(0.51) * e[:, 1]; zm = d / 0.00040 + e[:, 2]
    P(f'true file-level effect {d:+.4f}: P(PASS) {((zp >= 1.5) & (zv >= 1.0) & (zm >= -1.0)).mean():.3f} | variant val z >= 2.0, private z >= 1.0: {((zp >= 1.0) & (zv >= 2.0) & (zm >= -1.0)).mean():.3f}')
out.close()
