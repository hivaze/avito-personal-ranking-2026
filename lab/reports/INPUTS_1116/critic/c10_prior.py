"""CRITIC c10: what a PASS at stage A would mean under a stated prior [judgement in, arithmetic out]. Normal model of the three z clauses
(SE val 0.00036, lag-view holdout 0.00029, corr of the two z 0.2, holdout effect 0.7 x val), side clauses as a factor taken from c04 (full / z-only).
Prior over the true member-level val effect e: mass 0.30 at 0 (the lens' specificity: 2 of 5 controls convert at about 0), else exponential with mean 0.00041
(so that P(e >= +0.001) = 0.06 before the lag-view losses; printed)."""
import numpy as np
rng = np.random.default_rng(7); n = 2_000_000
e = np.where(rng.random(n) < 0.30, 0.0, rng.exponential(0.00041, n))
a = rng.standard_normal(n); b = 0.2 * a + np.sqrt(1 - 0.04) * rng.standard_normal(n)
zv = e / 0.00036 + a; zh = 0.7 * e / 0.00029 + b; ok = (zv >= 1) & (zh >= 1) & ((zv + zh) / np.sqrt(2) >= 2)
side = 0.85   # share of z-clause passes that also pass the private / pool / last-step / stale-val clauses (c04: 0.79-0.93 over the effect grid)
out = open('<repo>/reports/INPUTS_1116/critic/out/c10_prior.txt', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s); out.write(s + '\n')
P(f'prior: P(e = 0) 0.30; P(e >= +0.0005) {np.mean(e >= 0.0005):.3f}; P(e >= +0.001) {np.mean(e >= 0.001):.3f}; mean {e.mean():.5f}; median {np.median(e):.5f}')
P(f'P(PASS at the first read) = {ok.mean() * side:.3f}  (z clauses {ok.mean():.3f} x side clauses {side})')
P(f'given PASS: P(e = 0) {np.mean(e[ok] == 0):.3f}; P(e < +0.0003) {np.mean(e[ok] < 0.0003):.3f}; P(e >= +0.0005) {np.mean(e[ok] >= 0.0005):.3f}; P(e >= +0.001) {np.mean(e[ok] >= 0.001):.3f}')
P(f'given PASS: mean true effect {e[ok].mean():.5f}; mean ESTIMATED val effect {(zv[ok] * 0.00036).mean():.5f} (overstatement x{(zv[ok] * 0.00036).mean() / e[ok].mean():.2f})')
for t in (0.0, 0.0003, 0.0005, 0.001):
    zv2 = t / 0.00036 + a; zh2 = 0.7 * t / 0.00029 + b; ok2 = (zv2 >= 1) & (zh2 >= 1) & ((zv2 + zh2) / np.sqrt(2) >= 2)
    P(f'true effect {t:+.4f}: P(z clauses) {ok2.mean():.3f}; mean estimated val effect given PASS {(zv2[ok2] * 0.00036).mean():+.5f}; smallest {(zv2[ok2] * 0.00036).min():+.5f}')
out.close()
