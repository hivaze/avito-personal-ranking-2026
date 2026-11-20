"""(E) what perfect training labels could add: a fit to the recorded teacher-weight sweep, and scale references.

Recorded member differences against click-label members (experiments/distillation/README.md, validation, width 128,
4 seeds): teacher weight w = 0.3 / 0.5 / 0.8 / 1.0 -> +0.00152 / +0.00257 (± 0.00047) / +0.00273 / +0.00228.
Model of the gain (assumption, stated in the report):  g(w) = a * (1 - (1 - w)^2) - b * w^2
  a = value of removing ALL label-sampling variance from the target with no teacher bias (perfect soft labels),
  (1 - w)^2 = share of the one-hot target's sampling variance left in the target (1 - w) y + w q,
  b * w^2 = cost of the teacher's own error entering the target with weight w.
Also: the within-feed logit sd of the best val score (to put the train position offset of d_position.py on a scale).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403

w = np.array([0.3, 0.5, 0.8, 1.0])
g = np.array([0.00152, 0.00257, 0.00273, 0.00228])
X = np.stack([1 - (1 - w) ** 2, -(w ** 2)], 1)
coef, res, *_ = np.linalg.lstsq(X, g, rcond=None)
a, b = coef
fit = X @ coef
print(f"fit g(w) = a(1-(1-w)^2) - b w^2: a = {a:+.5f}, b = {b:+.5f}; fitted {np.round(fit, 5)} vs recorded {g}; "
      f"rms residual {np.sqrt(((fit - g) ** 2).mean()):.5f}")
# sensitivity: drop one point at a time
for i in range(4):
    m = np.arange(4) != i
    ci, *_ = np.linalg.lstsq(X[m], g[m], rcond=None)
    print(f"  without w={w[i]}: a = {ci[0]:+.5f}, b = {ci[1]:+.5f}")
# jitter by the recorded per-point SE (0.00047 at w = 0.5; assumed for all points)
rng = np.random.default_rng(0)
A = []
for _ in range(5000):
    gg = g + rng.normal(0, 0.00047, 4)
    A.append(np.linalg.lstsq(X, gg, rcond=None)[0][0])
A = np.array(A)
print(f"a under ±0.00047 noise on each point: {A.mean():+.5f} ± {A.std():.5f} (5-95%: {np.percentile(A, 5):+.5f} .. {np.percentile(A, 95):+.5f})")
print(f"perfect labels beyond the adopted w = 0.5 member: a - g(0.5) = {a - g[1]:+.5f}")
ratio_val = 0.00198 / 0.00257       # pool / member distillation gain, validation (README table: 0.39887 -> 0.40085)
ratio_hold = 0.00021 / 0.00200      # holdout (0.32328 -> 0.32349 pool; +0.00200 member)
print(f"pool / member transfer of the distillation gain: val {ratio_val:.2f}, holdout {ratio_hold:.2f} -> "
      f"pool-level value of perfect labels beyond the current file: {ratio_hold * (a - g[1]):+.5f} .. {ratio_val * (a - g[1]):+.5f}")
# data-scaling equivalence (local log-linear slope from 80% -> 100% of users)
for nm, d in [("member val", 0.0028), ("member holdout", 0.0022), ("3-pool val", 0.0017), ("3-pool holdout", 0.0019)]:
    slope = d / np.log(1 / 0.8)
    print(f"  {nm}: slope {slope:.4f} NDCG per e-fold of users; doubling the users ~ +{slope * np.log(2):.4f}; "
          f"perfect labels (a) ~ x{np.exp(a / slope):.2f} users; distillation w=0.5 ~ x{np.exp(g[1] / slope):.2f}")

# logit scale of the val score inside feeds
F, M, off = layout()
y = labels()
fid = np.repeat(np.arange(len(M)), M)
s = pool_score(BEST) / 1.0036
mu = np.bincount(fid, s, len(M)) / M
sd = np.sqrt(np.bincount(fid, (s - mu[fid]) ** 2, len(M)) / M)
print(f"\nbest val score / T: within-feed sd mean {sd.mean():.3f}, row-weighted rms {np.sqrt((sd ** 2 * M).sum() / M.sum()):.3f} "
      f"(M >= 50 feeds: {np.sqrt((sd[M >= 50] ** 2 * M[M >= 50]).sum() / M[M >= 50].sum()):.3f})")
