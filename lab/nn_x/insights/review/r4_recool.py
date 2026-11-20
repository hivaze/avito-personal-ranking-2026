"""Review R4: independent re-parse of the train logs for the re-cool val excess D.
(a) unpaired: F7x (m02 re-cool of F7 trunks, CDSEED 811 and 813) vs v30 (F7 recipe + m02 from scratch, 4 seeds):
    late-cooldown (k in {16320, 18820, 19670}) val-minus-holdout gain over each run's own trunk plateau (67.5k-77.5k).
(b) paired same-trunk: c0 re-cool (mix .1) vs the F7 original's continuous mix-.1 cooldown, same k grid.
(c) paired same-trunk: m02 re-cool vs c0 re-cool (both re-cools) as a null check of the metric (~0 expected for the gap).
Also: leave-one-seed-out spread for (a) and a seed-cluster-robust SE (F7x runs that share a CDSEED are one cluster)."""
import re, numpy as np
from pathlib import Path
R = Path("<repo>")
pat = re.compile(r"step (\d+)/\d+ .*?val_ndcg ([\d.]+) holdout_ndcg ([\d.]+)")
def curve(p):
    a = np.array([(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in map(pat.search, open(p)) if m])
    return a[:, 0], a[:, 1], a[:, 2]
def plateau(p):
    s, v, h = curve(p); w = (s >= 67500) & (s <= 77500); return v[w].mean(), h[w].mean()
KO = np.array([16320, 18820, 19670])
def late(s, v, h, t0):
    k = s - t0; mk = k > 0
    return np.interp(KO, k[mk], v[mk]).mean(), np.interp(KO, k[mk], h[mk]).mean()
def gap_scratch(run):
    s, v, h = curve(R / f"nn/logs/{run}.log")
    if s.max() < 98350: return None
    Pv, Ph = plateau(R / f"nn/logs/{run}.log"); lv, lh = late(s, v, h, 78680)
    return lv - Pv, lh - Ph
def gap_recool(run, root):
    s, v, h = curve(R / f"nn_x/r6/logs/{run}.log")
    Pv, Ph = plateau(R / f"nn/logs/v17_pre2s_s{root}.log"); lv, lh = late(s, v, h, 0)
    return lv - Pv, lh - Ph
def orig_cool(root):  # F7 original continuous mix-.1 cooldown
    p = R / f"nn/logs/v17_pre2s_s{root}.log"; s, v, h = curve(p); Pv, Ph = plateau(p); lv, lh = late(s, v, h, 78680)
    return lv - Pv, lh - Ph
v30 = {r: gap_scratch(f"v30_f7_m02_s{r}") for r in range(901, 907)}
v30 = {r: g for r, g in v30.items() if g}
print("v30 complete:", sorted(v30), " per-run (dval, dhold, gap):", {r: tuple(round(x, 5) for x in (g[0], g[1], g[0]-g[1])) for r, g in v30.items()})
roots = range(101, 108)
f7x = {r: gap_recool(f"r6_cd_m02_r{r}", r) for r in roots}
f7x13 = {r: gap_recool(f"r6_cd_m02_r{r}_cs813", r) for r in range(101, 105)}
c0 = {r: gap_recool(f"r6_cd_c0_r{r}", r) for r in roots}
c013 = {r: gap_recool(f"r6_cd_c0_r{r}_cs813", r) for r in range(101, 105)}
orig = {r: orig_cool(r) for r in roots}
G = lambda d: np.array([g[0] - g[1] for g in d.values()])
Vv = lambda d: np.array([g[0] for g in d.values()]); Hh = lambda d: np.array([g[1] for g in d.values()])
a, b = G(v30), G(f7x)
d = b.mean() - a.mean(); se = np.sqrt(b.var(ddof=1) / len(b) + a.var(ddof=1) / len(a))
print(f"\n(a) late gap (val-hold gain over own plateau): F7x(cs811) {b.mean():+.5f} (sd {b.std(ddof=1):.5f}, n {len(b)}) | v30 {a.mean():+.5f} (sd {a.std(ddof=1):.5f}, n {len(a)})")
print(f"    F7x - v30 = {d:+.5f} +- {se:.5f} (naive unpaired)")
# cluster-robust: F7x cs811 runs share data order -> treat the F7x mean as having at least the v30 between-seed variance / 1
se_c = np.sqrt(a.var(ddof=1) * (1 / len(a) + 1.0))
print(f"    if the 7 cs811 re-cools count as ONE data-order draw (worst case): SE ~ {se_c:.5f} -> z {d/se_c:.2f}")
b13 = G(f7x13); print(f"    F7x cs813 (4 roots) gap {b13.mean():+.5f} -> minus v30 {b13.mean()-a.mean():+.5f}")
print(f"    components: val gain F7x {Vv(f7x).mean():+.5f} v30 {Vv(v30).mean():+.5f} | hold gain F7x {Hh(f7x).mean():+.5f} v30 {Hh(v30).mean():+.5f}")
for drop in v30:
    a2 = np.array([g[0] - g[1] for r, g in v30.items() if r != drop]); print(f"    leave out v30 s{drop}: F7x - v30 {b.mean()-a2.mean():+.5f}")
# (b) paired same-trunk: c0 re-cool vs F7 original continuous mix .1
pc = np.array([(c0[r][0] - c0[r][1]) - (orig[r][0] - orig[r][1]) for r in roots])
print(f"\n(b) PAIRED same trunk, same recipe (mix .1): c0 re-cool minus F7 original continuous cooldown, late gap: {pc.mean():+.5f} +- {pc.std(ddof=1)/np.sqrt(len(pc)):.5f} (n 7, cs811)"
      f"; val {np.mean([c0[r][0]-orig[r][0] for r in roots]):+.5f} hold {np.mean([c0[r][1]-orig[r][1] for r in roots]):+.5f}")
pc13 = np.array([(c013[r][0] - c013[r][1]) - (orig[r][0] - orig[r][1]) for r in range(101, 105)])
print(f"    cs813: {pc13.mean():+.5f} +- {pc13.std(ddof=1)/2:.5f} (n 4)")
# (c) m02 - c0 re-cool (same trunk, same harness, same seed): gap difference
pm = np.array([(f7x[r][0] - f7x[r][1]) - (c0[r][0] - c0[r][1]) for r in roots])
print(f"(c) PAIRED m02 re-cool minus c0 re-cool, late gap: {pm.mean():+.5f} +- {pm.std(ddof=1)/np.sqrt(7):.5f}")
# implied: if the re-cool excess were mix-independent, (a) ~= (b) + [m02-vs-mix.1 gap effect in scratch runs]
print("\nImplication: same-trunk paired estimate (b) vs unpaired v30 estimate (a) of the same quantity.")
