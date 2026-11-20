"""Remaining checks: (a) the 11-point cross-generation calibration and the 'old+Cd fit misses the modern files' statistic with
its prediction SE; (b) Kendall tau of val / holdout orderings with public over the 7 and 5 pools; (c) the holdout's own
winner's curse for G3 (chosen at 13:55 on 11-12 as 'the holdout-best' among the decide4 interim pools); (d) M2 holdout
weekday / weekend split. Uses the k1 per-feed caches. Writes out/k3_misc.log."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "12"
import itertools
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path("<repo>"); HERE = ROOT / "nn_x/audit/valpub/critique"; OUT = HERE / "out"
PUB = {"Cd": 0.39208, "Pall": 0.392436, "PF7x": 0.392723, "G3": 0.392820, "M2": 0.392434, "S2": 0.392473, "TOonly": 0.392346}
OLD = {"v3": (0.3816, 0.3739), "v4": (0.3839, 0.3772), "v5": (0.3869, 0.3802), "v10": (0.39399, 0.38786)}
ORDER = ["Cd", "Pall", "PF7x", "G3", "M2", "S2", "TOonly"]; HC = ["Cd", "Pall", "PF7x", "G3", "M2"]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


pf = {k: np.load(OUT / f"k1_pfval_{k}.npy") for k in ORDER}; hp = {k: np.load(OUT / f"k1_pfhold_{k}.npy") for k in HC}
V = {k: pf[k].mean() for k in ORDER}; H = {k: hp[k].mean() for k in HC}

say("K3 (a). cross-generation calibration public = a + c * val")
xs = np.array([OLD[k][0] for k in OLD] + [V[k] for k in ORDER]); ys = np.array([OLD[k][1] for k in OLD] + [PUB[k] for k in ORDER])
labs = list(OLD) + ORDER
X = np.c_[np.ones(len(xs)), xs]; b, res, *_ = np.linalg.lstsq(X, ys, rcond=None); r = ys - X @ b
s2 = r @ r / (len(xs) - 2); cov = np.linalg.inv(X.T @ X) * s2
say(f"   11 files: a {b[0]:+.4f} c {b[1]:.3f} ± {np.sqrt(cov[1, 1]):.3f}, resid sd {np.sqrt(s2):.5f}")
say("   residuals: " + " ".join(f"{l}:{v:+.5f}" for l, v in zip(labs, r)))
m = np.array([l in ("v3", "v4", "v5", "v10", "Cd") for l in labs])
b5, *_ = np.linalg.lstsq(X[m], ys[m], rcond=None); r5 = ys[m] - X[m] @ b5; s5 = np.sqrt(r5 @ r5 / (m.sum() - 2)); c5 = np.linalg.inv(X[m].T @ X[m]) * s5 ** 2
say(f"   fit on v3/v4/v5/v10/Cd: a {b5[0]:+.4f} c {b5[1]:.3f} ± {np.sqrt(c5[1, 1]):.3f}, resid sd {s5:.5f}; predictions for the 6 later files with prediction SE (resid + parameter):")
for i, l in enumerate(labs):
    if m[i]:
        continue
    pred = X[i] @ b5; se_pred = np.sqrt(s5 ** 2 + X[i] @ c5 @ X[i]); miss = ys[i] - pred
    say(f"     {l:7s} pred {pred:.5f} ± {se_pred:.5f}  actual {ys[i]:.5f}  miss {miss:+.5f}  t {miss / se_pred:+.2f}")
say("   -> the 6 misses are one-sided but individually 0.1-2.6 t; they share one val->public offset (the 6 files are near-copies), so")
say("      they are ~1 independent reading of 'the modern generation sits ~0.001 below the old-era line', not 6.")

say("\nK3 (b). Kendall tau of orderings with public")


def tau(a, b):
    s = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (len(a) * (len(a) - 1) / 2)


p7 = [PUB[k] for k in ORDER]; v7 = [V[k] for k in ORDER]
Hx = dict(H); Hx["TOonly"] = H["G3"]; Hx["S2"] = H["G3"] - 0.00006
say(f"   7 pools: tau(val, public) {tau(v7, p7):+.2f}; tau(holdout incl. TOonly=G3, S2 recorded; public) {tau([Hx[k] for k in ORDER], p7):+.2f}")
p5 = [PUB[k] for k in HC]; v5 = [V[k] for k in HC]; h5 = [H[k] for k in HC]
say(f"   5 pools with a TO form: tau(val, public) {tau(v5, p5):+.2f}; tau(holdout, public) {tau(h5, p5):+.2f}; tau(val, holdout) {tau(v5, h5):+.2f}")
# exact permutation p for n=5
taus = np.array([tau(list(perm), p5) for perm in itertools.permutations(range(5))])
say(f"   P(tau >= 0.6 | random order of 5) = {np.mean(taus >= 0.6 - 1e-9):.3f}; P(tau >= 0.2) = {np.mean(taus >= 0.2 - 1e-9):.3f}")

say("\nK3 (c). the holdout's own winner's curse for G3")
say("   G3 was built at 13:55 on 11-12 as 'the holdout-best' (CURRENT_STATUS 233-235) among the decide4 interim pools (M, G, PF7x, Pall,")
say("   plus 'PF7x+F7x70, root-dedup, holdout-gated' compositions, CURRENT_STATUS 284): K >= 5 pools compared on the holdout.")
for k in ["M2", "PF7x", "Pall", "Cd"]:
    d = hp["G3"] - hp[k]; se = d.std(ddof=1) / np.sqrt(len(d))
    say(f"   G3 - {k:5s} holdout {d.mean():+.5f} ± {se:.5f}  (z {d.mean() / se:+.2f})")
rng = np.random.default_rng(0)
for K in [3, 5, 8]:
    em = rng.standard_normal((100000, K)).max(1).mean()
    say(f"   expected optimism of the holdout-best of K={K} near-copies at SE_h 0.00019-0.00022: {em * 0.00019:+.5f}..{em * 0.00022:+.5f}  (E[max] {em:.2f})")
say("   -> G3's holdout edge over M2 (+0.00018 ± 0.00019) and PF7x (+0.00020 ± 0.00022) is inside the winner's-curse margin of the")
say("      selection that produced it. The report applies the curse to val maxima (M2, S2) but not to the holdout maximum (G3).")

say("\nK3 (d). M2 vs G3 on the holdout by weekday / weekend and public-like / private-like (holdout ts from pubdiag/shift p1)")
z = np.load(ROOT / "nn_x/pubdiag/shift/p1_holdout_perfeed.npz"); ts = z["ts"].astype(np.int64)
wd = ((ts + 10800) // 86400 + 3) % 7; hr = ((ts + 10800) % 86400) // 3600
we = wd >= 5; publike = (wd == 5) | ((wd == 6) & (hr < 12)); privlike = ((wd == 6) & (hr >= 12)) | (wd == 0) | (wd == 1)
d = hp["M2"] - hp["G3"]


def ms(x):
    return f"{x.mean():+.5f} ± {x.std(ddof=1) / np.sqrt(len(x)):.5f} (n {len(x)})"


say(f"   all {ms(d)} | weekday {ms(d[~we])} | weekend {ms(d[we])} | public-like {ms(d[publike])} | private-like {ms(d[privlike])}")
say("   -> weekday vs weekend differ by ~2 SE; the report cites this split in two opposite directions (calib: 'against M2 on private's")
say("      weekdays'; semantics: 're-cooled members gain on holdout weekend feeds yet M2 under-performed on the all-weekend public').")
say("      Both readings are one 2-SE split each; neither is evidence about private.")
(OUT / "k3_misc.log").write_text("\n".join(LOG) + "\n")
print("written", OUT / "k3_misc.log")
