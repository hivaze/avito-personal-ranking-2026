"""Does the proposed T4 adoption gate (Kendall(stale-holdout, stale-val) >= 0.5 on the 6 H-defined contrasts) discriminate?
Apply it to lenses that already exist (numbers copied from dpa/validation/out/s5_stale_transfer.log, per-contrast table)."""
import itertools
import numpy as np
from scipy.stats import kendalltau
C = ["Cd", "Pall", "PF7x", "M2", "Q1a", "Q2a"]
L = {
 "public":      [-0.00074, -0.00038, -0.00010, -0.00039, +0.00007, +0.00039],
 "val":         [-0.00131, -0.00136, -0.00058, +0.00062, +0.00036, +0.00029],
 "val_TO":      [-0.00158, -0.00118, -0.00051, +0.00050, +0.00043, +0.00037],
 "holdout":     [-0.00060, -0.00045, -0.00020, -0.00018, -0.00052, -0.00002],
 "hold_weekend":[-0.00078, -0.00097, -0.00037, +0.00035, +0.00010, +0.00043],
 "hold_late3d": [-0.00044, -0.00045, -0.00040, -0.00043, -0.00045, -0.00000],
 "hold_luo0>=.3":[-0.00088, -0.00018, +0.00009, -0.00121, -0.00161, -0.00082],
 "stale_TO(S)": [-0.00195, -0.00158, -0.00087, -0.00042, +0.00009, +0.00003],
}
S = L["stale_TO(S)"]
print("Kendall tau vs S (stale-val, TO half) on the 6 H-defined contrasts; gate = tau >= 0.5")
for k, v in L.items():
    t, p = kendalltau(v, S)
    print(f"  {k:14s} tau {t:+.3f} (p {p:.3f})  gate {'PASS' if t >= 0.5 else 'fail'}")
# null distribution of tau for n = 6 (all permutations)
taus = np.array([kendalltau(np.arange(6), q)[0] for q in itertools.permutations(range(6))])
print(f"null P(tau >= 0.5 | n 6, independent orderings) = {(taus >= 0.5 - 1e-9).mean():.3f}; attainable taus >= 0.5: {sorted(set(np.round(taus[taus >= 0.5], 3)))}")
# Monte Carlo: a lens that measures the SAME true contrasts as S with SE 0.00025 (holdout-like) -> P(gate pass)
rng = np.random.default_rng(0)
seS = np.array([0.00052, 0.00047, 0.00043, 0.00034, 0.00034, 0.00036])
true = np.array(S)  # treat S as truth (optimistic)
pas = []
for _ in range(20000):
    a = true + rng.normal(0, 0.00025, 6); b = true + rng.normal(0, seS)
    pas.append(kendalltau(a, b)[0] >= 0.5)
print(f"MC: P(gate pass | stale-holdout measures exactly S's truth, SE 0.00025; S re-drawn with its SEs) = {np.mean(pas):.3f}")
# and a lens measuring the LIVE-holdout truth (different quantity) vs S
pas2 = []
for _ in range(20000):
    a = np.array(L["holdout"]) + rng.normal(0, 0.00025, 6) * 0  # holdout as is
    b = true + rng.normal(0, seS)
    pas2.append(kendalltau(a, b)[0] >= 0.5)
print(f"MC: P(gate pass | lens = live holdout values, S re-drawn with its SEs) = {np.mean(pas2):.3f}")
