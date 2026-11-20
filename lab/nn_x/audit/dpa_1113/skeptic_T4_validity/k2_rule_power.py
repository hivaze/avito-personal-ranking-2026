"""Skeptic check of T4: (a) PSI of the as-of holdout luo0 vs public; (b) holdout paired SE of the 6 H-contrasts on the full 30K
vs the 10,925 feeds that have any as-of history, vs S (stale_both) SEs from the proposer's s5 log; (c) does the CURRENT
(fresh) holdout already meet the Kendall clause vs stale-val; exact null P(tau >= 0.5) at n = 6. CPU, read-only."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import itertools
import numpy as np
import polars as pl
from scipy.stats import kendalltau
V = "/tmp/scratchpad/dpa/validation/out"
HERE = "/tmp/scratchpad/dpa/skeptic_T4_validity"
hd, fH, fC, curz = np.load(f"{HERE}/asof_feed.npy")
pub = pl.read_parquet(f"{V}/desc_public.parquet")["luo_zero"].to_numpy()
dh = pl.read_parquet(f"{V}/desc_holdout.parquet")
assert np.allclose(dh["luo_zero"].to_numpy(), curz, atol=2e-3), np.abs(dh["luo_zero"].to_numpy() - curz).max()
def psi(a, b, nb=10, eps=1e-4):
    e = np.unique(np.quantile(b, np.linspace(0, 1, nb + 1)[1:-1]))
    K = len(e) + 1
    pa = np.bincount(np.searchsorted(e, a, side="right"), minlength=K) / len(a)
    pb = np.bincount(np.searchsorted(e, b, side="right"), minlength=K) / len(b)
    pa = np.maximum(pa, eps); pb = np.maximum(pb, eps)
    return float(((pa - pb) * np.log(pa / pb)).sum())
late = hd >= 20710
print(f"(a) luo0 PSI vs public: current holdout {psi(curz, pub):.3f} | as-of holdout all 30K {psi(fH, pub):.3f} | "
      f"as-of days>=20710 ({late.sum()}) {psi(fH[late], pub):.3f}; feed-mean luo0 public {pub.mean():.4f}")
Ch = ["Cd", "Pall", "PF7x", "M2", "Q1a", "Q2a"]
ph = {p: np.load(f"{V}/pfh_{p}.npy") for p in ["G3"] + Ch}
S_se = {"Cd": 0.00052, "Pall": 0.00047, "PF7x": 0.00043, "M2": 0.00034, "Q1a": 0.00034, "Q2a": 0.00036}  # s5 log stale_both
S_d = {"Cd": -0.00195, "Pall": -0.00158, "PF7x": -0.00087, "M2": -0.00042, "Q1a": 0.00009, "Q2a": 0.00003}
print("\n(b) contrast | holdout delta full (SE) | ratio SE/S_SE | days>=20710 delta (SE) | ratio | days<=20709 delta")
hfull, hlate = [], []
for p in Ch:
    d = ph[p] - ph["G3"]
    se_f = d.std(ddof=1) / np.sqrt(len(d)); se_l = d[late].std(ddof=1) / np.sqrt(late.sum())
    hfull.append(d.mean()); hlate.append(d[late].mean())
    print(f"  {p:5s} {d.mean():+.5f} ({se_f:.5f}) {se_f / S_se[p]:.2f} | {d[late].mean():+.5f} ({se_l:.5f}) {se_l / S_se[p]:.2f} | {d[~late].mean():+.5f}")
sv = [S_d[p] for p in Ch]
t1, _ = kendalltau(hfull, sv); t2, _ = kendalltau(hlate, sv)
print(f"\n(c) Kendall(CURRENT fresh holdout, stale-val) on the 6 contrasts: full {t1:+.3f} | days>=20710 subset {t2:+.3f}  (rule threshold 0.5)")
# exact null distribution of tau at n=6 (no ties)
taus = []
for perm in itertools.permutations(range(6)):
    c = sum(1 if (perm[i] < perm[j]) else -1 for i in range(6) for j in range(i + 1, 6))
    taus.append(c / 15)
taus = np.array(taus)
print(f"   exact null P(tau >= 0.5 | n=6, independent rankings) = {np.mean(taus >= 0.5 - 1e-9):.4f}; P(tau >= 0.467) = {np.mean(taus >= 0.4666):.4f}")
# common-main-effect: Kendall among the existing proxies (from the s5 log rows)
prox = {"val": [-131, -136, -58, 62, 36, 29], "val_TO": [-158, -118, -51, 50, 43, 37], "holdout": [-60, -45, -20, -18, -52, -2],
        "public": [-74, -38, -10, -39, 7, 39], "stale_both": [-195, -158, -87, -42, 9, 3]}
print("   pairwise Kendall among proxies (s5 log deltas x1e5):")
ks = list(prox)
for a, b in itertools.combinations(ks, 2):
    print(f"     {a:10s} {b:10s} {kendalltau(prox[a], prox[b])[0]:+.3f}")
