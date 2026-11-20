"""A4: does a SHARED cooldown data order make members alike? (CPU, existing preds only, no model code)

All F7x members (r6_cd_m02_r101-107) were re-cooled with the same data seed 811, all F7x70 members with 812, and the
CDSEED test produced r10x_cs813 twins. The eval-curve residuals of the re-cools are step-locked across roots
(a2/a3), i.e. the end-of-cooldown iterate noise is driven by the batch sequence. Test on predictions:
  pairwise within-feed-z correlation for (same data seed, different trunk) vs (different data seed, different trunk)
  vs (same trunk, different data seed); and the 4-member pool NDCG for single-seed pools vs seed-mixed pools of the
  same roots (paired per feed).
Writes a4_dataorder.txt via stdout."""
import os
import sys

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "12")
sys.path.insert(0, "<repo>/nn_x/pubdiag/overfit")
import itertools

import numpy as np

from lib import H, ROOT, Split, se  # noqa: E402

RP = ROOT / "nn_x/r6/preds"
PP = ROOT / "preds"


def path(m, split):
    if split == "holdout":
        for d in (PP / "holdout", RP / "holdout"):
            f = d / f"nn_{m}_holdout.npy"
            if f.exists():
                return f
        return None
    for d in (PP, RP):
        f = d / f"nn_{m}_trainonly_{split}.npy"
        if f.exists():
            return f
    return None


SPL = {s: Split(s) for s in ("val", "holdout", "test")}
cache = {}


def Z(m, s):
    if (m, s) not in cache:
        p = path(m, s)
        cache[(m, s)] = None if p is None else SPL[s].z(np.load(p))
    return cache[(m, s)]


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


R4 = [101, 102, 103, 104]
m811 = {r: f"r6_cd_m02_r{r}" for r in R4}
m813 = {r: f"r6_cd_m02_r{r}_cs813" for r in R4}
c811 = {r: f"r6_cd_c0_r{r}" for r in R4}
c813 = {r: f"r6_cd_c0_r{r}_cs813" for r in R4}
p812 = {r: f"r6_cd_m02_p70k_r{r}" for r in R4}
f7 = {r: f"v17_pre2s_s{r}" for r in R4}
v28 = [f"v28_lr1_m02_s{s}" for s in (411, 412, 413, 414)]

print("== A4.1 Pairwise within-feed-z correlation of train-only predictions (mean over pairs; n pairs)")
groups = {
    "same data seed, different trunk  (m02 811 pairs)": [(m811[a], m811[b]) for a, b in itertools.combinations(R4, 2)],
    "same data seed, different trunk  (m02 813 pairs)": [(m813[a], m813[b]) for a, b in itertools.combinations(R4, 2)],
    "same data seed, different trunk  (c0 811 pairs)": [(c811[a], c811[b]) for a, b in itertools.combinations(R4, 2)],
    "same data seed, different trunk  (c0 813 pairs)": [(c813[a], c813[b]) for a, b in itertools.combinations(R4, 2)],
    "same data seed, different trunk  (p70k 812 pairs)": [(p812[a], p812[b]) for a, b in itertools.combinations(R4, 2)],
    "diff data seed, different trunk  (m02 811 x 813)": [(m811[a], m813[b]) for a in R4 for b in R4 if a != b],
    "diff data seed, different trunk  (c0 811 x 813)": [(c811[a], c813[b]) for a in R4 for b in R4 if a != b],
    "diff data seed, same trunk       (m02 811 vs 813)": [(m811[a], m813[a]) for a in R4],
    "diff data seed, same trunk       (c0 811 vs 813)": [(c811[a], c813[a]) for a in R4],
    "diff seed+snapshot, same root    (m02 811 vs p70k 812)": [(m811[a], p812[a]) for a in R4],
    "same data seed, same trunk, arm  (m02 vs c0, 811)": [(m811[a], c811[a]) for a in R4],
    "independent full runs            (F7 originals)": [(f7[a], f7[b]) for a, b in itertools.combinations(R4, 2)],
    "independent full runs            (v28 lr1 m02 scratch)": list(itertools.combinations(v28, 2)),
}
res = {}
for g, pairs in groups.items():
    out = []
    for s in ("val", "test", "holdout"):
        cs = [corr(Z(a, s), Z(b, s)) for a, b in pairs if Z(a, s) is not None and Z(b, s) is not None]
        out.append(f"{s} {np.mean(cs):.4f}+-{np.std(cs, ddof=1) / np.sqrt(len(cs)) if len(cs) > 1 else 0:.4f} (n{len(cs)})"
                   if cs else f"{s} n/a")
        res[(g, s)] = cs
    print(f"{g:56s} " + "  ".join(out))

print("\n== A4.2 4-member pools of the SAME four roots: one data seed vs two data seeds (member-equal per-feed z-mean;"
      " NDCG paired per feed)")


def pool_nd(ms, s):
    zs = [Z(m, s) for m in ms]
    if any(z is None for z in zs):
        return None
    return SPL[s].nd(np.mean(zs, 0))


for name, A, B in [("m02", m811, m813), ("c0", c811, c813)]:
    for s in ("val", "holdout"):
        single = [pool_nd([A[r] for r in R4], s), pool_nd([B[r] for r in R4], s)]
        mixed = [pool_nd([A[101], A[102], B[103], B[104]], s), pool_nd([B[101], B[102], A[103], A[104]], s),
                 pool_nd([A[101], B[102], A[103], B[104]], s), pool_nd([B[101], A[102], B[103], A[104]], s)]
        if any(x is None for x in single + mixed):
            print(f"   {name} {s}: missing preds")
            continue
        d = np.mean(mixed, 0) - np.mean(single, 0)
        mem = np.mean([SPL[s].nd(Z(A[r], s)).mean() for r in R4] + [SPL[s].nd(Z(B[r], s)).mean() for r in R4])
        print(f"   {name:4s} {s:8s} single-seed pools {np.mean([x.mean() for x in single]):.5f} | seed-mixed pools "
              f"{np.mean([x.mean() for x in mixed]):.5f} | mixed - single {d.mean():+.5f} +- {se(d):.5f} "
              f"(z {d.mean() / se(d):+.1f}) | mean member {mem:.5f}")
print("   (both arms use the same 8 runs; only the grouping differs, so the delta isolates the diversity value of a"
      " second data order)")
