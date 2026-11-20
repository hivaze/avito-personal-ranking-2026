"""S6: level calibration of val under each weighting vs the public level of the 9 modern scored pools, and the
cross-generation fit public = a + c*val over 4 old + 9 modern files. Writes out/s6_levels.log."""
import sys
from lib import *  # noqa
sys.path.insert(0, "<repo>/eda/errors/segments")
import pmix
LOG = open(OUT / "s6_levels.log", "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n")
W = pmix.weights(); wz = np.load(OUT / "w_val_pubmatch.npz")
EDGES = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250])
fv, Mv, _ = meta("val"); ft, Mt, _ = meta("test"); pub = public_mask()
bv = np.searchsorted(EDGES, Mv, side="right"); bt = np.searchsorted(EDGES, Mt[pub], side="right")
wM = (np.bincount(bt, minlength=10) / len(bt) / (np.bincount(bv, minlength=10) / len(bv)))[bv]; wM /= wM.mean()
ws = {"uniform": np.ones(len(Mv)), "calibM_pub": wM, "pmix_pub": W["public"], "EB_pub": wz["w_pub"]}
pr("level offsets (public - weighted val) per pool:")
pr(f"  {'pool':7s} " + " ".join(f"{k:>11s}" for k in ws))
offs = {k: [] for k in ws}
for p in POOLS:
    pf = np.load(OUT / f"pfv_{p}.npy")
    row = f"  {p:7s} "
    for k, w in ws.items():
        o = PUBLIC[p] - np.average(pf, weights=w); offs[k].append(o); row += f" {o:+11.5f}"
    pr(row)
pr("  mean   " + " ".join(f" {np.mean(v):+11.5f}" for v in offs.values()))
pr("  sd     " + " ".join(f" {np.std(v, ddof=1):11.5f}" for v in offs.values()))
# cross-generation
vals = [np.load(OUT / f"pfv_{p}.npy").mean() for p in POOLS] + [v for v, _ in OLDGEN.values()]
pubs = [PUBLIC[p] for p in POOLS] + [q for _, q in OLDGEN.values()]
vals, pubs = np.array(vals), np.array(pubs)
A = np.c_[np.ones(len(vals)), vals]; coef, res, *_ = np.linalg.lstsq(A, pubs, rcond=None)
r = pubs - A @ coef; s2 = (r ** 2).sum() / (len(r) - 2); cov = s2 * np.linalg.inv(A.T @ A)
pr(f"\ncross-generation (13 files): public = {coef[0]:+.4f} + {coef[1]:.3f} (± {np.sqrt(cov[1, 1]):.3f}) * val; resid sd {np.sqrt(s2):.5f}")
pr("  residuals: " + ", ".join(f"{n} {x:+.5f}" for n, x in zip(POOLS + list(OLDGEN), r)))
