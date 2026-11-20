"""Part 6: private-delta estimates vs PF7x.
 - reweighted val (final pool) under each label-free private weighting (p2b weights; 1-D post-strat in p3)
 - val train-only half vs holdout (train-only) discrepancy = the part of the val gain the in-period holdout does not see
 - lambda-discounted estimates: est = w-val delta - sel_bias - lambda * (val_to - holdout)
Public LB numbers are printed as context only and are not used in any estimate."""
from shiftlib import *  # noqa: F401,F403

R = dict(np.load(OUT / "p2_val_perfeed.npz")); H = dict(np.load(OUT / "p1_holdout_perfeed.npz"))
W = dict(np.load(OUT / "p2_adv_weights.npz"))
PUBLIC = {"PF7x": 0.392723, "G3": 0.392819, "M2": 0.392434, "Cd": 0.39208}
SEL = {"M2": 0.00018, "G3": 0.0, "Cd": 0.0}  # winner's-curse estimate for the val-best pick (p4: truth = 0.5 x observed)
schemes = ["private_intrinsic", "private_intrinsic+hour", "private_M+cold", "private_overlap"]
rows = []
print(f"{'pool-PF7x':10s} {'val':>17s} " + " ".join(f"{s.replace('private_', 'pw:'):>22s}" for s in schemes) +
      f" {'val to-half':>17s} {'holdout':>17s} {'to-hold disc':>17s}")
for a in ["M2", "G3", "Cd"]:
    d = R[a] - R["PF7x"]
    v = mse(d); ws = [mse(d, W[s]) for s in schemes]
    vt = mse(R[a + "_to"] - R["PF7x_to"]); ho = mse(H[a] - H["PF7x"])
    disc = (vt[0] - ho[0], np.hypot(vt[1], ho[1]))
    print(f"{a + '-PF7x':10s} {v[0]:+.5f}±{v[1]:.5f} " + " ".join(f"{w[0]:+.5f}±{w[1]:.5f}      " for w in ws) +
          f" {vt[0]:+.5f}±{vt[1]:.5f} {ho[0]:+.5f}±{ho[1]:.5f} {disc[0]:+.5f}±{disc[1]:.5f}")
    base = ws[1]  # private intrinsic+hour weights (the label-free weighting with the smallest weight spread among
    #               those that use hour; chosen a priori because the overlap/full weightings put val's cold-heavy
    #               feeds in place of test's, which pmix showed is level-miscalibrated)
    for lam in (0.0, 0.5, 1.0):
        est = base[0] - SEL[a] - lam * disc[0]
        se = np.hypot(base[1], lam * disc[1])
        rows.append(dict(pool=a, ref="PF7x", lam=lam, est=est, se=se, wval=base[0], sel=SEL[a], disc=disc[0],
                         val=v[0], holdout=ho[0], val_to=vt[0]))
print("\n== private-delta estimates vs PF7x: est = private-weighted val (intrinsic+hour) - selection bias - lambda x (val_to - holdout) ==")
for r in rows:
    print(f"{r['pool']:3s}-PF7x lambda {r['lam']:.1f}: {r['est']:+.5f} ± {r['se']:.5f}")
pl.DataFrame(rows).write_csv(OUT / "p6_private_estimate.csv")
print("\ncontext only (not used): public LB deltas vs PF7x: " +
      " | ".join(f"{a} {PUBLIC[a] - PUBLIC['PF7x']:+.5f}" for a in ("M2", "G3", "Cd")))

# share of re-cooled (r6 c0/m02-type) members in each pool and the pool-level val_to - holdout discrepancy vs Cd
print("\n== pool composition vs val_to - holdout discrepancy (vs Cd) ==")
recool = set(FAM["F7x"] + FAM["F7x70"] + FAM["F7b"])
for a in ["PF7x", "G3", "M2", "F7x", "F7b", "F7"]:
    mem = POOLS[a]
    sh = np.mean([m in recool for m in mem])
    vt = mse(R[a + "_to"] - R["Cd_to"]); ho = mse(H[a] - H["Cd"])
    print(f"{a:5s} re-cooled share {sh:.2f} | val_to-Cd {vt[0]:+.5f} | holdout-Cd {ho[0]:+.5f} | disc {vt[0] - ho[0]:+.5f} ± {np.hypot(vt[1], ho[1]):.5f}")
