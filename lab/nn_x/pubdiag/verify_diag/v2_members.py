"""Member/root-level re-cool checks (F7b-F7, F7x-F7, F7x-F7b on val TO/FT and holdout), m02-part decomposition,
TO/FT halves of pool deltas, Tue/Thu, errors-in-variables transfer factor, and cold/warm splits on val + holdout."""
from v_lib import *
from scipy import stats
import polars as pl

fv, Mv, ov = meta("val"); _, Mh, oh = meta("holdout")
yv = y_of("val"); yh = y_of("holdout")
g = groups_val()
roots = range(101, 108)
nm = {"F7": "v17_pre2s_s{}", "F7b": "r6_cd_c0_r{}", "F7x": "r6_cd_m02_r{}", "F7x70": "r6_cd_m02_p70k_r{}"}

# ---------- member-level per-root deltas
pfm = {}
for fam, pat in nm.items():
    for r in roots:
        m = pat.format(r)
        pfm[(fam, r, "to")] = pf("val", np.load(member_file(m, "val", "to")))
        pfm[(fam, r, "ft")] = pf("val", np.load(member_file(m, "val", "ft")))
        pfm[(fam, r, "ho")] = pf("holdout", np.load(member_file(m, "holdout", "to")))
print("member-level, root-averaged per-feed deltas (mean +- SE over feeds of the root-averaged delta)")
for a, b in [("F7b", "F7"), ("F7x", "F7"), ("F7x", "F7b"), ("F7x70", "F7"), ("F7x70", "F7x")]:
    res = {}
    for h in ["to", "ft", "ho"]:
        d = np.mean([pfm[(a, r, h)] - pfm[(b, r, h)] for r in roots], 0)
        res[h] = (d.mean(), se_iid(d) if h == "ho" else se_clu(d, g))
    per_root = [(pfm[(a, r, "to")] - pfm[(b, r, "to")]).mean() - (pfm[(a, r, "ho")] - pfm[(b, r, "ho")]).mean() for r in roots]
    gap = res["to"][0] - res["ho"][0]; gse = np.hypot(res["to"][1], res["ho"][1])
    print(f"{a}-{b}: valTO {res['to'][0]:+.5f}+-{res['to'][1]:.5f} valFT {res['ft'][0]:+.5f}+-{res['ft'][1]:.5f} "
          f"ho {res['ho'][0]:+.5f}+-{res['ho'][1]:.5f} | gap(valTO-ho) {gap:+.5f} z {gap/gse:+.2f} | per-root gaps {np.round(per_root,5).tolist()} pos {sum(x>0 for x in per_root)}/7")

# ---------- 7-member TO mixes F7b vs F7 (val TO, holdout)
for split in ["val", "holdout"]:
    a = pf(split, pool_half(["F7b"], split, "to")); b = pf(split, pool_half(["F7"], split, "to"))
    c = pf(split, pool_half(["F7x"], split, "to"))
    d = a - b; d2 = c - a; d3 = c - b
    s = (lambda x: se_clu(x, g)) if split == "val" else se_iid
    print(f"7-mix TO {split}: F7b-F7 {d.mean():+.5f}+-{s(d):.5f}  F7x-F7b {d2.mean():+.5f}+-{s(d2):.5f}  F7x-F7 {d3.mean():+.5f}+-{s(d3):.5f}")

# ---------- m02 part decomposition (counterfactual pools with F7 in place of F7x / F7x70)
CF = {"Pall": ["F5cd", "F6", "F7", "F9", "F10", "F12"], "G3c": ["F10", "F7", "F7"], "M2c": ["F7", "F7"]}
def pfin(fams, split):
    return pf(split, pool_final(fams, split))
POOLS2 = dict(POOLS); POOLS2.update(CF)
V = {k: pfin(f, "val") for k, f in POOLS2.items()}
Hh = {k: pfin(f, "holdout") for k, f in POOLS2.items()}
for a, b in [("PF7x", "Pall"), ("G3", "G3c"), ("M2", "M2c")]:
    d = V[a] - V[b]; dh = Hh[a] - Hh[b]
    nd = V[b] - V["Cd"]
    print(f"m02 part {a}-{b}: val {d.mean():+.5f}+-{se_clu(d,g):.5f}  holdout {dh.mean():+.5f}+-{se_iid(dh):.5f}  | non-m02 ({b}-Cd) val {nd.mean():+.5f}")

# ---------- TO-only / FT-only halves vs Cd on val
for half in ["to", "ft"]:
    s = {k: pf("val", pool_half(POOLS[k], "val", half)) for k in ["Cd", "PF7x", "G3", "M2"]}
    print(half, "half only:", "  ".join(f"{k}-Cd {(s[k]-s['Cd']).mean():+.5f}+-{se_clu(s[k]-s['Cd'],g):.5f}" for k in ["PF7x", "G3", "M2"]))

# ---------- Tue/Thu
day = (fv["timestamp"].to_numpy() + 10800) // 86400
tue = day < 20720
for a, b in [("M2", "Cd"), ("M2", "M2c"), ("G3", "Cd"), ("PF7x", "Cd")]:
    d = V[a] - V[b]
    print(f"{a}-{b}: Tue {d[tue].mean():+.5f}+-{se_iid(d[tue]):.5f}  Thu {d[~tue].mean():+.5f}+-{se_iid(d[~tue]):.5f}")

# ---------- errors-in-variables transfer factor, profile LR
nP = 8562
pubLB = {"Cd": 0.39208, "PF7x": 0.392723, "G3": 0.392819, "M2": 0.392434}
A = ["PF7x", "G3", "M2"]
Dm = np.stack([V[a] - V["Cd"] for a in A], 1)
_, inv = np.unique(g, return_inverse=True)
Sc = np.stack([np.bincount(inv, weights=Dm[:, j] - Dm[:, j].mean()) for j in range(3)], 1)
Vv = Sc.T @ Sc / len(Mv) ** 2; Vp = np.cov(Dm.T) / nP
v = Dm.mean(0); p = np.array([pubLB[a] - pubLB["Cd"] for a in A])
Vvi, Vpi = np.linalg.inv(Vv), np.linalg.inv(Vp)
def prof(b):
    Hm = Vvi + b * b * Vpi; mu = np.linalg.solve(Hm, Vvi @ v + b * Vpi @ p)
    return (v - mu) @ Vvi @ (v - mu) + (p - b * mu) @ Vpi @ (p - b * mu)
bs = np.linspace(-3, 4, 7001); L = np.array([prof(b) for b in bs]); bh = bs[L.argmin()]
ci = bs[L - L.min() < 3.84]
print(f"EIV transfer b_hat {bh:.2f}, 95% CI [{ci.min():.2f},{ci.max():.2f}], LR p(b=1) {1-stats.chi2.cdf(prof(1)-L.min(),1):.3f}, p(b=0) {1-stats.chi2.cdf(prof(0)-L.min(),1):.3f}")
