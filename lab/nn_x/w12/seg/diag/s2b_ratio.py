"""A2' (post hoc rule, DECLARATION block 1): ratio of skill H=256 / NEAR per ranker with a cluster bootstrap by user."""
from lib import *
ND = dict(np.load(OUT / "rk_ndcg.npz")); AU = dict(np.load(OUT / "rk_auc.npz")); E0 = np.load(OUT / "rand_ndcg.npy")
SK = {k: (v - E0) / (1 - E0) for k, v in ND.items()}; AX = {k: v - 0.5 for k, v in AU.items()}; NX = {k: v - E0 for k, v in ND.items()}
rng = np.random.default_rng(7); B = 2000
keys = [k for k in ND if not k.startswith(("f3", "co-vis"))]
def stat(D, ref):
    # per-cluster sums for CAP and ref
    out = {}
    for k in keys:
        x = np.nan_to_num(D[k]); ok = ~np.isnan(D[k])
        out[k] = (np.bincount(clu, weights=x * (CAP & ok), minlength=NCL), np.bincount(clu, weights=(CAP & ok).astype(float), minlength=NCL),
                  np.bincount(clu, weights=x * (ref & ok), minlength=NCL), np.bincount(clu, weights=(ref & ok).astype(float), minlength=NCL))
    return out
for refname, ref in (("NEAR", NEAR), ("REST", REST)):
    for title, D in (("skill", SK), ("NDCG - E_random", NX), ("AUC - 0.5", AX)):
        st = stat(D, ref); W = rng.multinomial(NCL, np.ones(NCL) / NCL, size=B).astype(np.float64)
        R = {k: (W @ a) / (W @ b) / ((W @ c) / (W @ d)) for k, (a, b, c, d) in st.items()}
        pt = {k: a.sum() / b.sum() / (c.sum() / d.sum()) for k, (a, b, c, d) in st.items()}
        rows = []
        for k in keys:
            dd = R["FINAL"] - R[k]
            rows.append([k, f"{pt[k]:.3f}", f"{R[k].std():.3f}", f"[{np.quantile(R[k], 0.025):.3f}, {np.quantile(R[k], 0.975):.3f}]", "" if k == "FINAL" else f"{pt['FINAL'] - pt[k]:+.3f}+-{dd.std():.3f} (z {(pt['FINAL'] - pt[k]) / dd.std():+.1f})"])
        print(f"\n### R = {title}(H=256) / {title}({refname}); cluster bootstrap {B} draws\n" + tab(rows, ["ranker", "R", "boot SE", "95% CI", "R_FINAL - R_ranker"]))
