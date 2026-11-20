"""Core recomputation: pool val/holdout NDCG, paired deltas, public/private SE, chi2 1:1 transfer test,
8-point calibration, and submitted-file integrity."""
from v_lib import *
import polars as pl
from scipy import stats

out = {}
fv, Mv, ov = meta("val"); ft_, Mt, ot = meta("test")
ts = ft_["timestamp"].to_numpy(); pub = ts < 1790497920; npub, npriv = pub.sum(), (~pub).sum()
print("n_val", len(Mv), "n_pub", npub, "n_priv", npriv)

S = {}; PFv = {}; PFh = {}
for k, fams in POOLS.items():
    S[k] = pool_final(fams, "val")
    PFv[k] = pf("val", S[k])
    PFh[k] = pf("holdout", pool_final(fams, "holdout"))
    print(f"{k:5s} val {PFv[k].mean():.5f}  holdout {PFh[k].mean():.5f}")
# cross-check against saved ens arrays
for k in ["Cd", "PF7x", "G3", "M2", "Pall"]:
    sv = 0.5 * z("val", np.load(P / f"ens_{k}_ft_val.npy")) + 0.5 * z("val", np.load(P / f"ens_{k}_to_val.npy"))
    print(k, "saved-ens val NDCG", round(pf("val", sv).mean(), 5), "maxabs diff vs mine", float(np.abs(sv - S[k]).max()))

g = groups_val()
pairs = [("PF7x", "Cd"), ("G3", "Cd"), ("M2", "Cd"), ("M2", "PF7x"), ("G3", "PF7x"), ("M2", "G3"), ("PF7x", "Pall")]
pubLB = {"Cd": 0.39208, "PF7x": 0.392723, "G3": 0.392819, "M2": 0.392434}
print("\npair: val delta, se_iid, se_clu | pubSE(iid scaled) privSE | holdout delta se | pub delta resid z")
for a, b in pairs:
    d = PFv[a] - PFv[b]; dh = PFh[a] - PFh[b]
    s_i, s_c = se_iid(d), se_clu(d, g)
    sd = d.std(ddof=1)
    pse, prse = sd / np.sqrt(npub), sd / np.sqrt(npriv)
    line = f"{a}-{b}: {d.mean():+.5f} {s_i:.5f} {s_c:.5f} | {pse:.5f} {prse:.5f} | ho {dh.mean():+.5f} {se_iid(dh):.5f}"
    if a in pubLB and b in pubLB:
        r = (pubLB[a] - pubLB[b]) - d.mean(); tse = np.sqrt(s_c ** 2 + pse ** 2)
        line += f" | pub {pubLB[a]-pubLB[b]:+.5f} resid {r:+.5f} +- {tse:.5f} z {r/tse:+.2f}"
    print(line)

# M-reweighted public SE: weight val feeds by public/val M-bin density ratio
bins = np.array([0, 10, 20, 30, 50, 75, 100, 150, 200, 250, 1000])
bv = np.digitize(Mv, bins); bt = np.digitize(Mt[pub], bins)
w = np.zeros(len(Mv))
for b_ in np.unique(bv):
    w[bv == b_] = (bt == b_).mean() / (bv == b_).mean()
print("\nM-reweighted public SE:")
for a, b in pairs[:6]:
    d = PFv[a] - PFv[b]; mu = np.average(d, weights=w); var = np.average((d - mu) ** 2, weights=w)
    print(f"  {a}-{b}: wmean {mu:+.5f} pubSE {np.sqrt(var/npub):.5f}")

# joint chi2 for 3 contrasts vs Cd
A = ["PF7x", "G3", "M2"]
Dm = np.stack([PFv[a] - PFv["Cd"] for a in A], 1)
_, inv = np.unique(g, return_inverse=True)
Sc = np.stack([np.bincount(inv, weights=Dm[:, j] - Dm[:, j].mean()) for j in range(3)], 1)
Vval = Sc.T @ Sc / len(Mv) ** 2
Vpub = np.cov(Dm.T) / npub
r = np.array([pubLB[a] - pubLB["Cd"] - Dm[:, j].mean() for j, a in enumerate(A)])
chi2 = float(r @ np.linalg.solve(Vval + Vpub, r))
print(f"\njoint chi2(3) = {chi2:.2f}, p = {1-stats.chi2.cdf(chi2,3):.3f}; resid {r}")
# transfer factor b: pub = b*val ; GLS with known cov (ignoring val noise in regressor -> approximate) and LR
v = Dm.mean(0); pubd = np.array([pubLB[a] - pubLB["Cd"] for a in A])
def nll(b):
    Sg = Vpub + b * b * Vval
    rr = pubd - b * v
    return 0.5 * (rr @ np.linalg.solve(Sg, rr) + np.log(np.linalg.det(Sg)))
bs = np.linspace(-2, 3, 5001); L = np.array([nll(b) for b in bs]); bh = bs[L.argmin()]
ci = bs[L - L.min() < 1.92]
print(f"transfer factor b_hat {bh:.2f}, CI [{ci.min():.2f}, {ci.max():.2f}], LR p(b=1) {1-stats.chi2.cdf(2*(nll(1)-L.min()),1):.3f}")

# 8-point calibration
pts = {"v3": (0.3816, 0.3739), "v4": (0.3839, 0.3772), "v5": (0.3869, 0.3802), "v10": (0.39399, 0.38786),
       "Cd": (0.39901, 0.39208), "PF7x": (0.39974, 0.392723), "G3": (0.40032, 0.392819), "M2": (0.40094, 0.392434)}
x = np.array([p[0] for p in pts.values()]); yv = np.array([p[1] for p in pts.values()])
X = np.c_[np.ones(8), x]; beta, *_ = np.linalg.lstsq(X, yv, rcond=None); res = yv - X @ beta
s = np.sqrt(res @ res / 6); se_b = s * np.sqrt(np.linalg.inv(X.T @ X)[1, 1])
Hh = X @ np.linalg.inv(X.T @ X) @ X.T
print(f"\ncalib: pub = {beta[0]:.4f} + {beta[1]:.3f}(+-{se_b:.3f}) val, resid sd {s:.5f}")
for i, k in enumerate(pts):
    s_i = np.sqrt((res @ res - res[i] ** 2 / (1 - Hh[i, i])) / 5)
    t = res[i] / (s_i * np.sqrt(1 - Hh[i, i]))
    print(f"  {k:5s} resid {res[i]:+.5f} t_ext {t:+.2f} p {2*(1-stats.t.cdf(abs(t),5)):.3f}")
# note: deg of freedom for t_ext with n=8, p=2 is n-p-1 = 5

# submitted parquet integrity: within-feed rank of sub vs my pool final on test
f_, M_, o_ = meta("test")
fid = f_["feed_id"].to_numpy()[np.load(FLAT / "test/cand_fidx.npy")]
item = np.load(FLAT / "test/cand_item_id.npy")
key = pl.DataFrame({"feed_id": fid.astype(np.int32), "item_id": item.astype(np.int32), "r": np.arange(len(fid))})
for k, fn in [("Cd", "cand_Cd_memeq_F5cdF6F7.parquet"), ("PF7x", "cand_PF7x_memeq_6fam.parquet"),
              ("G3", "cand_G_F10F7xF7x70_memeq.parquet"), ("M2", "cand_M_F7xF7x70_memeq.parquet")]:
    sub = pl.read_parquet(f"<repo>/subs/{fn}")
    j = key.join(sub, on=["feed_id", "item_id"], how="left").sort("r")
    sc = j["score"].to_numpy().astype(np.float64)
    mine = pool_final(POOLS[k], "test")
    zr_sub = z("test", sc); zr_mine = z("test", feed_rank_z(o_, M_, mine.astype(np.float32))[:, 0])
    # within-feed rank agreement
    rk_sub = feed_rank_z(o_, M_, sc.astype(np.float32))[:, 0]; rk_mine = feed_rank_z(o_, M_, mine.astype(np.float32))[:, 0]
    print(f"sub {k}: null {np.isnan(sc).sum()}, max |rank diff| {np.abs(rk_sub-rk_mine).max():.4g}, frac rows differing {(np.abs(rk_sub-rk_mine)>1e-6).mean():.2e}")
