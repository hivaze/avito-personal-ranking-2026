"""critique2 N3: the public-probe programme's numbers, re-derived from the stored per-feed arrays.
 (a) label-free composition of val / public / private (hour, weekend, M, cold share) and the public boundary;
 (b) the pubmix expectation and public paired SE of M2 / TO-only / PF7x / Cd / NEW12 / FTonly / msplit from
     out/cand_*_val.npy of the publicprobe step (same calib.weights), i.e. does the table reproduce;
 (c) the 'dose-response' WLS slope of the realised public residual on re-cool share, AND the competing explanation the
     programme does not test: regression of the residual on the pool's val delta (val-selected pools regress on public),
     fitted on all five scored pools (M2, PF7x, Cd, TO-only, S2) instead of the three the WLS used; predictions for the
     unscored share-0 probes under each model; collinearity of share and d_val among the scored pools;
 (d) the power of the P3 (msplit) 2-SE bar and the readability of P1 under its own >= 0.001 criterion.
CPU only. Writes out/c2_03_pubresid.log (stdout)."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
import sys
from pathlib import Path
import numpy as np, polars as pl
from scipy.stats import norm
ROOT = Path("<repo>"); sys.path.insert(0, str(ROOT / "src"))
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
import calib
PP = ROOT / "nn_x/audit/publicprobe/out"
BOUND = 1790497920
fv, Mv, ov = feed_meta("val"); ft_, Mt, ot = feed_meta("test"); y = np.load(FLAT / "val/cand_target.npy")
tst = ft_["timestamp"].to_numpy().astype(np.int64); pub = tst < BOUND
tsv = fv["timestamp"].to_numpy().astype(np.int64)
import datetime as dt
def msk(t): return dt.datetime.fromtimestamp(int(t), dt.timezone(dt.timedelta(hours=3))).strftime("%a %Y-%m-%d %H:%M MSK")
print(f"(a) public boundary {BOUND} = {msk(BOUND)}; test range {msk(tst.min())} .. {msk(tst.max())}; public n {pub.sum()} private n {(~pub).sum()}")
hour_t = ((tst + 10800) // 3600) % 24; hour_v = ((tsv + 10800) // 3600) % 24
dow_t = ((tst + 10800) // 86400 + 3) % 7  # 1970-01-01 was a Thursday (=3 with Mon=0)
dow_v = ((tsv + 10800) // 86400 + 3) % 7
segv = pl.read_parquet(ROOT / "eda/errors/segments/feed_table.parquet"); segt = pl.read_parquet(ROOT / "eda/errors/segments/test_segments.parquet")
csv_, cst_ = segv["cold_share"].to_numpy(), segt["cold_share"].to_numpy()
for name, mv, mt in [("h14-24", hour_v >= 14, hour_t >= 14), ("h18-24", hour_v >= 18, hour_t >= 18), ("h00-10", hour_v < 10, hour_t < 10),
                     ("weekend", dow_v >= 5, dow_t >= 5), ("M<50", Mv < 50, Mt < 50), ("cold>.3", csv_ > 0.3, cst_ > 0.3), ("cold>.45", csv_ > 0.45, cst_ > 0.45)]:
    print(f"    {name:8s} share: val {mv.mean():.3f}  public {mt[pub].mean():.3f}  private {mt[~pub].mean():.3f}")
print(f"    test days by split: public {np.unique(dow_t[pub]).tolist()} (0=Mon) private {np.unique(dow_t[~pub]).tolist()}; mean M val {Mv.mean():.1f} public {Mt[pub].mean():.1f} private {Mt[~pub].mean():.1f}")
# (b) reproduce the pubmix table
g3 = np.load(ROOT / "preds/ens_G3_final_val.npy"); base = ndcg_per_feed(g3, y, ov, Mv)
Wpub = calib.weights("public"); Wpriv = calib.weights("private"); N_PUB = int(pub.sum())
def wstat(d, w):
    m = np.average(d, weights=w); v = np.average((d - m) ** 2, weights=w); return m, v
print("(b) pubmix expectation / public paired SE, recomputed (name: d_val ± SE_val | d_pubmix ± SE_pub)")
rows = {}
for nm in ["M2_check", "TOonly_check", "PF7x_check", "Cd_check", "NEW12", "wft100_FTonly", "msplit_M50", "Pall", "F10only", "N1_G3+NEW"]:
    s = np.load(PP / f"cand_{nm}_val.npy"); d = ndcg_per_feed(s, y, ov, Mv) - base
    mP, vP = wstat(d, Wpub); rows[nm] = (d.mean(), d.std(ddof=1) / np.sqrt(len(d)), mP, np.sqrt(vP / N_PUB))
    print(f"    {nm:14s} {d.mean():+.5f} ± {rows[nm][1]:.5f} | {mP:+.5f} ± {rows[nm][3]:.5f}")
# msplit slice form: exact public delta = (n_S_pub / N_PUB) * mean_S(d) with M-reweighting inside S
s = np.load(PP / "cand_msplit_M50_val.npy"); d = ndcg_per_feed(s, y, ov, Mv) - base; mv = Mv < 50; nS = int((Mt[pub] < 50).sum())
mS, vS = wstat(d[mv], Wpub[mv]); print(f"    msplit slice form: n_S_pub {nS}, expectation {nS / N_PUB * mS:+.5f}, SE {np.sqrt(nS * vS) / N_PUB:.5f}")
# (c) residual models
real = {"M2_check": 0.392434 - 0.392820, "TOonly_check": 0.392346 - 0.392820, "PF7x_check": 0.39272 - 0.39282, "Cd_check": 0.39208 - 0.39282, "S2": 0.392473 - 0.392820}
share = {"M2_check": 1.00, "TOonly_check": 0.70, "PF7x_check": 0.18, "Cd_check": 0.0, "S2": 0.70}
dval = {k: rows[k][0] for k in ["M2_check", "TOonly_check", "PF7x_check", "Cd_check"]}; dval["S2"] = 0.40088 - 0.40032
exp_pub = {k: rows[k][2] for k in ["M2_check", "TOonly_check", "PF7x_check", "Cd_check"]}; exp_pub["S2"] = dval["S2"]  # S2 has no stored array; use d_val as its pubmix
se_tot = {k: np.sqrt(rows[k][3] ** 2 + rows[k][1] ** 2) for k in ["M2_check", "TOonly_check", "PF7x_check", "Cd_check"]}; se_tot["S2"] = 0.0006
resid = {k: real[k] - exp_pub[k] for k in real}
print("(c) residual (realised - pubmix expectation): " + ", ".join(f"{k} {resid[k]:+.5f}" for k in resid))
# WLS through the G3 point (share 0.70, resid 0) on the three points the programme used
pts = ["Cd_check", "PF7x_check", "M2_check"]
x = np.array([share[k] - 0.70 for k in pts]); r = np.array([resid[k] for k in pts]); w = np.array([1 / se_tot[k] ** 2 for k in pts])
b = (w * x * r).sum() / (w * x * x).sum(); seb = 1 / np.sqrt((w * x * x).sum())
print(f"    programme's WLS (3 points through G3): slope {b:+.5f} ± {seb:.5f} per unit share (z {b/seb:+.1f})")
# the same WLS with the two share-0.70 points added (TO-only, S2): they sit at x = 0 and so cannot move the slope, but they
# are residuals that the share model predicts to be 0
print(f"    share model predicts resid 0 for TO-only and S2 (share 0.70); realised {resid['TOonly_check']:+.5f}, {resid['S2']:+.5f}")
# competing model: resid = a + c * d_val over all five scored pools (OLS and WLS)
ks = list(real); xv = np.array([dval[k] for k in ks]); rv = np.array([resid[k] for k in ks]); wv = np.array([1 / se_tot[k] ** 2 for k in ks])
def wls(x, r, w):
    X = np.column_stack([np.ones_like(x), x]); W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ r); cov = np.linalg.inv(X.T @ W @ X)
    return beta, np.sqrt(np.diag(cov))
beta, sb = wls(xv, rv, wv)
print(f"    shrinkage model (5 points): resid = {beta[0]:+.5f} ± {sb[0]:.5f} + ({beta[1]:+.3f} ± {sb[1]:.3f}) * d_val  "
      f"-> public delta = {beta[0]:+.5f} + {1+beta[1]:.3f} * d_val; corr(share, d_val) over the 5 pools {np.corrcoef([share[k] for k in ks], xv)[0,1]:+.3f}")
sh = [share[k] for k in ks]; beta2, sb2 = wls(np.array(sh) - 0.70, rv, wv)
print(f"    share model refit on all 5 (with intercept): resid = {beta2[0]:+.5f} ± {sb2[0]:.5f} + ({beta2[1]:+.5f} ± {sb2[1]:.5f}) * (share - 0.70)")
chi_share = ((rv - (np.array(sh) - 0.70) * b) ** 2 * wv).sum(); chi_shr = ((rv - (beta[0] + beta[1] * xv)) ** 2 * wv).sum()
print(f"    chi2 of the 5 residuals: share model (no intercept, slope {b:+.5f}) {chi_share:.1f} on 5 df; shrinkage model {chi_shr:.1f} on 3 df")
print("    predictions for the unscored probes (public delta vs G3):")
for nm, shr in [("NEW12", 0.0), ("Pall", 0.0), ("F10only", 0.0), ("N1_G3+NEW", 0.44)]:
    h1 = rows[nm][2]; h2 = h1 + b * (shr - 0.70); h3 = h1 + beta[0] + beta[1] * rows[nm][0]
    print(f"      {nm:10s} H1 1:1 {h1:+.5f} | H2 share {h2:+.5f} | shrinkage {h3:+.5f} | SE_pub {rows[nm][3]:.5f} -> H2 - shrinkage = {h2-h3:+.5f} ({(h2-h3)/rows[nm][3]:.1f} SE)")
# (d) power of P3's bar and P1's readability
e, se = rows["msplit_M50"][2], rows["msplit_M50"][3]
print(f"(d) P3 msplit: P(r >= +0.0007 | val expectation {e:+.5f}, SE {se:.5f}) = {1 - norm.cdf((0.0007 - e) / se):.2f}; "
      f"P(r >= +0.0007 | uniform null -0.00018) = {1 - norm.cdf((0.0007 + 0.00018) / se):.3f}; P(r <= +0.0002 | val) = {norm.cdf((0.0002 - e) / se):.2f}")
h1, h2 = rows["NEW12"][2], rows["NEW12"][2] + b * (0 - 0.70)
print(f"    P1 NEW12: H1 {h1:+.5f} vs H2 {h2:+.5f}: separation {h2-h1:+.5f} = {(h2-h1)/rows['NEW12'][3]:.2f} SE_pub (the programme's own readability bar is >= 0.001)")
print("done")
