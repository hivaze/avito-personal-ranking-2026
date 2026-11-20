# Three checks of whether val can order near-identical pools: (a) the spread of 7 pools on public
# against the sampling noise expected from val; (b) the null distribution of the val-to-public
# slope from 400 random half-splits of val; (c) val against the holdout for train-only pools.
# How it was run: CPU, 2026-11-13 00:06 UTC, from lab/nn_x/audit/valpub/semantics; the command line
# is not on record. Its helpers lib.py and shiftlib.py are not copied.
# What it produced: out/s05_dispersion.log, copied as results/val_public_dispersion.txt (the 7
# pools differ on val at chi2 49.1 on 6 df and on public at 3.1; a slope as low as the realised
# 0.10 occurs in 1.5% of the half-splits).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/audit/valpub/semantics/s05_dispersion.py
"""Part 2/4: (a) dispersion of the 7 modern pools' public scores vs the sampling noise expected from val per-feed data; (b) the
within-val analogue of the val->public transfer: random half-splits of val (n ~7.2K vs public 8.6K) give the null
distribution of the transfer slope b when nothing but sampling noise separates the two samples; Tue->Thu as the 2-day
time-split analogue; (c) val-TO -> holdout slope (selection-free train-period sample). Writes out/s05_dispersion.log."""
from lib import *  # noqa: F401,F403
import io
sys.path.insert(0, str(SHIFT))
from shiftlib import holdout_index, holdout_pool, FAM  # noqa: E402

buf = io.StringIO()


def say(*a):
    print(*a); print(*a, file=buf)


dv = descriptors("val"); fv, Mv, offv = feed_meta("val")
dt = descriptors("test"); ft, Mt, offt = feed_meta("test")
pub = dt["public"].to_numpy() == 1; n_pub = int(pub.sum())
PF = {p: per_feed(p) for p in MODERN}
# public-mix weights (M9 bins, as calib.py)
EDGES_M = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250])
kv = np.searchsorted(EDGES_M, Mv, side="right"); kt = np.searchsorted(EDGES_M, Mt, side="right")
w_pub, _ = cell_weights(kv, kt, pub)

say("=" * 120)
say("5A. DISPERSION of the 7 modern pools on public vs expected sampling noise. Each pool's deviation from the 7-pool mean; expected")
say("    variance from val per-feed (NDCG_i - mean_j NDCG_j), M-reweighted to public, / n_pub. chi2 on 6 df. Same statistic on val.")
say("=" * 120)
A = np.column_stack([PF[p] for p in MODERN]); dev = A - A.mean(1, keepdims=True)
var_pub = np.array([np.average((dev[:, i] - np.average(dev[:, i], weights=w_pub)) ** 2, weights=w_pub) for i in range(7)]) / n_pub
pubv = np.array([PUBLIC[p] for p in MODERN]); pdev = pubv - pubv.mean()
chi_pub = float((pdev ** 2 / var_pub).sum())
var_val = dev.var(0, ddof=1) / len(Mv); vdev = A.mean(0) - A.mean()
chi_val = float((vdev ** 2 / var_val).sum())
from scipy.stats import chi2  # noqa: E402
say(f"  public: deviations " + " ".join(f"{p} {d:+.5f}" for p, d in zip(MODERN, pdev)) + f" | sd {pdev.std(ddof=1):.5f} vs expected mean SE {np.sqrt(var_pub).mean():.5f}")
say(f"  chi2 public {chi_pub:.2f} on 6 df (p_low {chi2.cdf(chi_pub, 6):.3f}: small = pools MORE alike on public than val noise predicts)")
say(f"  val:    deviations " + " ".join(f"{p} {d:+.5f}" for p, d in zip(MODERN, vdev)) + f" | sd {vdev.std(ddof=1):.5f} vs SE {np.sqrt(var_val).mean():.5f}")
say(f"  chi2 val {chi_val:.2f} on 6 df (p {1 - chi2.cdf(chi_val, 6):.4f}: val differences are real differences ON VAL)")
say("  Reading: on val the 7 pools differ far beyond noise; on public they do not differ beyond noise, and if anything less than")
say("  noise (the 7 share most members, so the realised public deviations are correlated; the test is approximate).")

say("")
say("=" * 120)
say("5B. NULL DISTRIBUTION of the transfer slope under pure sampling noise: val split into two random halves (A: 'val', B: 'public'");
say("    analogue, n_B = 7211); b = WLS slope of d_B on d_A over the 6 contrasts vs G3. Also the Tue -> Thu split (time, 2 days).")
say("    Realised val -> public: b = +0.10 +- 0.32 (s04, raw), chi2(b=1) 10.85/6.")
say("=" * 120)
D = np.column_stack([PF[p] - PF["G3"] for p in MODERN if p != "G3"])
rng = np.random.default_rng(0)


def slope(dA, dB, nA, nB):
    pv = dA.mean(0); r = dB.mean(0)
    var = dA.var(0, ddof=1) / nA + dB.var(0, ddof=1) / nB
    b = (pv * r / var).sum() / (pv ** 2 / var).sum(); se_b = 1 / np.sqrt((pv ** 2 / var).sum())
    chi = (((r - pv) ** 2) / var).sum()
    return b, se_b, chi


bs, chis = [], []
for it in range(400):
    perm = rng.permutation(len(Mv)); a = perm[:7212]; b_ = perm[7212:]
    b, se_b, chi = slope(D[a], D[b_], len(a), len(b_)); bs.append(b); chis.append(chi)
bs = np.array(bs); chis = np.array(chis)
say(f"  random halves (400 draws): b mean {bs.mean():+.2f} sd {bs.std():.2f} q05 {np.quantile(bs, .05):+.2f} q95 {np.quantile(bs, .95):+.2f}; P(b <= 0.10) = {np.mean(bs <= 0.10):.3f}; chi2(b=1) mean {chis.mean():.2f} P(chi2 >= 10.85) = {np.mean(chis >= 10.85):.3f}")
day = dv["day"].to_numpy(); tue = np.where(day == 0)[0]; thu = np.where(day == 2)[0]
b, se_b, chi = slope(D[tue], D[thu], len(tue), len(thu))
say(f"  Tue -> Thu: b {b:+.2f} ± {se_b:.2f}, chi2(b=1) {chi:.2f}/6 | Thu -> Tue: b {slope(D[thu], D[tue], len(thu), len(tue))[0]:+.2f}")
say("  Tue/Thu deltas: " + "  ".join(f"{p} {D[tue, i].mean():+.5f}/{D[thu, i].mean():+.5f}" for i, p in enumerate([p for p in MODERN if p != "G3"])))
say("  Reading: if P(b <= 0.10) under the null is small, the failure of val deltas to transfer to public is not sampling noise of the")
say("  two samples; it is a property of the pools (val-fitted / val-selected components) or of the public period.")

say("")
say("=" * 120)
say("5C. SELECTION-FREE CHECK: val TO-half deltas vs the 30K train-period holdout (train-only pools, no val fitting, no selection on")
say("    either sample for the reference pools). Slope of holdout delta on val-TO delta over the 4 contrasts with holdout pools.")
say("=" * 120)
y = np.load(FLAT / "val/cand_target.npy")
hold, fh, Mh, offh = holdout_index(); yh = np.load(P / "holdout/holdout_y.npy")
HP = {"Cd": FAM["F5cd"] + FAM["F6"] + FAM["F7"], "Pall": FAM["F5cd"] + FAM["F6"] + FAM["F7"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
      "PF7x": FAM["F5cd"] + FAM["F6"] + FAM["F7x"] + FAM["F9"] + FAM["F10"] + FAM["F12"], "G3": FAM["F10"] + FAM["F7x"] + FAM["F7x70"],
      "M2": FAM["F7x"] + FAM["F7x70"]}
PFH = {k: ndcg_per_feed(holdout_pool(v, Mh), yh, offh, Mh) for k, v in HP.items()}
PFT = {k: ndcg_per_feed(np.load(P / f"ens_{k}_to_val.npy"), y, offv, Mv) for k in ["Cd", "Pall", "PF7x", "G3", "M2"]}
m50 = Mv >= 50
say(f"  {'contrast':10s} {'val TO (all M)':>18s} {'val TO (M>=50)':>18s} {'holdout':>18s} {'public (mix pools)':>18s}")
pv, rh, var = [], [], []
for p in ["Cd", "Pall", "PF7x", "M2"]:
    dvto = PFT[p] - PFT["G3"]; dh = PFH[p] - PFH["G3"]
    a1 = wmean_se(dvto); a2 = wmean_se(dvto[m50]); a3 = wmean_se(dh)
    say(f"  {p + '-G3':10s} {a1[0]:+.5f}±{a1[1]:.5f}  {a2[0]:+.5f}±{a2[1]:.5f}  {a3[0]:+.5f}±{a3[1]:.5f}  {PUBLIC[p] - PUBLIC['G3']:+.5f}")
    pv.append(a2[0]); rh.append(a3[0]); var.append(a2[1] ** 2 + a3[1] ** 2)
pv, rh, var = map(np.array, (pv, rh, var))
b = (pv * rh / var).sum() / (pv ** 2 / var).sum(); se_b = 1 / np.sqrt((pv ** 2 / var).sum())
say(f"  slope holdout-on-valTO(M>=50): b {b:+.2f} ± {se_b:.2f}; chi2(b=1) {(((rh - pv) ** 2) / var).sum():.2f}/4")
say("  Reading: the holdout is a train-period production-random sample (M>=50), scored by train-only pools that were never fitted or")
say("  selected on it; its slope on val says how much of the val ordering of TO pools is val-specific (re-cool selection etc.).")

(OUT / "s05_dispersion.log").write_text(buf.getvalue())
print("written", OUT / "s05_dispersion.log")
