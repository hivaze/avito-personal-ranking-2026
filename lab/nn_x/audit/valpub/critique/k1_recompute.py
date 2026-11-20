"""Adversarial critique of reports/VAL_PUBLIC_1113.md: independent recomputation of the decision-relevant numbers.
CPU only (12 threads). Reads preds/, preds/holdout/, data/flat, nn_x/pubdiag/shift/desc_*.parquet. Writes only out/ here.
Own code throughout (no import of the three analyses' libraries); the recorded val/holdout/public numbers are used as checks.
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "12"
import sys, json
from pathlib import Path
import numpy as np
import polars as pl
from scipy import stats
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; P = ROOT / "preds"
HERE = ROOT / "nn_x/audit/valpub/critique"; OUT = HERE / "out"; OUT.mkdir(exist_ok=True, parents=True)
BOUND = 1790497920; N_PUB = 8562
PUB = {"Cd": 0.39208, "Pall": 0.392436, "PF7x": 0.392723, "G3": 0.392820, "M2": 0.392434, "S2": 0.392473, "TOonly": 0.392346}
VALREC = {"Cd": 0.39901, "Pall": 0.39896, "PF7x": 0.39974, "G3": 0.40032, "M2": 0.40094, "S2": 0.40088, "TOonly": 0.40035}
HOLDREC = {"Cd": 0.32319, "Pall": 0.32335, "PF7x": 0.32360, "G3": 0.32379, "M2": 0.32361}
ORDER = ["Cd", "Pall", "PF7x", "G3", "M2", "S2", "TOonly"]
CON = ["Cd", "Pall", "PF7x", "M2", "S2", "TOonly"]
FAM = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1"] + [f"v14_s{i}" for i in range(92, 97)],
    "F7": [f"v17_pre2s_s{i}" for i in range(101, 108)],
    "F9": [f"v22_pre2s_lr1_s{i}" for i in range(401, 407)],
    "F10": [f"v23_gru_pre2s_s{i}" for i in range(501, 507)],
    "F12": [f"v26_d192_pre2s_s{i}" for i in range(701, 708)],
    "F7x": [f"r6_cd_m02_r{i}" for i in range(101, 108)],
    "F7x70": [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)],
}
HPOOLS = {"Cd": FAM["F5cd"] + FAM["F6"] + FAM["F7"],
          "Pall": FAM["F5cd"] + FAM["F6"] + FAM["F7"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
          "PF7x": FAM["F5cd"] + FAM["F6"] + FAM["F7x"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
          "G3": FAM["F10"] + FAM["F7x"] + FAM["F7x70"], "M2": FAM["F7x"] + FAM["F7x70"]}
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def zfeed(s, M):
    s = np.asarray(s, np.float64); n = len(M); fid = np.repeat(np.arange(n), M)
    mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]


fv = pl.read_parquet(FLAT / "val/feeds.parquet"); Mv = fv["M"].to_numpy().astype(np.int64); offv = fv["cand_off"].to_numpy().astype(np.int64)
yv = np.load(FLAT / "val/cand_target.npy")
ft = pl.read_parquet(FLAT / "test/feeds.parquet"); Mt = ft["M"].to_numpy().astype(np.int64); tst = ft["timestamp"].to_numpy().astype(np.int64)
pubmask = tst < BOUND
assert pubmask.sum() == N_PUB


def final_val(name):
    if name == "S2":
        return np.load(P / "nnstk_g3gate3r_val.npy").astype(np.float64)
    if name == "TOonly":
        return np.load(P / "ens_G3_toonly_val.npy").astype(np.float64)
    if name == "G3":
        return np.load(P / "ens_G3_final_val.npy").astype(np.float64)
    return 0.5 * zfeed(np.load(P / f"ens_{name}_ft_val.npy"), Mv) + 0.5 * zfeed(np.load(P / f"ens_{name}_to_val.npy"), Mv)


def pf_val(name):
    c = OUT / f"k1_pfval_{name}.npy"
    if c.exists():
        return np.load(c)
    x = ndcg_per_feed(final_val(name).astype(np.float32), yv, offv, Mv); np.save(c, x); return x


say("K1: INDEPENDENT RECOMPUTATION (critique of VAL_PUBLIC_1113)")
say("=" * 110)
pf = {k: pf_val(k) for k in ORDER}
say("val means (recomputed | recorded):")
for k in ORDER:
    say(f"   {k:7s} {pf[k].mean():.5f} | {VALREC[k]:.5f}  diff {pf[k].mean() - VALREC[k]:+.6f}")

# public M-mix weights (src/calib bins)
EDGES = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250])
bv = np.searchsorted(EDGES, Mv, side="right"); bt = np.searchsorted(EDGES, Mt[pubmask], side="right")
pv = np.bincount(bv, minlength=10) / len(bv); pt = np.bincount(bt, minlength=10) / len(bt)
w_pub = (pt / np.maximum(pv, 1e-12))[bv]; w_pub /= w_pub.mean()

D = np.stack([pf[k] - pf["G3"] for k in CON], 1)       # n_val x 6 paired contrasts
r = np.array([PUB[k] - PUB["G3"] for k in CON])         # exact public deltas
x_u = D.mean(0); se_u = D.std(0, ddof=1) / np.sqrt(len(D))
mw = (w_pub[:, None] * D).sum(0) / w_pub.sum()
Rw = D - mw
S_pub = (w_pub[:, None] * Rw).T @ Rw / w_pub.sum() / N_PUB       # public sampling cov (shares G3 -> off-diagonal)
S_val = np.cov(D.T, ddof=1) / len(D)                               # val sampling cov of the uniform means
S = S_pub + S_val
se_tot = np.sqrt(np.diag(S))
Si = np.linalg.inv(S)
say("\n--- N1. contrasts vs G3: val d, public d, SE_tot (full cov incl. shared G3) ---")
for i, k in enumerate(CON):
    say(f"   {k:7s} val {x_u[i]:+.5f}±{se_u[i]:.5f}  public {r[i]:+.5f}  SE_pub {np.sqrt(S_pub[i, i]):.5f}  SE_tot {se_tot[i]:.5f}  resid {r[i] - x_u[i]:+.5f} z {(r[i] - x_u[i]) / se_tot[i]:+.2f}")
corr = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
say("   correlation of the 6 contrasts' errors (shared G3):")
for i, k in enumerate(CON):
    say(f"   {k:7s} " + " ".join(f"{corr[i, j]:+.2f}" for j in range(6)))


def gls_k(x, r, S, m=None):
    m = np.ones(len(r), bool) if m is None else m
    Si = np.linalg.inv(S[np.ix_(m, m)]); xm, rm = x[m], r[m]
    k = (xm @ Si @ rm) / (xm @ Si @ xm); v = 1 / (xm @ Si @ xm)
    c2_1 = (rm - xm) @ Si @ (rm - xm); c2_k = (rm - k * xm) @ Si @ (rm - k * xm); c2_0 = rm @ Si @ rm
    return k, np.sqrt(v), c2_1, c2_k, c2_0


k, sek, c1, ck, c0 = gls_k(x_u, r, S)
say(f"\n   GLS transfer factor (full cov): k_v = {k:+.3f} ± {sek:.3f}; 95% CI [{k - 1.96 * sek:+.2f}, {k + 1.96 * sek:+.2f}]")
say(f"   chi2(1:1) = {c1:.2f} on 6 df (p {1 - stats.chi2.cdf(c1, 6):.3f});  chi2(k free) = {ck:.2f} on 5 df;  chi2(k=0) = {c0:.2f} on 6 df (p {1 - stats.chi2.cdf(c0, 6):.3f})")
say(f"   LR test k=1 vs free: dchi2 {c1 - ck:.2f}, p {1 - stats.chi2.cdf(c1 - ck, 1):.4f};  z = (1-k)/se = {(1 - k) / sek:.2f} (two-sided p {2 * (1 - stats.norm.cdf((1 - k) / sek)):.4f})")
# diagonal-only (what landscape/semantics did)
Sd = np.diag(np.diag(S)); kd, sekd, c1d, ckd, c0d = gls_k(x_u, r, Sd)
say(f"   same with DIAGONAL cov (ignoring the shared G3 draw): k = {kd:+.3f} ± {sekd:.3f}; chi2(1:1) {c1d:.2f}; LR dchi2 {c1d - ckd:.2f} p {1 - stats.chi2.cdf(c1d - ckd, 1):.4f}")
# with an intercept (G3's own draw as a free offset)
X = np.c_[np.ones(6), x_u]; A = X.T @ Si @ X; beta = np.linalg.solve(A, X.T @ Si @ r); cov = np.linalg.inv(A)
res = r - X @ beta; c2ab = res @ Si @ res
res1 = r - x_u; a1 = (np.ones(6) @ Si @ res1) / (np.ones(6) @ Si @ np.ones(6)); c2a1 = (res1 - a1) @ Si @ (res1 - a1)
say(f"   with intercept (full cov): a {beta[0]:+.5f} ± {np.sqrt(cov[0, 0]):.5f}, b {beta[1]:+.3f} ± {np.sqrt(cov[1, 1]):.3f}, chi2 {c2ab:.2f}/4;"
    f"  1:1 + offset: a {a1:+.5f}, chi2 {c2a1:.2f}/5 (p {1 - stats.chi2.cdf(c2a1, 5):.3f}); LR b=1 vs free: dchi2 {c2a1 - c2ab:.2f} p {1 - stats.chi2.cdf(c2a1 - c2ab, 1):.4f}")

say("\n   leave-one-contrast-out k_v (full cov):")
for i, kk in enumerate(CON):
    m = np.ones(6, bool); m[i] = False
    k_, se_, c1_, ck_, _ = gls_k(x_u, r, S, m)
    say(f"     drop {kk:7s}: k_v {k_:+.3f} ± {se_:.3f}  (1-k)/se {(1 - k_) / se_:.2f}  chi2(1:1) {c1_:.2f}/5")
for drop in [["M2", "S2"], ["M2", "S2", "TOonly"], ["Cd", "Pall"], ["Cd", "Pall", "PF7x"]]:
    m = np.array([c not in drop for c in CON])
    k_, se_, c1_, ck_, _ = gls_k(x_u, r, S, m)
    say(f"     drop {'+'.join(drop):16s}: k_v {k_:+.3f} ± {se_:.3f}  (1-k)/se {(1 - k_) / se_:.2f}  chi2(1:1) {c1_:.2f}/{m.sum()}  (remaining: {[c for c in CON if c not in drop]})")

# ---------------------------------------------------------------- N2. holdout
say("\n--- N2. holdout: per-feed NDCG of the train-only pools (rebuilt from preds/holdout) ---")
Mh, offh = np.load(P / "holdout/holdout_meta.npy"); Mh = Mh.astype(np.int64); offh = offh.astype(np.int64)
yh = np.load(P / "holdout/holdout_y.npy")
hp = {}
for name, mem in HPOOLS.items():
    c = OUT / f"k1_pfhold_{name}.npy"
    if c.exists():
        hp[name] = np.load(c); continue
    acc = None
    for mm in mem:
        z = zfeed(np.load(P / f"holdout/nn_{mm}_holdout.npy"), Mh); acc = z if acc is None else acc + z
    hp[name] = ndcg_per_feed((acc / len(mem)).astype(np.float32), yh, offh, Mh); np.save(c, hp[name])
for k_ in HPOOLS:
    say(f"   {k_:6s} holdout {hp[k_].mean():.5f} | recorded {HOLDREC[k_]:.5f}")
HC = ["Cd", "Pall", "PF7x", "M2"]
Hd = np.stack([hp[k_] - hp["G3"] for k_ in HC], 1); h_m = Hd.mean(0); S_h = np.cov(Hd.T, ddof=1) / len(Hd)
idx = [CON.index(k_) for k_ in HC]
S4 = S[np.ix_(idx, idx)]; r4 = r[idx]; x4 = x_u[idx]
say("   contrast  d_hold ± SE      d_val     public")
for j, k_ in enumerate(HC):
    say(f"   {k_:7s} {h_m[j]:+.5f}±{np.sqrt(S_h[j, j]):.5f}  {x4[j]:+.5f}  {r4[j]:+.5f}")


def chi2_of(pred, Scov, rr):
    e = rr - pred; return float(e @ np.linalg.inv(Scov) @ e)


Sh_tot = S4 + S_h
say("\n   chi2 of fixed predictors of the 4 public deltas (cov = public + val + holdout sampling, full):")
for lab, pred, Sc in [("holdout d (1:1, no parameter)", h_m, Sh_tot), ("zero (all pools = G3)", np.zeros(4), S4),
                      ("0.15 x val", 0.15 * x4, S4), ("0.3 x val", 0.3 * x4, S4), ("0.5 x val", 0.5 * x4, S4), ("1.0 x val", x4, S4),
                      ("0.5 holdout + 0.5 x 0.3 val", 0.5 * h_m + 0.15 * x4, 0.25 * S_h + S4)]:
    c2 = chi2_of(pred, Sc, r4)
    say(f"     {lab:34s} chi2 {c2:5.2f} on 4 df  p {1 - stats.chi2.cdf(c2, 4):.3f}   per-contrast z " + " ".join(f"{(r4[j] - pred[j]) / np.sqrt(Sc[j, j]):+.2f}" for j in range(4)))
say("   -> every predictor whose values lie within ±0.0005 of zero is accepted; chi2 differences of ~1-2 on 4 df are not evidence.")
# k_h and k_v on the same 4 contrasts, LOO predictions
kh, sekh, _, _, _ = gls_k(h_m, r4, Sh_tot); kv4, sekv4, _, _, _ = gls_k(x4, r4, S4)
say(f"   fitted k_h {kh:+.2f} ± {sekh:.2f};  k_v (same 4) {kv4:+.2f} ± {sekv4:.2f}")
say("   LOO out-of-sample z of each contrast: holdout(no param) | k*val (k fitted on the other 3) | k*holdout (k fitted on other 3)")
zz = {"hold": [], "kval": [], "khold": []}
for j, k_ in enumerate(HC):
    m = np.ones(4, bool); m[j] = False
    kv_, _, _, _, _ = gls_k(x4, r4, S4, m); kh_, _, _, _, _ = gls_k(h_m, r4, Sh_tot, m)
    z1 = (r4[j] - h_m[j]) / np.sqrt(Sh_tot[j, j]); z2 = (r4[j] - kv_ * x4[j]) / np.sqrt(S4[j, j]); z3 = (r4[j] - kh_ * h_m[j]) / np.sqrt(Sh_tot[j, j])
    zz["hold"].append(z1); zz["kval"].append(z2); zz["khold"].append(z3)
    say(f"     {k_:7s} {z1:+.2f} | {z2:+.2f} (k {kv_:+.2f}) | {z3:+.2f} (k {kh_:+.2f})")
say(f"   LOO sum z^2: holdout {np.sum(np.square(zz['hold'])):.2f} | k*val {np.sum(np.square(zz['kval'])):.2f} | k*holdout {np.sum(np.square(zz['khold'])):.2f}")
say("   same, dropping M2 from the comparison (3 contrasts Cd/Pall/PF7x):")
m3 = np.array([k_ != "M2" for k_ in HC])
for lab, pred, Sc in [("holdout d", h_m, Sh_tot), ("zero", np.zeros(4), S4), ("0.3 x val", 0.3 * x4, S4), ("0.5 x val", 0.5 * x4, S4), ("1.0 x val", x4, S4)]:
    c2 = chi2_of(pred[m3], Sc[np.ix_(m3, m3)], r4[m3]); say(f"     {lab:12s} chi2 {c2:5.2f} on 3 df p {1 - stats.chi2.cdf(c2, 3):.3f}")
kv3, sekv3, _, _, _ = gls_k(x4, r4, S4, m3); kh3, sekh3, _, _, _ = gls_k(h_m, r4, Sh_tot, m3)
say(f"     k_v (Cd/Pall/PF7x only) {kv3:+.2f} ± {sekv3:.2f};  k_h {kh3:+.2f} ± {sekh3:.2f}")

say("\n   absolute-level fits over the 5 pools with a holdout (public = a + c*x), LOO rmse; and the same without M2:")
pools5 = ["Cd", "Pall", "PF7x", "G3", "M2"]
pubs = np.array([PUB[k_] for k_ in pools5]); vals = np.array([pf[k_].mean() for k_ in pools5]); holds = np.array([hp[k_].mean() for k_ in pools5])


def loo_fit(x, y):
    errs = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        b = np.polyfit(x[m], y[m], 1); errs.append(y[i] - np.polyval(b, x[i]))
    b = np.polyfit(x, y, 1)
    return b[0], float(np.sqrt(np.mean(np.square(errs)))), np.array(errs)


for lab, m in [("5 pools", np.ones(5, bool)), ("without M2", np.array([k_ != "M2" for k_ in pools5])), ("without G3", np.array([k_ != "G3" for k_ in pools5]))]:
    ch, lh, eh = loo_fit(holds[m], pubs[m]); cv, lv, ev = loo_fit(vals[m], pubs[m])
    say(f"     {lab:11s} holdout: c {ch:+.2f} LOO rmse {lh:.5f} | val: c {cv:+.2f} LOO rmse {lv:.5f}   (n={m.sum()})")

# ---------------------------------------------------------------- N3. half-split null for the transfer slope
say("\n--- N3. is the non-transfer sampling noise? null distribution of the WLS slope under 'same pools, two independent samples' ---")


def wls_slope(dA, dB, nA, nB, wB=None):
    pv = dA.mean(0); rB = dB.mean(0) if wB is None else (wB[:, None] * dB).sum(0) / wB.sum()
    var = dA.var(0, ddof=1) / nA + dB.var(0, ddof=1) / nB
    return (pv * rB / var).sum() / (pv ** 2 / var).sum()


# realised, same (diagonal) formula: predictor = full val (n 14423), outcome = public (n 8562) with public-M variance
var_real = D.var(0, ddof=1) / len(D) + np.diag(S_pub)
b_real = (x_u * r / var_real).sum() / (x_u ** 2 / var_real).sum()
rng = np.random.default_rng(1)
bs_half, bs_pubsize = [], []
n = len(D)
pw = w_pub / w_pub.sum()
for it in range(1000):
    perm = rng.permutation(n); a, b_ = perm[: n // 2], perm[n // 2:]
    bs_half.append(wls_slope(D[a], D[b_], len(a), len(b_)))
    # design 2: 'public' = 8562 feeds drawn with the public-M probabilities (no replacement), 'val' = the complement (5861)
    sel = rng.choice(n, N_PUB, replace=False, p=pw); rest = np.setdiff1d(np.arange(n), sel)
    bs_pubsize.append(wls_slope(D[rest], D[sel], len(rest), len(sel)))
bs_half = np.array(bs_half); bs_pubsize = np.array(bs_pubsize)
say(f"   realised slope (diagonal WLS, as in semantics s05): b = {b_real:+.3f}")
say(f"   null A (two random halves 7211/7212, 1000 draws): mean {bs_half.mean():+.2f} sd {bs_half.std():.2f}; P(b <= {b_real:.2f}) = {np.mean(bs_half <= b_real):.3f}")
say(f"   null B (public-sized M-weighted draw 8562 vs complement 5861): mean {bs_pubsize.mean():+.2f} sd {bs_pubsize.std():.2f}; P(b <= {b_real:.2f}) = {np.mean(bs_pubsize <= b_real):.3f}")
say(f"   parametric (GLS, full cov): z = {(1 - k) / sek:.2f}, one-sided p {1 - stats.norm.cdf((1 - k) / sek):.4f}")
say("   NB: all three nulls assume the public per-feed delta variance equals val's (M-reweighted); they test 'same pools, same")
say("   population, independent sample'. Rejection says the val ordering is val-specific; it does NOT say whether the cause is")
say("   selection on val or a label-conditional difference between the val days and the public days.")

# public dispersion of the 7 scores vs expected
A7 = np.stack([pf[k_] for k_ in ORDER], 1); dev = A7 - A7.mean(1, keepdims=True)
var7 = np.array([np.average((dev[:, i] - np.average(dev[:, i], weights=w_pub)) ** 2, weights=w_pub) for i in range(7)]) / N_PUB
p7 = np.array([PUB[k_] for k_ in ORDER]); pd7 = p7 - p7.mean()
say(f"   dispersion of the 7 public scores around their mean: chi2 {(pd7 ** 2 / var7).sum():.2f} on 6 df (approx.; 'all 7 equal on public' is not rejected)")

# ---------------------------------------------------------------- N4. selection / winner's curse arithmetic
say("\n--- N4. winner's-curse arithmetic and what the 'val-selected indicator' actually is ---")
for K in [2, 5, 10, 20, 30, 50]:
    em = rng.standard_normal((100000, K)).max(1).mean()
    say(f"   K {K:3d}: E[max] {em:.2f} -> at SE_val 0.00034 (M2 vs G3) {em * 0.00034:+.5f}; at 0.00028 (S2) {em * 0.00028:+.5f}")
sel = np.array([1.0 if c in ("M2", "S2") else 0.0 for c in CON])
say(f"   corr(val-selected indicator, d_val) over the 6 contrasts = {np.corrcoef(sel, x_u)[0, 1]:+.2f}; corr(indicator, [d_val > 0]) = {np.corrcoef(sel, (x_u > 0).astype(float))[0, 1]:+.2f}")
say("   the indicator marks exactly the two pools with d_val > 0 (TOonly's +0.00003 aside): 'val-selected' and 'above G3 on val' are the same")
say("   partition of these 6 points, so the indicator model is a re-labelling of shrinkage toward G3, not an independent mechanism test.")
say("   M2 Tue/Thu val deltas vs G3: " + f"Tue {D[:, CON.index('M2')][(fv['timestamp'].to_numpy() + 10800) // 86400 < 20720].mean():+.5f}  Thu {D[:, CON.index('M2')][(fv['timestamp'].to_numpy() + 10800) // 86400 >= 20720].mean():+.5f}"
    " (pure selection on the val total inflates both days equally in expectation)")

# ---------------------------------------------------------------- N5. pick 2 and the combined 'private' estimate
say("\n--- N5. pick 2 (PF7x vs Pall) and the combined private estimate ---")
d = pf["PF7x"] - pf["Pall"]; dh = hp["PF7x"] - hp["Pall"]
se_pub_pp = np.sqrt(np.average((d - np.average(d, weights=w_pub)) ** 2, weights=w_pub) / N_PUB)
say(f"   PF7x - Pall: val {d.mean():+.5f} ± {d.std(ddof=1) / np.sqrt(len(d)):.5f} | holdout {dh.mean():+.5f} ± {dh.std(ddof=1) / np.sqrt(len(dh)):.5f} | public {PUB['PF7x'] - PUB['Pall']:+.5f} (model SE {se_pub_pp:.5f})")
say("   (confirms the report; PF7x > Pall on all three, the public one at ~1 SE)")
say("   combined 'private' estimate (calib 4e) = inverse-variance of holdout d and public d. For TOonly the holdout d is 0 with SE 0 by")
say("   construction (identical TO members): the holdout carries NO information about w_ft, yet the combination reports -0.00001 ± 0.00010.")
say("   Honest TOonly estimate = public alone: " + f"{r[CON.index('TOonly')]:+.5f} ± {se_tot[CON.index('TOonly')]:.5f}. Same caveat for S2 (holdout -0.00006 is a recorded OOS stacker reading, not a TO pool).")
for k_ in HC:
    j = HC.index(k_); i = CON.index(k_)
    wh, wp = 1 / S_h[j, j], 1 / S[i, i]
    comb = (wh * h_m[j] + wp * r[i]) / (wh + wp); sec = 1 / np.sqrt(wh + wp)
    say(f"   {k_:6s} holdout {h_m[j]:+.5f}±{np.sqrt(S_h[j, j]):.5f}  public {r[i]:+.5f}±{np.sqrt(S[i, i]):.5f}  -> combined {comb:+.5f} ± {sec:.5f}  (holdout weight {wh / (wh + wp):.2f}); P(>0) {1 - stats.norm.cdf(-comb / sec):.2f}")
say("   the combination gives the holdout 65-85% of the weight and treats it as an unbiased private estimate with sampling error only")
say("   (train period, M>=50 only, checkpoint selector, pool selector for G3 at 13:55): the P(beats G3) numbers carry no model uncertainty.")

(OUT / "k1_recompute.log").write_text("\n".join(LOG) + "\n")
json.dump({"k_v": k, "se_kv": sek, "chi2_11": c1, "chi2_k": ck, "k_diag": kd, "se_diag": sekd, "chi2_11_diag": c1d,
           "b_real": b_real, "p_half": float(np.mean(bs_half <= b_real)), "p_pubsize": float(np.mean(bs_pubsize <= b_real)),
           "x_val": x_u.tolist(), "r_pub": r.tolist(), "se_tot": se_tot.tolist(), "h_m": h_m.tolist(), "se_h": np.sqrt(np.diag(S_h)).tolist()},
          open(OUT / "k1_recompute.json", "w"), indent=1)
print("written", OUT / "k1_recompute.log")
