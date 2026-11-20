"""Step 2: val -> public transfer with ALL scored files.
A. level line (18 files).  B. errors-in-variables GLS of public deltas on each offline predictor (modern 14; sub-blocks).
C. pair table, near-copy vs not.  Writes out/s2_transfer.log + out/s2_pairs.csv."""
from lib import *
import io, itertools
from scipy import stats
buf = io.StringIO()


def say(*a):
    print(*a, flush=True); print(*a, file=buf)


z = np.load(OUT / "nd_val.npz"); names = list(z["names"]); ND = dict(zip(names, z["nd"]))
zh = np.load(OUT / "nd_hold.npz"); HN = dict(zip(list(zh["names"]), zh["nd"]))
zz = np.load(OUT / "nd_val_halves.npz"); HFT = dict(zip(list(zz["names"]), zz["ft"])); HTO = dict(zip(list(zz["names"]), zz["to"]))
cz = np.load(OUT / "corr_test.npz"); CN = list(cz["names"]); CT = cz["all"]
corr = lambda a, b: CT[CN.index(a), CN.index(b)]
fv, Mv, ov = meta("val"); pub, tday, Mt = test_parts()
wpub = mix_weights(Mv, Mt[pub]); wpriv = mix_weights(Mv, Mt[~pub])
day = val_day(); tue = day == day.min(); thu = ~tue
say(f"val days: {np.unique(day, return_counts=True)}  (Tue n={tue.sum()}, Thu n={thu.sum()})")

# ---------------------------------------------------------------- A. level line
say("\n== A. LEVEL LINE public = a + c * val")
def ols(x, y):
    X = np.c_[np.ones(len(x)), x]; b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    s2 = (r ** 2).sum() / (len(x) - 2); cov = s2 * np.linalg.inv(X.T @ X)
    Hm = X @ np.linalg.inv(X.T @ X) @ X.T; loo = r / (1 - np.diag(Hm))
    return b, np.sqrt(np.diag(cov)), r, loo
allf = ["v3", "v4", "v5", "v10"] + MODERN
xv = np.array([ND[n].mean() if n in ND else VAL_REC[n] for n in allf]); yp = np.array([PUBLIC[n] for n in allf])
for tag, idx in [("all 18", np.arange(len(allf))), ("16 without W8/S16 (state of 11-13)", np.array([i for i, n in enumerate(allf) if n not in ("W8", "S16")])),
                 ("modern 14 only", np.array([i for i, n in enumerate(allf) if n in MODERN]))]:
    b, sb, r, loo = ols(xv[idx], yp[idx])
    say(f" {tag}: a {b[0]:+.4f}  c {b[1]:.3f} +- {sb[1]:.3f}  resid sd {r.std(ddof=2):.5f}  LOO rmse {np.sqrt((loo ** 2).mean()):.5f}")
    if tag.startswith("all"):
        for n, rr, ll in zip(np.array(allf)[idx], r, loo):
            say(f"    {n:7s} val {xv[allf.index(n)]:.5f} public {PUBLIC[n]:.6f} resid {rr:+.5f} LOO {ll:+.5f}")
b16, *_ = ols(xv[[i for i, n in enumerate(allf) if n not in ("W8", "S16")]], yp[[i for i, n in enumerate(allf) if n not in ("W8", "S16")]])
for n in ("W8", "S16"):
    pr = b16[0] + b16[1] * ND[n].mean(); say(f" out-of-sample (line fitted without W8/S16): {n} pred {pr:.5f} actual {PUBLIC[n]:.6f} miss {PUBLIC[n] - pr:+.5f}")
say(" fixed line 0.0078 + 0.962 val: " + "  ".join(f"{n} {PUBLIC[n] - (0.0078 + 0.962 * ND[n].mean()):+.5f}" for n in MODERN))
say(" M-reweighted val (src/calib public mix) minus public, files with arrays:")
off = {n: PUBLIC[n] - (wpub * ND[n]).mean() for n in MODERN + OLDARR}
say("   " + "  ".join(f"{n} {off[n]:+.5f}" for n in off))
om = np.array([off[n] for n in MODERN]); say(f"   modern 14: mean offset {om.mean():+.5f}  sd over files {om.std(ddof=1):.5f}")
sdv = np.mean([ND[n].std() for n in MODERN]); sdvw = np.sqrt(np.mean([np.average((ND[n] - np.average(ND[n], weights=wpub)) ** 2, weights=wpub) for n in MODERN]))
say(f"   per-feed NDCG sd (modern mean): val {sdv:.4f}, val under public M mix {sdvw:.4f} -> sampling SE of an ABSOLUTE level: val {sdv / np.sqrt(NVAL):.5f}, public {sdvw / np.sqrt(NPUB):.5f}, private {sdvw / np.sqrt(NPRIV):.5f}")
say(f"   => SE of (public level - M-reweighted val level) from sampling alone = {np.sqrt(sdvw ** 2 / NPUB + sdv ** 2 / NVAL):.5f}  (observed offset {om.mean():+.5f})")

# ---------------------------------------------------------------- B. EIV GLS on deltas
say("\n== B. PUBLIC DELTAS vs OFFLINE PREDICTORS (errors-in-variables GLS; public noise covariance from val per-feed NDCG under the public M mix)")
def cov_mean(Xm, w=None):
    """covariance of the (weighted) means of the rows of Xm [n_files x n_feeds]."""
    n = Xm.shape[1]
    if w is None:
        return np.cov(Xm) / n
    w = w / w.sum(); m = Xm @ w; R = (Xm - m[:, None]) * w[None, :]
    return R @ R.T
def pop_cov(Xm, w):
    w = w / w.sum(); m = Xm @ w; R = (Xm - m[:, None]) * np.sqrt(w)[None, :]
    return R @ R.T
def eiv(P_, X_, SP, SX, label, grid=np.linspace(-3, 4, 1401), verbose=True):
    n = len(P_); D = np.eye(n)[1:] - np.eye(n)[0]         # contrasts vs the first file (invariant to the choice)
    out = {}
    nll = []
    for k in grid:
        r = D @ (P_ - k * X_); C = D @ (SP + k * k * SX) @ D.T
        c2 = r @ np.linalg.solve(C, r); nll.append((0.5 * (c2 + np.linalg.slogdet(C)[1]), c2))
    nll = np.array(nll); i = nll[:, 0].argmin(); kh = grid[i]
    inside = grid[nll[:, 0] <= nll[i, 0] + 0.5]; in2 = grid[nll[:, 0] <= nll[i, 0] + 1.92]
    c0 = nll[np.abs(grid).argmin(), 1]; c1 = nll[np.abs(grid - 1).argmin(), 1]
    lr1 = 2 * (nll[np.abs(grid - 1).argmin(), 0] - nll[i, 0]); lr0 = 2 * (nll[np.abs(grid).argmin(), 0] - nll[i, 0])
    out = dict(k=kh, lo=inside.min(), hi=inside.max(), lo95=in2.min(), hi95=in2.max(), chi0=c0, chi1=c1, chik=nll[i, 1], df=n - 1,
               p_k1=1 - stats.chi2.cdf(lr1, 1), p_k0=1 - stats.chi2.cdf(lr0, 1))
    if verbose:
        say(f"  {label:26s} k {kh:+.2f} [1s {out['lo']:+.2f},{out['hi']:+.2f}] [95% {out['lo95']:+.2f},{out['hi95']:+.2f}]  chi2: k=0 {c0:5.1f}  k=1 {c1:5.1f}  k^ {nll[i, 1]:5.1f} /{n - 1}df  LR p(k=1) {out['p_k1']:.3f}  p(k=0) {out['p_k0']:.3f}")
    return out

def run_block(files, title, preds=("val", "val_pubM", "val_M<50", "val_M>=50", "val_Tue", "val_Thu", "val_FT", "val_TO", "holdout")):
    say(f"\n -- block {title}: {files}")
    A = np.stack([ND[n] for n in files]); P_ = np.array([PUBLIC[n] for n in files])
    SP = pop_cov(A, wpub) / NPUB
    say(f"    public: spread {P_.max() - P_.min():.5f}, sd over files {P_.std(ddof=1):.5f}; chi2 of 'all files equal on public' = {eiv(P_, np.zeros(len(files)), SP, np.zeros_like(SP), '', verbose=False)['chi0']:.1f} / {len(files) - 1} df")
    res = {}
    for pn in preds:
        fl = list(files)
        if pn == "val": X_ = A.mean(1); SX = cov_mean(A)
        elif pn == "val_pubM": X_ = A @ (wpub / wpub.sum()); SX = cov_mean(A, wpub)
        elif pn in ("val_M<50", "val_M>=50", "val_Tue", "val_Thu"):
            m = {"val_M<50": Mv < 50, "val_M>=50": Mv >= 50, "val_Tue": tue, "val_Thu": thu}[pn]
            X_ = A[:, m].mean(1); SX = cov_mean(A[:, m])
        elif pn in ("val_FT", "val_TO"):
            src = HFT if pn == "val_FT" else HTO; fl = [n for n in files if n in src]
            if len(fl) < 3: continue
            B = np.stack([src[n] for n in fl]); X_ = B.mean(1); SX = cov_mean(B)
        elif pn == "holdout":
            fl = [n for n in files if n in HN]
            if len(fl) < 3: continue
            B = np.stack([HN[n] for n in fl]); X_ = B.mean(1); SX = cov_mean(B)
        ix = [files.index(n) for n in fl]; Pf = P_[ix]; SPf = SP[np.ix_(ix, ix)]
        r = eiv(Pf, X_, SPf, SX, f"{pn} (n={len(fl)})")
        dP = Pf - Pf.mean(); dX = X_ - X_.mean()
        r["pearson"] = np.corrcoef(dP, dX)[0, 1]; r["spearman"] = stats.spearmanr(dP, dX)[0]
        r["rmse11"] = np.sqrt(((dP - dX) ** 2).mean()); r["rmse0"] = np.sqrt((dP ** 2).mean())
        kk = (dP @ dX) / (dX @ dX); r["ols"] = kk; r["rmse_ols"] = np.sqrt(((dP - kk * dX) ** 2).mean())
        # leave-one-file-out prediction with the OLS slope refitted without the file
        e = []
        for j in range(len(fl)):
            m = np.arange(len(fl)) != j; a = dP[m] - dP[m].mean(); b = dX[m] - dX[m].mean(); kj = (a @ b) / (b @ b)
            e.append((Pf[j] - Pf[m].mean()) - kj * (X_[j] - X_[m].mean()))
        r["loo"] = np.sqrt(np.mean(np.square(e))); e0 = [(Pf[j] - Pf[np.arange(len(fl)) != j].mean()) for j in range(len(fl))]; r["loo0"] = np.sqrt(np.mean(np.square(e0)))
        res[pn] = r
    say("    summary (centred over files): predictor | pearson | spearman | naive OLS slope | RMSE 1:1 | RMSE OLS | LOO rmse (slope refit) | 'no info' LOO")
    for pn, r in res.items():
        say(f"      {pn:10s} {r['pearson']:+.2f} {r['spearman']:+.2f}  ols {r['ols']:+.2f}  rmse11 {r['rmse11']:.5f}  rmse_ols {r['rmse_ols']:.5f}  loo {r['loo']:.5f}  loo0 {r['loo0']:.5f}")
    return res

G3FAM = ["Cd", "Pall", "PF7x", "G3", "M2", "S2", "TOonly", "Q1a", "Q2a", "C1", "C2", "C3"]
R_all = run_block(MODERN, "ALL modern 14")
R_fam = run_block(G3FAM, "G3-family 12 (state of 11-13; every file corr >= 0.995 to G3, S2 by rank)")
R_nosel = run_block([n for n in MODERN if n not in ("M2", "S2", "TOonly")], "modern 11 without val-selected M2/S2 and the FT-less TOonly")
R_hl = run_block(["Cd", "Pall", "PF7x", "G3", "Q1a", "Q2a", "C1", "C2", "C3"], "hard-label unselected 9")
R_kd = run_block(["C3", "W8", "S16"], "KD step: C3, W8, S16", preds=("val", "val_pubM", "val_M<50", "val_M>=50", "val_Tue", "val_Thu", "val_FT", "val_TO", "holdout"))
R_kd2 = run_block(["G3", "C3", "W8", "S16"], "G3, C3, W8, S16")

# two-predictor fit (val + holdout), GLS on public noise only (descriptive)
say("\n -- two-predictor GLS (public noise covariance only; predictors treated as exact; descriptive): public_d = kv*val_d + kh*hold_d")
for title, files in [("modern 13 (no S2)", [n for n in MODERN if n != "S2"]), ("unselected 10 (no M2,S2,TOonly; +W8,S16)", [n for n in MODERN if n not in ("M2", "S2", "TOonly")])]:
    A = np.stack([ND[n] for n in files]); P_ = np.array([PUBLIC[n] for n in files]); SP = pop_cov(A, wpub) / NPUB
    n = len(files); D = np.eye(n)[1:] - np.eye(n)[0]
    Xm = np.c_[A.mean(1), np.array([HN[f].mean() for f in files])]
    Ci = np.linalg.inv(D @ SP @ D.T); Xd = D @ Xm; yd = D @ P_
    cov = np.linalg.inv(Xd.T @ Ci @ Xd); bb = cov @ Xd.T @ Ci @ yd; r = yd - Xd @ bb
    say(f"    {title}: kv {bb[0]:+.2f} +- {np.sqrt(cov[0, 0]):.2f}  kh {bb[1]:+.2f} +- {np.sqrt(cov[1, 1]):.2f}  corr(kv,kh) {cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):+.2f}  chi2 {r @ Ci @ r:.1f}/{n - 3}")
    for j, lab in ((0, "val only"), (1, "holdout only")):
        x = Xd[:, [j]]; c = np.linalg.inv(x.T @ Ci @ x); b1 = c @ x.T @ Ci @ yd; r1 = yd - x @ b1
        say(f"       {lab}: k {b1[0]:+.2f} +- {np.sqrt(c[0, 0]):.2f}  chi2 {r1 @ Ci @ r1:.1f}/{n - 2}")
    say(f"       k=0 (all equal): chi2 {yd @ Ci @ yd:.1f}/{n - 1}")

# ---------------------------------------------------------------- C. pair table
say("\n== C. PAIRS: public delta vs val delta, near copies (test corr >= 0.995) vs the rest")
rows = []
pf = MODERN + OLDARR
for a, b in itertools.combinations(pf, 2):
    d = ND[a] - ND[b]; c = corr(a, b)
    if "S2" in (a, b):   # S2 is rank-valued: Pearson understates; use recorded Spearman 0.997 to G3 as near-copy of the G3 family
        c = 0.997 if (a in G3FAM and b in G3FAM) else c
    sdw = np.sqrt(np.average((d - np.average(d, weights=wpub)) ** 2, weights=wpub))
    rows.append(dict(a=a, b=b, corr=c, dval=d.mean(), se_val=se(d), dval_pubM=float((wpub * d).mean()), dpub=PUBLIC[a] - PUBLIC[b], se_pub=sdw / np.sqrt(NPUB),
                     dhold=(HN[a].mean() - HN[b].mean()) if (a in HN and b in HN) else np.nan,
                     se_hold=se(HN[a] - HN[b]) if (a in HN and b in HN) else np.nan))
T = pl.DataFrame(rows); T.write_csv(OUT / "s2_pairs.csv")
t = T.to_pandas()
t["se_tot"] = np.sqrt(t.se_val ** 2 + t.se_pub ** 2); t["z_resid"] = (t.dpub - t.dval) / t.se_tot
for lab, m in [("near copies corr>=0.995", t["corr"] >= 0.995), ("0.99<=corr<0.995", (t["corr"] < 0.995) & (t["corr"] >= 0.99)), ("corr<0.99 (old generations)", t["corr"] < 0.99)]:
    s = t[m]
    if len(s) == 0: continue
    k = (s.dpub * s.dval).sum() / (s.dval ** 2).sum()
    sig = s[np.abs(s.dval) > 2 * s.se_val]
    agree = (np.sign(s.dpub) == np.sign(s.dval)).mean(); agree_sig = (np.sign(sig.dpub) == np.sign(sig.dval)).mean() if len(sig) else np.nan
    say(f" {lab:28s} pairs {len(s):3d}  through-origin slope {k:+.2f}  sign agreement {agree:.2f}  (pairs with |dval|>2SE: {len(sig)}, agreement {agree_sig:.2f})  "
        f"mean |dval| {np.abs(s.dval).mean():.5f} mean |dpub| {np.abs(s.dpub).mean():.5f}  mean SE_pub {s.se_pub.mean():.5f}  rms z(dpub - dval) {np.sqrt((s.z_resid ** 2).mean()):.2f}")
    hh = s[~np.isnan(s.dhold)]
    if len(hh):
        kh = (hh.dpub * hh.dhold).sum() / (hh.dhold ** 2).sum()
        say(f" {'':28s} with holdout: {len(hh)} pairs, slope on holdout delta {kh:+.2f}, sign agreement {(np.sign(hh.dpub) == np.sign(hh.dhold)).mean():.2f}; corr(dpub,dval) {np.corrcoef(hh.dpub, hh.dval)[0, 1]:+.2f} corr(dpub,dhold) {np.corrcoef(hh.dpub, hh.dhold)[0, 1]:+.2f}")
say("\n pairs involving W8 / S16 (a - b):")
for _, r in t[(t.a.isin(["W8", "S16"])) | (t.b.isin(["W8", "S16"]))].iterrows():
    say(f"   {r.a:6s}-{r.b:6s} corr {r['corr']:.4f}  val {r.dval:+.5f}+-{r.se_val:.5f}  pubM-val {r.dval_pubM:+.5f}  hold {r.dhold:+.5f}+-{r.se_hold:.5f}  public {r.dpub:+.6f} (SE_pub {r.se_pub:.5f})  z(pub-val) {r.z_resid:+.2f}")
(OUT / "s2_transfer.log").write_text(buf.getvalue())
