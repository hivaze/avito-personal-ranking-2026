"""Part 2: pre-specified reweighting search. Which val feed weighting w(feed) reproduces the 6 exact public deltas vs G3?
chi2 over the 6 contrasts with the full covariance (public sampling under w + val sampling of the weighted mean), LOO for
fitted 1-parameter families. Writes out/c2_reweight.log, out/c2_reweight.json."""
import json
from scipy import stats, optimize
from vplib import *  # noqa: F401,F403
import pmix, calib

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def expected_and_cov(D, w, pubvar_w=None):
    """weighted mean of the per-feed deltas (n x K) and the covariance of (public realised - expected):
    public sampling: Cov_w(d) / N_PUB (public feeds are drawn from the w-weighted val population);
    val sampling of the weighted mean: sum w_i^2 (d_i - m)(d_i - m)' / (sum w)^2."""
    w = np.asarray(w, float)
    e = (w[:, None] * D).sum(0) / w.sum()
    R = D - e
    Spub = ((pubvar_w if pubvar_w is not None else w)[:, None] * R).T @ R / (pubvar_w if pubvar_w is not None else w).sum() / N_PUB
    Sval = ((w ** 2)[:, None] * R).T @ R / w.sum() ** 2
    return e, Spub, Sval


def chi2_of(r, e, S):
    z = r - e
    Si = np.linalg.pinv(S)
    c2 = float(z @ Si @ z)
    return c2, z / np.sqrt(np.diag(S))


def main():
    R = {}
    Dv, Dt = desc("val"), desc("test"); pubm = Dt["ts"] < BOUND; privm = ~pubm
    pf = {k: val_perfeed(k) for k in ORDER}
    D = np.stack([pf[k] - pf["G3"] for k in CONTRASTS], 1)  # n x 6 per-feed val deltas vs G3
    r = np.array([pub_delta(k) for k in CONTRASTS])            # exact public deltas vs G3
    say("PART 2: REWEIGHTING SEARCH. Contrasts vs G3:", ", ".join(f"{k} {ri:+.5f}" for k, ri in zip(CONTRASTS, r)))
    say("chi2 = (r - e_w)' S^-1 (r - e_w), S = Cov_w(d)/8562 (public sampling) + val-sampling cov of the weighted mean; df = 6")
    say("(for fitted families df = 6 - p and LOO = fit on 5 contrasts, predict the 6th). The 6 contrasts share G3, so S is full.")

    # ------------------------------------------------------------- weightings (pre-specified list)
    W = {"uniform": np.ones(len(D))}
    W["M (calib public)"] = calib.weights("public")
    W["M x cold (pmix public)"] = pmix.weights()["public"]
    lost = {}
    for kind, lab in [("hour5", "hour5"), ("hour24", "hour24"), ("Mxhour", "M x hour5"), ("coldxhour", "cold x hour5"),
                      ("Mxcoldxhour", "M x cold x hour5"), ("Mxuser", "M x user"), ("user", "user (tr0/1/2-4/5+/uid0)"),
                      ("H", "H bins"), ("cih", "cand-cat-in-hist bins"), ("ncat", "n_cat bins"), ("topcat", "top category"),
                      ("luocold", "luo-cold bins"), ("Mxluocold", "M x luo-cold"), ("MxcoldxH", "M x cold x H")]:
        W[f"{lab} (public mix)"], lost[lab] = poststrat_weights(Dv, Dt, pubm, kind)
    subs = {
        "M<50 only": Dv["M"] < 50, "M>=50 only": Dv["M"] >= 50, "Tue only": Dv["tue"] > 0, "Thu only": Dv["tue"] == 0,
        "cold>.3 only": Dv["cold_share"] > 0.3, "cold<=.3 only": Dv["cold_share"] <= 0.3, "cold>=.45 only": Dv["cold_share"] >= 0.45,
        "cold<.15 only": Dv["cold_share"] < 0.15, "user0 only": Dv["uid0"] > 0,
        "tr0 (no train history, not uid0)": (Dv["n_train_feeds"] == 0) & (Dv["uid0"] == 0), "tr1+ (train history)": Dv["n_train_feeds"] > 0,
        "H>=192": Dv["H"] >= 192, "H<64": Dv["H"] < 64, "H=256 (capped)": Dv["H"] >= 256, "H<=19": Dv["H"] <= 19,
        "cih>.95": Dv["cand_cat_in_hist_share"] > 0.95, "cih<=.8": Dv["cand_cat_in_hist_share"] <= 0.8,
        "hour 14-24 only": Dv["hour"] >= 14, "hour 0-13 only": Dv["hour"] < 14, "hour 0-9 only": Dv["hour"] < 10,
        "luo_cold>.5": Dv["luo_cold_share"] > 0.5, "luo_cold<=.35": Dv["luo_cold_share"] <= 0.35,
    }
    for k, m in subs.items():
        W[k] = m.astype(float) / m.mean()

    say(f"\n=== 2a. fixed weightings (no fitted parameter): chi2(6), p, and the per-contrast z = (r - e)/SE ===")
    say(f"   {'weighting':38s} {'n_eff':>6s} {'chi2':>6s} {'p':>6s} | " + " ".join(f"{k:>8s}" for k in CONTRASTS) + " | expected e_w")
    rows = []
    for lab, w in W.items():
        e, Sp, Sv = expected_and_cov(D, w)
        c2, z = chi2_of(r, e, Sp + Sv)
        p = 1 - stats.chi2.cdf(c2, 6)
        say(f"   {lab:38s} {neff(w):6.0f} {c2:6.2f} {p:6.3f} | " + " ".join(f"{zi:+8.2f}" for zi in z) + " | " +
            " ".join(f"{ei:+.5f}" for ei in e))
        rows.append(dict(weighting=lab, neff=float(neff(w)), chi2=c2, p=p, z=z.tolist(), e=e.tolist()))
    R["fixed"] = rows
    best = sorted(rows, key=lambda d: d["chi2"])[:6]
    say("   best 6 by chi2: " + "; ".join(f"{d['weighting']} ({d['chi2']:.2f}, p {d['p']:.2f})" for d in best))
    say("   lost public mass (test cells without val support): " + ", ".join(f"{k} {v:.3f}" for k, v in lost.items() if v > 0) or "   no lost mass")

    # ------------------------------------------------------------- the pure-noise and pure-shrinkage references
    say("\n=== 2b. reference models ===")
    e0, Sp0, Sv0 = expected_and_cov(D, W["uniform"])
    c2, z = chi2_of(r, np.zeros(6), Sp0 + Sv0)
    say(f"   'public deltas are all zero' (no real differences among the 7): chi2(6) {c2:.2f} p {1 - stats.chi2.cdf(c2, 6):.3f}")
    c2u, _ = chi2_of(r, e0, Sp0 + Sv0)
    say(f"   uniform 1:1 transfer: chi2(6) {c2u:.2f} p {1 - stats.chi2.cdf(c2u, 6):.3f}")

    def fit_k(e, S, mask=None):
        m = np.ones(len(e), bool) if mask is None else mask
        Si = np.linalg.pinv(S[np.ix_(m, m)])
        k = (e[m] @ Si @ r[m]) / (e[m] @ Si @ e[m]); sek = 1 / np.sqrt(e[m] @ Si @ e[m])
        return k, sek

    for lab in ["uniform", "M x cold (pmix public)", "M x cold x hour5 (public mix)"]:
        e, Sp, Sv = expected_and_cov(D, W[lab]); S = Sp + Sv
        k, sek = fit_k(e, S)
        c2k, zk = chi2_of(r, k * e, S)
        # LOO over contrasts
        loo = []
        for i in range(6):
            m = np.arange(6) != i
            ki, _ = fit_k(e, S, m); loo.append((r[i] - ki * e[i]) / np.sqrt(S[i, i]))
        say(f"   shrinkage r = k * e_w [{lab}]: k {k:.2f} ± {sek:.2f}  chi2(5) {c2k:.2f} p {1 - stats.chi2.cdf(c2k, 5):.3f}  "
            f"LOO sum z^2 {np.sum(np.square(loo)):.2f}  LOO z " + " ".join(f"{x:+.2f}" for x in loo))
        R[f"shrink_{lab}"] = dict(k=k, se=sek, chi2=c2k, loo_z=loo)
        # regression-to-G3 with intercept: r = a + k e
        X = np.c_[np.ones(6), e]; Si = np.linalg.pinv(S)
        beta = np.linalg.solve(X.T @ Si @ X, X.T @ Si @ r); cov = np.linalg.inv(X.T @ Si @ X)
        c2b, _ = chi2_of(r, X @ beta, S)
        say(f"      with intercept: a {beta[0]:+.5f} ± {np.sqrt(cov[0, 0]):.5f}  k {beta[1]:.2f} ± {np.sqrt(cov[1, 1]):.2f}  chi2(4) {c2b:.2f} p {1 - stats.chi2.cdf(c2b, 4):.3f}")

    # ------------------------------------------------------------- 1-parameter exponential tilt families
    say("\n=== 2c. fitted 1-parameter tilt families w = exp(theta * x) on top of the public M x cold weights; LOO over contrasts ===")
    base = W["M x cold (pmix public)"]
    feats = {"cold_share": Dv["cold_share"], "luo_cold_share": Dv["luo_cold_share"], "logM": np.log(Dv["M"]),
             "hour/24": Dv["hour"] / 24, "H/256": Dv["H"] / 256, "cih": Dv["cand_cat_in_hist_share"],
             "tue": Dv["tue"], "uid0": Dv["uid0"], "tr0": ((Dv["n_train_feeds"] == 0) & (Dv["uid0"] == 0)).astype(float),
             "n_cat/20": Dv["n_cat"] / 20, "f1_mean": Dv["f1_mean"], "max_cat_share": Dv["max_cat_share"]}

    def chi2_theta(theta, x, mask=None, w0=base):
        w = w0 * np.exp(np.clip(theta * (x - x.mean()), -20, 20)); w = w / w.mean()
        e, Sp, Sv = expected_and_cov(D, w); S = Sp + Sv
        m = np.ones(6, bool) if mask is None else mask
        Si = np.linalg.pinv(S[np.ix_(m, m)]); z = (r - e)[m]
        return float(z @ Si @ z), e, S, w

    say(f"   {'tilt on':16s} {'theta':>7s} {'chi2(5)':>7s} {'p':>6s} {'n_eff':>6s} {'LOO sum z^2':>11s}  implied val mix: mean(x) under w vs uniform vs public")
    rows = []
    for lab, x in feats.items():
        res = optimize.minimize_scalar(lambda t: chi2_theta(t, x)[0], bounds=(-8, 8), method="bounded")
        c2, e, S, w = chi2_theta(res.x, x)
        loo = []
        for i in range(6):
            m = np.arange(6) != i
            ri = optimize.minimize_scalar(lambda t: chi2_theta(t, x, m)[0], bounds=(-8, 8), method="bounded")
            _, ei, Si_, _ = chi2_theta(ri.x, x)
            loo.append((r[i] - ei[i]) / np.sqrt(Si_[i, i]))
        xt = Dt["cold_share"] if lab == "cold_share" else None
        pubx = {"cold_share": Dt["cold_share"][pubm].mean(), "luo_cold_share": Dt["luo_cold_share"][pubm].mean(),
                "logM": np.log(Dt["M"][pubm]).mean(), "hour/24": Dt["hour"][pubm].mean() / 24, "H/256": Dt["H"][pubm].mean() / 256,
                "cih": Dt["cand_cat_in_hist_share"][pubm].mean(), "uid0": Dt["uid0"][pubm].mean(),
                "tr0": ((Dt["n_train_feeds"] == 0) & (Dt["uid0"] == 0))[pubm].mean(), "n_cat/20": Dt["n_cat"][pubm].mean() / 20,
                "f1_mean": Dt["f1_mean"][pubm].mean(), "max_cat_share": Dt["max_cat_share"][pubm].mean()}.get(lab, np.nan)
        say(f"   {lab:16s} {res.x:+7.2f} {c2:7.2f} {1 - stats.chi2.cdf(c2, 5):6.3f} {neff(w):6.0f} {np.sum(np.square(loo)):11.2f}  "
            f"{np.average(x, weights=w):.3f} vs {x.mean():.3f} vs public {pubx:.3f}")
        rows.append(dict(tilt=lab, theta=float(res.x), chi2=c2, neff=float(neff(w)), loo_sumz2=float(np.sum(np.square(loo))),
                         mean_x_w=float(np.average(x, weights=w)), mean_x_val=float(x.mean()), mean_x_pub=float(pubx)))
    R["tilt"] = rows
    say("   Reading: a tilt that 'fits' by pushing the mean of x far outside the public range is a label fit, not a composition")
    say("   match; a tilt whose fitted mean matches the public mean AND lowers chi2 is a genuine composition explanation.")

    # ------------------------------------------------------------- implied effective val for the best fixed weighting
    say("\n=== 2d. 'effective val' implied by the best-fitting fixed weightings vs the real public composition ===")
    def comp(w, Dx, m=None):
        m = np.ones(len(Dx["M"]), bool) if m is None else m
        f = lambda a: np.average(a[m], weights=(w[m] if w is not None else None))
        return dict(M=f(Dx["M"]), Mlt50=f((Dx["M"] < 50).astype(float)), cold=f(Dx["cold_share"]), cold_gt3=f((Dx["cold_share"] > .3).astype(float)),
                    cold_ge45=f((Dx["cold_share"] >= .45).astype(float)), luocold=f(Dx["luo_cold_share"]), h14=f((Dx["hour"] >= 14).astype(float)),
                    uid0=f(Dx["uid0"]), tr0=f(((Dx["n_train_feeds"] == 0) & (Dx["uid0"] == 0)).astype(float)), H=f(Dx["H"]),
                    cih=f(Dx["cand_cat_in_hist_share"]))
    keys = ["M", "Mlt50", "cold", "cold_gt3", "cold_ge45", "luocold", "h14", "uid0", "tr0", "H", "cih"]
    say("   " + f"{'composition':38s}" + " ".join(f"{k:>9s}" for k in keys))
    for lab, w, Dx, m in [("val uniform", None, Dv, None), ("public (true)", None, Dt, pubm), ("private (true)", None, Dt, privm)] + \
                         [(f"val under {d['weighting']}", W[d["weighting"]], Dv, None) for d in best[:4]]:
        c = comp(w, Dx, m)
        say("   " + f"{lab:38s}" + " ".join(f"{c[k]:9.3f}" for k in keys))
    R["best_fixed"] = [d["weighting"] for d in best]
    (OUT / "c2_reweight.json").write_text(json.dumps(R, indent=1, default=float))
    (OUT / "c2_reweight.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
