"""Part 1: val->public, holdout->public and combined calibrations with leave-one-out, over the scored files.
Writes out/c1_calib.log and out/c1_calib.json."""
import json
from vplib import *  # noqa: F401,F403
import pmix, calib  # src modules (M x cold public/private weights, M-only weights)

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def ols(x, y, slope=None):
    """OLS y = a + c x (or fixed slope). returns a, c, se_c, resid, sd"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if slope is None:
        X = np.c_[np.ones(len(x)), x]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        dof = max(len(x) - 2, 1)
        s2 = (r ** 2).sum() / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        return beta[0], beta[1], np.sqrt(cov[1, 1]), r, np.sqrt(s2)
    a = (y - slope * x).mean(); r = y - a - slope * x
    return a, slope, 0.0, r, np.sqrt((r ** 2).sum() / max(len(x) - 1, 1))


def loo(x, y, slope=None):
    """leave-one-out predictions and RMSE"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    pred = np.empty(len(x))
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        a, c, *_ = ols(x[m], y[m], slope)
        pred[i] = a + c * x[i]
    e = y - pred
    return pred, e, np.sqrt((e ** 2).mean())


def report_fit(label, names, x, y, slope=None):
    a, c, sec, r, sd = ols(x, y, slope)
    pred, e, rmse = loo(x, y, slope)
    say(f"-- {label}: n={len(x)}  a {a:+.5f}  c {c:.3f}" + (f" ± {sec:.3f}" if slope is None else " (fixed)") +
        f"  resid sd {sd:.5f}  LOO rmse {rmse:.5f}")
    for n, xi, yi, ri, li in zip(names, x, y, r, e):
        say(f"     {n:7s} x {xi:.5f} pub {yi:.5f} resid {ri:+.5f} LOO-err {li:+.5f}")
    return dict(a=a, c=c, se_c=sec, sd=sd, loo_rmse=rmse, resid=dict(zip(names, r.tolist())), loo=dict(zip(names, e.tolist())))


def main():
    R = {}
    say("PART 1: CALIBRATION val -> public, holdout -> public, combinations. Overlay excluded (control: public = G3's).")
    say("Public noise floor: paired SE between near-copies 0.0004-0.0007; absolute SE of one score ~0.0027 (common to all).")
    old_n = list(OLD); mod_n = ORDER
    xo = [OLD[k]["val"] for k in old_n]; yo = [OLD[k]["pub"] for k in old_n]
    xm = [MODERN[k]["val"] for k in mod_n]; ym = [MODERN[k]["pub"] for k in mod_n]

    say("\n=== 1a. public = a + c * val ===")
    R["all11"] = report_fit("all 11 (4 old + 7 modern)", old_n + mod_n, xo + xm, yo + ym)
    R["all11_slope1"] = report_fit("all 11, slope fixed 1", old_n + mod_n, xo + xm, yo + ym, slope=1.0)
    R["old5"] = report_fit("4 old + Cd (the pre-modern era)", old_n + ["Cd"], xo + [MODERN["Cd"]["val"]], yo + [MODERN["Cd"]["pub"]])
    R["mod7"] = report_fit("7 modern only", mod_n, xm, ym)
    R["mod7_slope1"] = report_fit("7 modern, slope fixed 1", mod_n, xm, ym, slope=1.0)
    R["mod7_slope0"] = report_fit("7 modern, slope fixed 0 (public does not follow val at all)", mod_n, xm, ym, slope=0.0)
    # out-of-sample: fit on old 4 + Cd, predict the other 6 modern
    a, c, sec, r, sd = ols(xo + [MODERN["Cd"]["val"]], yo + [MODERN["Cd"]["pub"]])
    say(f"-- fit on 4 old + Cd (a {a:+.5f} c {c:.3f}), predict the 6 later modern files:")
    oos = []
    for k in mod_n[1:]:
        pr = a + c * MODERN[k]["val"]; oos.append(MODERN[k]["pub"] - pr)
        say(f"     {k:7s} pred {pr:.5f} actual {MODERN[k]['pub']:.5f} miss {MODERN[k]['pub'] - pr:+.5f}")
    say(f"     mean miss {np.mean(oos):+.5f}  rmse {np.sqrt(np.mean(np.square(oos))):.5f}")
    R["oos_from_old"] = dict(a=a, c=c, miss=dict(zip(mod_n[1:], oos)))
    # modern slope: what is the slope of public on val among the 7 modern (near-copies)?
    say("\n     Reading: the 11-point slope is dominated by the old->modern level jump (0.38 -> 0.40 val); among the 7 modern files")
    say("     (val range 0.0020) the slope is a different quantity: the within-generation transfer of val differences.")

    say("\n=== 1b. holdout -> public (modern pools with a train-only holdout; TOonly shares G3's holdout by construction) ===")
    hn = [k for k in mod_n if MODERN[k]["hold"] is not None and k != "TOonly"]
    xh = [MODERN[k]["hold"] for k in hn]; yh = [MODERN[k]["pub"] for k in hn]
    R["hold5"] = report_fit("5 pools, public = a + c * holdout", hn, xh, yh)
    R["hold5_slope1"] = report_fit("5 pools, slope fixed 1", hn, xh, yh, slope=1.0)
    R["val5"] = report_fit("same 5 pools, public = a + c * val (for comparison)", hn, [MODERN[k]["val"] for k in hn], yh)
    R["val5_slope1"] = report_fit("same 5 pools, val slope fixed 1", hn, [MODERN[k]["val"] for k in hn], yh, slope=1.0)
    hn6 = hn + ["TOonly"]
    R["hold6"] = report_fit("6 incl. TOonly (holdout = G3's)", hn6, [MODERN[k]["hold"] for k in hn6], [MODERN[k]["pub"] for k in hn6])

    say("\n=== 1c. combinations on the 5 pools with holdout: public = a + c * (lam * val + (1 - lam) * holdout) ===")
    xv = np.array([MODERN[k]["val"] for k in hn]); xhh = np.array(xh); yy = np.array(yh)
    best = None
    for lam in np.round(np.arange(0, 1.01, 0.1), 2):
        x = lam * xv + (1 - lam) * xhh
        a, c, sec, r, sd = ols(x, yy); pred, e, rmse = loo(x, yy)
        say(f"   lam {lam:.1f}: c {c:.3f} ± {sec:.3f}  resid sd {sd:.5f}  LOO rmse {rmse:.5f}  LOO errs " +
            " ".join(f"{k}:{ei:+.5f}" for k, ei in zip(hn, e)))
        if best is None or rmse < best[1]:
            best = (lam, rmse)
    say(f"   best lam by LOO rmse: {best[0]} (rmse {best[1]:.5f})  [5 points: lam is NOT identifiable to better than ~±0.5]")
    R["combo_best_lam"] = best
    # 2-predictor OLS
    X = np.c_[np.ones(5), xv, xhh]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None); r = yy - X @ beta
    say(f"   free 2-predictor OLS: a {beta[0]:+.4f}  b_val {beta[1]:+.3f}  b_hold {beta[2]:+.3f}  resid sd {np.sqrt((r ** 2).sum() / 2):.5f} (2 dof)")
    pred = np.empty(5)
    for i in range(5):
        m = np.arange(5) != i
        b, *_ = np.linalg.lstsq(X[m], yy[m], rcond=None); pred[i] = X[i] @ b
    say(f"   free 2-predictor LOO rmse {np.sqrt(((yy - pred) ** 2).mean()):.5f}")

    say("\n=== 1d. reweighted / subset val as the single predictor of public (7 modern, per-feed arrays) ===")
    Dv, Dt = desc("val"), desc("test"); pubm = Dt["ts"] < BOUND
    pf = {k: val_perfeed(k) for k in mod_n}
    W = {}
    W["uniform"] = np.ones(len(Dv["M"]))
    W["M (calib.py public)"] = calib.weights("public")
    W["M x cold (pmix public)"] = pmix.weights()["public"]
    for kind, lab in [("hour5", "hour5 (public)"), ("Mxhour", "M x hour5 (public)"), ("coldxhour", "cold x hour5 (public)"),
                      ("Mxcoldxhour", "M x cold x hour5 (public)"), ("Mxuser", "M x user (public)"), ("Mxluocold", "M x luo-cold (public)"),
                      ("hour24", "hour24 (public)")]:
        W[lab], _ = poststrat_weights(Dv, Dt, pubm, kind)
    subs = {
        "M<50 only": Dv["M"] < 50, "M>=50 only": Dv["M"] >= 50, "Tue only": Dv["tue"] > 0, "Thu only": Dv["tue"] == 0,
        "cold>.3 only": Dv["cold_share"] > 0.3, "cold<=.3 only": Dv["cold_share"] <= 0.3, "cold>=.45 only": Dv["cold_share"] >= 0.45,
        "user0 only": Dv["uid0"] > 0, "no-train-history users (tr0, not uid0)": (Dv["n_train_feeds"] == 0) & (Dv["uid0"] == 0),
        "train-history users (tr1+)": Dv["n_train_feeds"] > 0, "H>=192": Dv["H"] >= 192, "H<64": Dv["H"] < 64,
        "cih>.95 (candidates in history cats)": Dv["cand_cat_in_hist_share"] > 0.95, "cih<=.8": Dv["cand_cat_in_hist_share"] <= 0.8,
    }
    for k, m in subs.items():
        W[k] = m.astype(float) / m.mean()
    rows = []
    say(f"   {'predictor':42s} {'n_eff':>6s} {'c':>7s} {'±':>6s} {'sd':>8s} {'LOOrmse':>8s} {'LOOrmse(slope1)':>15s}  d-vs-G3 corr with public d")
    r_pub = np.array([pub_delta(k) for k in CONTRASTS])
    for lab, w in W.items():
        x = np.array([np.average(pf[k], weights=w) for k in mod_n])
        a, c, sec, r, sd = ols(x, ym); _, _, rmse = loo(x, ym); _, _, rmse1 = loo(x, ym, slope=1.0)
        d = np.array([np.average(pf[k] - pf["G3"], weights=w) for k in CONTRASTS])
        cc = np.corrcoef(d, r_pub)[0, 1]
        say(f"   {lab:42s} {neff(w):6.0f} {c:7.3f} {sec:6.3f} {sd:8.5f} {rmse:8.5f} {rmse1:15.5f}  corr {cc:+.2f}")
        rows.append(dict(predictor=lab, neff=float(neff(w)), c=c, se_c=sec, sd=sd, loo_rmse=rmse, loo_rmse_slope1=rmse1, corr_delta=cc,
                         x=dict(zip(mod_n, x.tolist()))))
    R["reweighted"] = rows
    say("   NOTE: with 7 points and a free slope, LOO rmse below ~0.0004 is at the public noise floor; differences between")
    say("   predictors of < 0.0001 in LOO rmse are not meaningful. 'corr' is the correlation of the 6 val deltas vs G3 with the")
    say("   6 exact public deltas (sign and ordering only; 6 points).")

    say("\n=== 1e. what the 7 modern points say about slope (public spread / val spread) ===")
    say(f"   val spread (max-min) {max(xm) - min(xm):.5f}; public spread {max(ym) - min(ym):.5f}; ratio {(max(ym) - min(ym)) / (max(xm) - min(xm)):.2f}")
    say(f"   rank correlation val vs public over the 7: Spearman {np.corrcoef(np.argsort(np.argsort(xm)), np.argsort(np.argsort(ym)))[0, 1]:+.2f}")
    (OUT / "c1_calib.json").write_text(json.dumps(R, indent=1, default=float))
    (OUT / "c1_calib.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
