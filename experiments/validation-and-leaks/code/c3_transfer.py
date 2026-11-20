# Asks what explains the gap between the val and the public differences of six pools against the
# best pool (G3). It fits one-parameter models by generalised least squares (shrinkage of the val
# difference, share of re-cooled members, fine-tune weight, "chosen on val", val minus holdout)
# and tests the holdout difference as a direct predictor of the public difference.
# How it was run: CPU, 2026-11-12 23:59 UTC, from lab/nn_x/audit/valpub/calib; the command line is
# not on record. Its helper vplib.py (same lab folder) is not copied.
# What it produced: out/c3_transfer.{log,json}; the log is results/val_public_transfer.txt
# (transfer factor of val differences 0.16 +- 0.34; holdout = public at 1:1 gives chi2(5) 0.72).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/audit/valpub/calib/c3_transfer.py
"""Part 3: what explains the val->public residuals of the 6 contrasts vs G3? Competing 1-parameter models fitted by GLS
with the full covariance, compared by chi2 and leave-one-contrast-out; plus the holdout as a direct predictor of public.
Writes out/c3_transfer.log, out/c3_transfer.json."""
import json
from scipy import stats
from vplib import *  # noqa: F401,F403
import pmix

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


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
HPOOLS = {"Cd": FAM["F5cd"] + FAM["F6"] + FAM["F7"], "Pall": FAM["F5cd"] + FAM["F6"] + FAM["F7"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
          "PF7x": FAM["F5cd"] + FAM["F6"] + FAM["F7x"] + FAM["F9"] + FAM["F10"] + FAM["F12"], "G3": FAM["F10"] + FAM["F7x"] + FAM["F7x70"],
          "M2": FAM["F7x"] + FAM["F7x70"]}


def holdout_perfeed():
    """per-feed holdout NDCG of the 5 train-only pools (cached) + holdout meta (ts, uid, M)."""
    c = OUT / "pf_holdout.npz"
    if c.exists():
        z = np.load(c); return {k: z[k] for k in z.files}
    Mh, offh = np.load(P / "holdout/holdout_meta.npy"); Mh = Mh.astype(np.int64); offh = offh.astype(np.int64)
    y = np.load(P / "holdout/holdout_y.npy")
    out = {}
    for name, mem in HPOOLS.items():
        acc = None
        for m in mem:
            z = zfeed(np.load(P / f"holdout/nn_{m}_holdout.npy"), Mh)
            acc = z if acc is None else acc + z
        out[name] = ndcg_per_feed((acc / len(mem)).astype(np.float32), y, offh, Mh)
    z = np.load(ROOT / "nn_x/pubdiag/shift/p1_holdout_perfeed.npz")
    out["ts"] = z["ts"]; out["uid"] = z["uid"]; out["M"] = Mh
    # sanity vs the pubdiag arrays
    for k in ["Cd", "PF7x", "G3", "M2"]:
        assert abs(out[k].mean() - z[k].mean()) < 1e-6, (k, out[k].mean(), z[k].mean())
    np.savez(c, **out)
    return out


def gls_fit(X, r, S, mask=None):
    m = np.ones(len(r), bool) if mask is None else mask
    Si = np.linalg.pinv(S[np.ix_(m, m)]); Xm = X[m]
    A = Xm.T @ Si @ Xm; beta = np.linalg.solve(A, Xm.T @ Si @ r[m]); cov = np.linalg.inv(A)
    return beta, cov


def main():
    R = {}
    Dv = desc("val")
    pf = {k: val_perfeed(k) for k in ORDER}
    D = np.stack([pf[k] - pf["G3"] for k in CONTRASTS], 1)
    r = np.array([pub_delta(k) for k in CONTRASTS])
    wpm = pmix.weights()["public"]
    # covariance of the public deltas (public sampling under the pmix public weights) + val sampling of the uniform mean
    e_u = D.mean(0)
    Ru = D - e_u
    S_pub = (wpm[:, None] * (D - (wpm[:, None] * D).sum(0) / wpm.sum())).T @ (D - (wpm[:, None] * D).sum(0) / wpm.sum()) / wpm.sum() / N_PUB
    S_val = Ru.T @ Ru / len(D) ** 2
    S = S_pub + S_val
    se = np.sqrt(np.diag(S))
    e_pm = (wpm[:, None] * D).sum(0) / wpm.sum()
    say("PART 3: TRANSFER-RATIO ANATOMY. 6 contrasts vs G3; resid = public delta - val expectation (uniform val; pubmix in brackets).")
    say(f"   {'contrast':8s} {'d_val':>9s} {'d_pubmix':>9s} {'public':>9s} {'resid':>9s} {'SE':>8s} {'z':>6s}")
    for i, k in enumerate(CONTRASTS):
        say(f"   {k:8s} {e_u[i]:+9.5f} {e_pm[i]:+9.5f} {r[i]:+9.5f} {r[i] - e_u[i]:+9.5f} {se[i]:8.5f} {(r[i] - e_u[i]) / se[i]:+6.2f}  [{(r[i] - e_pm[i]) / se[i]:+.2f}]")

    # ---------------------------------------------------------------- predictors per contrast
    H = holdout_perfeed()
    d_hold = np.array([H[k].mean() - H["G3"].mean() if k in H else (0.0 if k == "TOonly" else np.nan) for k in CONTRASTS])
    se_hold = np.array([(H[k] - H["G3"]).std() / np.sqrt(len(H["G3"])) if k in H else 0.0 for k in CONTRASTS])
    share = np.array([MODERN[k]["share"] - 0.70 for k in CONTRASTS])
    wft = np.array([MODERN[k]["wft"] - 0.5 for k in CONTRASTS])
    sel = np.array([MODERN[k]["sel"] for k in CONTRASTS], float)
    disc = e_u - d_hold  # val - holdout discrepancy (nan for S2)
    # FT / TO halves (val) for the 4 pools that have them; S2 / TOonly undefined
    d_ft = np.array([val_half_perfeed(k, "ft").mean() - val_half_perfeed("G3", "ft").mean() if k in HPOOLS else np.nan for k in CONTRASTS])
    d_to = np.array([val_half_perfeed(k, "to").mean() - val_half_perfeed("G3", "to").mean() if k in HPOOLS else np.nan for k in CONTRASTS])
    say("\n=== 3a. predictors per contrast (what each model uses) ===")
    say(f"   {'contrast':8s} {'d_val':>9s} {'d_hold':>9s} {'±':>8s} {'val-hold':>9s} {'share-.7':>9s} {'wft-.5':>7s} {'valsel':>6s} {'d_ft':>9s} {'d_to':>9s}")
    for i, k in enumerate(CONTRASTS):
        say(f"   {k:8s} {e_u[i]:+9.5f} {d_hold[i]:+9.5f} {se_hold[i]:8.5f} {disc[i]:+9.5f} {share[i]:+9.2f} {wft[i]:+7.2f} {sel[i]:6.0f} {d_ft[i]:+9.5f} {d_to[i]:+9.5f}")
    say("   (TOonly's holdout = G3's by construction: the holdout only sees train-only members; S2 has no train-only form.)")

    preds = {"generic shrinkage: r = k*d_val (k-1 = slope)": e_u,
             "re-cool share: resid = b*(share-0.70)": share,
             "FT weight: resid = f*(wft-0.5)": wft,
             "val-selected indicator: resid = s*sel": sel,
             "val-holdout discrepancy: resid = -g*(d_val-d_hold)": disc,
             "FT-half excess: resid = h*(d_ft-d_to)": d_ft - d_to}
    say("\n=== 3b. correlation of the predictor vectors across the contrasts (collinearity; nan-free pairs only) ===")
    names = list(preds)
    for i, a in enumerate(names):
        row = []
        for j, b in enumerate(names):
            x, y = preds[a], preds[b]; m = ~(np.isnan(x) | np.isnan(y))
            row.append(np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 2 and x[m].std() > 0 and y[m].std() > 0 else np.nan)
        say(f"   {a[:28]:28s} " + " ".join(f"{v:+5.2f}" if not np.isnan(v) else "   na" for v in row))

    say("\n=== 3c. 1-parameter residual models (GLS, no intercept): resid_i = beta * x_i; chi2 over the usable contrasts; LOO ===")
    resid = r - e_u
    rows = []
    for lab, x in preds.items():
        m = ~np.isnan(x)
        if lab.startswith("generic"):
            # r = k * e_u  <=> resid = (k-1) e_u
            X = x[:, None]
            beta, cov = gls_fit(X, r, S, m); k = beta[0]; sek = np.sqrt(cov[0, 0])
            fitted = k * x
            z = (r - fitted)[m]; Si = np.linalg.pinv(S[np.ix_(m, m)]); c2 = float(z @ Si @ z)
            loo = []
            for i in np.where(m)[0]:
                mm = m.copy(); mm[i] = False
                b, _ = gls_fit(X, r, S, mm); loo.append((r[i] - b[0] * x[i]) / se[i])
            say(f"   {lab:52s} beta {k:+.3f} ± {sek:.3f}  chi2({m.sum() - 1}) {c2:5.2f} p {1 - stats.chi2.cdf(c2, m.sum() - 1):.3f}  LOO sum z^2 {np.sum(np.square(loo)):5.2f}  LOO z " + " ".join(f"{v:+.2f}" for v in loo))
            rows.append(dict(model=lab, beta=k, se=sek, chi2=c2, df=int(m.sum() - 1), loo=float(np.sum(np.square(loo)))))
            continue
        X = x[:, None]
        beta, cov = gls_fit(X, resid, S, m); b = beta[0]; seb = np.sqrt(cov[0, 0])
        z = (resid - b * x)[m]; Si = np.linalg.pinv(S[np.ix_(m, m)]); c2 = float(z @ Si @ z)
        loo = []
        for i in np.where(m)[0]:
            mm = m.copy(); mm[i] = False
            if x[mm].std() == 0 and abs(x[mm]).sum() == 0:
                loo.append((resid[i]) / se[i]); continue
            bb, _ = gls_fit(X, resid, S, mm); loo.append((resid[i] - bb[0] * x[i]) / se[i])
        say(f"   {lab:52s} beta {b:+.5f} ± {seb:.5f}  chi2({m.sum() - 1}) {c2:5.2f} p {1 - stats.chi2.cdf(c2, m.sum() - 1):.3f}  LOO sum z^2 {np.sum(np.square(loo)):5.2f}  LOO z " + " ".join(f"{v:+.2f}" for v in loo))
        rows.append(dict(model=lab, beta=b, se=seb, chi2=c2, df=int(m.sum() - 1), loo=float(np.sum(np.square(loo))), n=int(m.sum())))
    # null: resid = 0 (1:1 transfer), and resid = const
    z = resid; c2 = float(z @ np.linalg.pinv(S) @ z)
    say(f"   {'1:1 transfer (resid = 0, no parameter)':52s} chi2(6) {c2:5.2f} p {1 - stats.chi2.cdf(c2, 6):.3f}  sum z^2 (no fit) {np.sum(np.square(resid / se)):5.2f}")
    X = np.ones((6, 1)); beta, cov = gls_fit(X, resid, S); z = resid - beta[0]; c2 = float(z @ np.linalg.pinv(S) @ z)
    say(f"   {'constant offset: resid = a':52s} a {beta[0]:+.5f} ± {np.sqrt(cov[0, 0]):.5f}  chi2(5) {c2:5.2f} p {1 - stats.chi2.cdf(c2, 5):.3f}")
    R["models"] = rows
    say("   Reading: every 1-parameter model is accepted (chi2 small) because the 6 residuals are only ~1.5-2 SE from 0 in total;")
    say("   LOO sum z^2 ranks them, but with 5-6 points a difference of < ~2 in LOO sum z^2 is not evidence.")

    say("\n=== 3d. the holdout as a direct predictor of the public delta (the holdout is selection-free and train-period) ===")
    m = ~np.isnan(d_hold)
    S_h = S.copy()
    for i in range(6):
        S_h[i, i] += se_hold[i] ** 2  # add holdout sampling noise (ignoring the small cross-covariances)
    z = (r - d_hold)[m]; Si = np.linalg.pinv(S_h[np.ix_(m, m)]); c2 = float(z @ Si @ z)
    say(f"   r = d_hold (1:1, no parameter), {m.sum()} contrasts: chi2({m.sum()}) {c2:.2f} p {1 - stats.chi2.cdf(c2, m.sum()):.3f}; per-contrast z " +
        " ".join(f"{k}:{(r[i] - d_hold[i]) / np.sqrt(S_h[i, i]):+.2f}" for i, k in enumerate(CONTRASTS) if m[i]))
    X = d_hold[:, None]; beta, cov = gls_fit(np.nan_to_num(X), r, S_h, m)
    z = (r - beta[0] * np.nan_to_num(d_hold))[m]; c2 = float(z @ Si @ z)
    loo = []
    for i in np.where(m)[0]:
        mm = m.copy(); mm[i] = False
        b, _ = gls_fit(np.nan_to_num(X), r, S_h, mm); loo.append((r[i] - b[0] * d_hold[i]) / np.sqrt(S_h[i, i]))
    say(f"   r = k_h * d_hold: k_h {beta[0]:+.2f} ± {np.sqrt(cov[0, 0]):.2f}  chi2({m.sum() - 1}) {c2:.2f}  LOO sum z^2 {np.sum(np.square(loo)):.2f}  LOO z " + " ".join(f"{v:+.2f}" for v in loo))
    # val vs holdout as predictors on the SAME 5 contrasts
    X = e_u[:, None]; beta_v, cov_v = gls_fit(X, r, S, m)
    z = (r - beta_v[0] * e_u)[m]; c2v = float(z @ np.linalg.pinv(S[np.ix_(m, m)]) @ z)
    say(f"   same 5 contrasts, r = k_v * d_val: k_v {beta_v[0]:+.2f} ± {np.sqrt(cov_v[0, 0]):.2f}  chi2(4) {c2v:.2f}")
    # two-predictor: r = k_v d_val + k_h d_hold
    X2 = np.c_[e_u, np.nan_to_num(d_hold)]; beta2, cov2 = gls_fit(X2, r, S_h, m)
    z = (r - X2 @ beta2)[m]; c22 = float(z @ Si @ z)
    say(f"   r = k_v*d_val + k_h*d_hold: k_v {beta2[0]:+.2f} ± {np.sqrt(cov2[0, 0]):.2f}  k_h {beta2[1]:+.2f} ± {np.sqrt(cov2[1, 1]):.2f}  chi2(3) {c22:.2f}  corr(k_v,k_h) {cov2[0, 1] / np.sqrt(cov2[0, 0] * cov2[1, 1]):+.2f}")
    R["holdout"] = dict(k_h=float(beta[0]), se_kh=float(np.sqrt(cov[0, 0])), k_v=float(beta_v[0]), se_kv=float(np.sqrt(cov_v[0, 0])),
                        two=dict(k_v=float(beta2[0]), k_h=float(beta2[1])))

    say("\n=== 3e. the val-only excess: pools' val delta vs G3 split into the FT half and the TO half (4 pools), and the holdout ===")
    say(f"   {'pool':6s} {'d_val':>9s} {'d_ft':>9s} {'d_to':>9s} {'d_hold':>9s} {'public':>9s}   to->public ratio   hold->public ratio")
    for i, k in enumerate(CONTRASTS):
        if k in HPOOLS:
            say(f"   {k:6s} {e_u[i]:+9.5f} {d_ft[i]:+9.5f} {d_to[i]:+9.5f} {d_hold[i]:+9.5f} {r[i]:+9.5f}   {r[i] / d_to[i] if d_to[i] != 0 else np.nan:+8.2f}           {r[i] / d_hold[i] if d_hold[i] != 0 else np.nan:+8.2f}")

    say("\n=== 3f. what 6 contrasts (5 with holdout) cannot decide ===")
    say("   - share, val-selection, FT-half excess and d_val are collinear (3b): any of them 'explains' the same ~-0.0010 of M2/S2.")
    say("   - A 1-parameter model is accepted whenever its predictor is roughly ordered like (M2, S2) < (TOonly) < (Cd, Pall, PF7x).")
    say("   - Only a contrast that breaks the collinearity (a share-0 pool that is val-BETTER than G3, or a val-selected pool with a")
    say("     low share, or an FT-only file) can separate them; none is scored. The holdout needs no such separation: it predicts")
    say("     public directly at k_h ~ 1 on the 5 contrasts where it exists, and it is the only predictor that does.")
    (OUT / "c3_transfer.json").write_text(json.dumps(R, indent=1, default=float))
    (OUT / "c3_transfer.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
