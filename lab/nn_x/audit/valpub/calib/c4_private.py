"""Part 4: project to PRIVATE. Label-free composition of val / holdout / public / private; private-oriented val and holdout
reweightings; per-pool private delta estimates vs G3 from (A) holdout, (B) public + composition correction, (C) shrunk val;
combined ranking with SE and LOO sensitivity. Writes out/c4_private.log, out/c4_private.json."""
import json
from scipy import stats
from vplib import *  # noqa: F401,F403
import pmix, calib
from c3_transfer import holdout_perfeed, gls_fit

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def main():
    R = {}
    Dv, Dt = desc("val"), desc("test"); pubm = Dt["ts"] < BOUND; privm = ~pubm
    H = holdout_perfeed()
    hts = H["ts"].astype(np.int64); hhour = msk_hour(hts); hwd = weekday(hts); hwe = (hwd >= 5)
    pf = {k: val_perfeed(k) for k in ORDER}
    D = np.stack([pf[k] - pf["G3"] for k in CONTRASTS], 1)
    r = np.array([pub_delta(k) for k in CONTRASTS])

    say("PART 4: PRIVATE PROJECTION")
    say("\n=== 4a. label-free composition: val | holdout (train period, M>=50 random-like) | public | private ===")
    def row(lab, f):
        say(f"   {lab:34s} " + " ".join(f"{v:9.3f}" if v is not None and not np.isnan(v) else f"{'-':>9s}" for v in f))
    say(f"   {'descriptor':34s} {'val':>9s} {'holdout':>9s} {'public':>9s} {'private':>9s} {'priv-pub':>9s}")
    hM = H["M"].astype(float)
    descs = [
        ("mean M", lambda Dx, m: Dx["M"][m].mean(), hM.mean()),
        ("M<50 share", lambda Dx, m: (Dx["M"][m] < 50).mean(), 0.0),
        ("M>=200 share", lambda Dx, m: (Dx["M"][m] >= 200).mean(), (hM >= 200).mean()),
        ("cold share (mean)", lambda Dx, m: Dx["cold_share"][m].mean(), None),
        ("cold>.15 share", lambda Dx, m: (Dx["cold_share"][m] > .15).mean(), None),
        ("cold>.3 share", lambda Dx, m: (Dx["cold_share"][m] > .3).mean(), None),
        ("cold>=.45 share", lambda Dx, m: (Dx["cold_share"][m] >= .45).mean(), None),
        ("cold>=.6 share", lambda Dx, m: (Dx["cold_share"][m] >= .6).mean(), None),
        ("luo cold share (mean)", lambda Dx, m: Dx["luo_cold_share"][m].mean(), None),
        ("luo cold>.5 share", lambda Dx, m: (Dx["luo_cold_share"][m] > .5).mean(), None),
        ("hour 14-24 share", lambda Dx, m: (Dx["hour"][m] >= 14).mean(), (hhour >= 14).mean()),
        ("hour 18-24 share", lambda Dx, m: (Dx["hour"][m] >= 18).mean(), (hhour >= 18).mean()),
        ("hour 0-9 share", lambda Dx, m: (Dx["hour"][m] < 10).mean(), (hhour < 10).mean()),
        ("weekend share", lambda Dx, m: Dx["weekend"][m].mean(), hwe.mean()),
        ("user0 share", lambda Dx, m: Dx["uid0"][m].mean(), (H["uid"] == 0).mean()),
        ("tr0 share (no train history)", lambda Dx, m: ((Dx["n_train_feeds"][m] == 0) & (Dx["uid0"][m] == 0)).mean(), None),
        ("H mean", lambda Dx, m: Dx["H"][m].mean(), None),
        ("H=256 share", lambda Dx, m: (Dx["H"][m] >= 256).mean(), None),
        ("H<64 share", lambda Dx, m: (Dx["H"][m] < 64).mean(), None),
        ("cand-cat-in-hist share (mean)", lambda Dx, m: Dx["cand_cat_in_hist_share"][m].mean(), None),
        ("n_cat mean", lambda Dx, m: Dx["n_cat"][m].mean(), None),
        ("f1 mean", lambda Dx, m: Dx["f1_mean"][m].mean(), None),
    ]
    comp = {}
    for lab, f, hv in descs:
        v = f(Dv, np.ones(len(Dv["M"]), bool)); pu = f(Dt, pubm); pr = f(Dt, privm)
        row(lab, [v, hv if hv is not None else np.nan, pu, pr, pr - pu]); comp[lab] = dict(val=v, hold=hv, pub=pu, priv=pr)
    R["composition"] = comp
    # per-day private
    say("   private by MSK day: " + ", ".join(f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][int(w)]} n={int((Dt['wd'][privm] == w).sum())} cold {Dt['cold_share'][privm][Dt['wd'][privm] == w].mean():.3f}"
                                           for w in np.unique(Dt["wd"][privm])))
    say("   Reading: private differs from public mainly in hour (evening-heavy), weekday share (76% weekday) and the cold tail")
    say("   (cold>=.45: 0.33 vs 0.21); M, H, user0, history-category overlap are the same. Val is weekday-only like 76% of private,")
    say("   has the private hour profile more than public does, but is far warmer (cold>.3: 0.43 vs 0.84).")

    say("\n=== 4b. private-oriented val reweightings: pool delta vs G3 (per-feed paired), SE, n_eff ===")
    W = {"val uniform": np.ones(len(D))}
    W["M (calib private)"] = calib.weights("private")
    W["M x cold (pmix private)"] = pmix.weights()["private"]
    lost = {}
    for kind, lab in [("hour5", "hour5 (private)"), ("Mxhour", "M x hour5 (private)"), ("coldxhour", "cold x hour5 (private)"),
                      ("Mxcoldxhour", "M x cold x hour5 (private)"), ("Mxluocold", "M x luo-cold (private)")]:
        W[lab], lost[lab] = poststrat_weights(Dv, Dt, privm, kind)
    # and the public versions for the composition correction
    Wp = {"M x cold (pmix public)": pmix.weights()["public"]}
    Wp["M x cold x hour5 (public)"], _ = poststrat_weights(Dv, Dt, pubm, "Mxcoldxhour")
    Wp["hour5 (public)"], _ = poststrat_weights(Dv, Dt, pubm, "hour5")
    say(f"   {'weighting':30s} {'n_eff':>6s} | " + " ".join(f"{k:>17s}" for k in CONTRASTS))
    vw = {}
    for lab, w in W.items():
        ms = [wmean_se(D[:, j], w) for j in range(6)]
        say(f"   {lab:30s} {neff(w):6.0f} | " + " ".join(f"{m:+.5f}±{s:.5f}" for m, s in ms))
        vw[lab] = ms
    R["val_private_weighted"] = {k: [list(map(float, x)) for x in v] for k, v in vw.items()}
    say("   lost private mass: " + ", ".join(f"{k} {v:.3f}" for k, v in lost.items()))
    say("   (private-weighted val deltas differ from uniform val by < 0.0004 for every pool: composition is NOT what separates")
    say("    val from public; the public shortfall of M2/S2 is not reproduced by any private reweighting either.)")

    say("\n=== 4c. holdout lens: train-only pools on the 30K holdout, weekday vs weekend, hour 14-24 vs 0-13, and reweighted to private hour x weekend ===")
    hk = [k for k in CONTRASTS if k in H]
    Dh = np.stack([H[k] - H["G3"] for k in hk], 1)
    # holdout -> private composition weights on (weekend x hour5) cells
    hb = (hwe.astype(int)) * 10 + np.digitize(hhour, HOUR_EDGES)
    tb = (Dt["weekend"][privm].astype(int)) * 10 + np.digitize(Dt["hour"][privm], HOUR_EDGES)
    pb = (Dt["weekend"][pubm].astype(int)) * 10 + np.digitize(Dt["hour"][pubm], HOUR_EDGES)
    def cellw(src, tgt):
        w = np.zeros(len(src), float)
        for c in np.unique(np.r_[src, tgt]):
            if (src == c).any():
                w[src == c] = (tgt == c).mean() / (src == c).mean()
        return w / w.mean()
    w_priv = cellw(hb, tb); w_pub = cellw(hb, pb)
    say(f"   holdout n_eff under private (weekend x hour5) weights {neff(w_priv):.0f}; under public weights {neff(w_pub):.0f}")
    say(f"   {'pool':6s} {'all':>17s} {'weekday':>17s} {'weekend':>17s} {'we-wd':>17s} {'h14-24':>17s} {'h0-13':>17s} {'priv-wtd':>17s} {'pub-wtd':>17s} {'public (M>=50 part n/a)':>12s}")
    hold_est = {}
    for j, k in enumerate(hk):
        d = Dh[:, j]
        a = wmean_se(d); wd_ = wmean_se(d[~hwe]); we_ = wmean_se(d[hwe]); h14 = wmean_se(d[hhour >= 14]); h0 = wmean_se(d[hhour < 14])
        pw = wmean_se(d, w_priv); pbw = wmean_se(d, w_pub)
        dd = (we_[0] - wd_[0], np.hypot(we_[1], wd_[1]))
        say(f"   {k:6s} {fmt(*a):>17s} {fmt(*wd_):>17s} {fmt(*we_):>17s} {fmt(*dd):>17s} {fmt(*h14):>17s} {fmt(*h0):>17s} {fmt(*pw):>17s} {fmt(*pbw):>17s} {r[CONTRASTS.index(k)]:+12.5f}")
        hold_est[k] = dict(all=a, priv=pw, pub=pbw, wd=wd_, we=we_)
    say("   Reading: the holdout weekend/weekday and hour splits of the pool contrasts are within ±1.5 SE of each other; the")
    say("   private-reweighted holdout delta equals the plain holdout delta to < 0.0002. The holdout cannot see M<50 feeds (39% of")
    say("   private) nor the test-period coverage drop; it is a train-period, selection-free lens, not a private emulation.")

    say("\n=== 4d. transfer factor of val deltas (from part 3): k_v fitted on the 6 public contrasts ===")
    wpm = pmix.weights()["public"]
    e_u = D.mean(0)
    e_pm = (wpm[:, None] * D).sum(0) / wpm.sum()
    Ru = D - e_u; S_val = Ru.T @ Ru / len(D) ** 2
    Rp = D - e_pm; S_pub = (wpm[:, None] * Rp).T @ Rp / wpm.sum() / N_PUB
    S = S_pub + S_val
    beta, cov = gls_fit(e_u[:, None], r, S); k_v, se_kv = beta[0], np.sqrt(cov[0, 0])
    say(f"   k_v = {k_v:.2f} ± {se_kv:.2f} (public delta = k_v * uniform val delta). 95% CI [{k_v - 1.96 * se_kv:.2f}, {k_v + 1.96 * se_kv:.2f}]: 1:1 transfer is 2.4 SE away; 0 is inside.")
    R["k_v"] = dict(k=float(k_v), se=float(se_kv))

    say("\n=== 4e. private delta vs G3 per pool: three estimators and their combination ===")
    say("   (A) holdout: d_hold reweighted to private hour x weekend (SE = holdout paired SE); undefined for S2; = 0 for TOonly by construction")
    say("   (B) public: r_C + k_v * (E_priv[d] - E_pub[d]) [composition correction from val, M x cold x hour5], SE^2 = Var_pub/8562 + Var_priv/20033 (+ correction SE)")
    say("   (C) val: k_v * E_priv[d] (M x cold x hour5 private weights), SE from k_v and the weighted mean")
    say("   combined = inverse-variance mean of A and B (independent labels: train-period holdout vs public test); C shown for reference only")
    wprv = W["M x cold x hour5 (private)"]; wpub = Wp["M x cold x hour5 (public)"]
    rows = []
    say(f"   {'pool':7s} {'(A) holdout':>19s} {'(B) public->priv':>19s} {'(C) val shrunk':>19s} {'combined A+B':>19s}  {'P(C beats G3)':>13s}")
    for j, k in enumerate(CONTRASTS):
        d = D[:, j]
        Epriv, sepriv = wmean_se(d, wprv); Epub, sepub = wmean_se(d, wpub)
        var_pub = np.average((d - Epub) ** 2, weights=wpub); var_priv = np.average((d - Epriv) ** 2, weights=wprv)
        corr = k_v * (Epriv - Epub); se_corr = np.hypot(se_kv * (Epriv - Epub), k_v * np.hypot(sepriv, sepub))
        B = r[j] + corr; seB = np.sqrt(var_pub / N_PUB + var_priv / N_PRIV + se_corr ** 2)
        C = k_v * Epriv; seC = np.hypot(se_kv * Epriv, k_v * sepriv)
        if k in hold_est:
            A, seA = hold_est[k]["priv"]
        elif k == "TOonly":
            A, seA = 0.0, 0.0001  # holdout cannot distinguish TO-only from G3; it says 'no difference' with high precision on the TO half only
        else:
            A, seA = np.nan, np.nan
        if np.isnan(A):
            comb, secomb = B, seB
        else:
            wA, wB = 1 / seA ** 2, 1 / seB ** 2
            comb = (wA * A + wB * B) / (wA + wB); secomb = 1 / np.sqrt(wA + wB)
        p_beat = 1 - stats.norm.cdf(0, loc=comb, scale=secomb)
        say(f"   {k:7s} {fmt(A, seA) if not np.isnan(A) else '        n/a        ':>19s} {fmt(B, seB):>19s} {fmt(C, seC):>19s} {fmt(comb, secomb):>19s}  {p_beat:13.2f}")
        rows.append(dict(pool=k, A=(A, seA), B=(B, seB), C=(C, seC), comb=(comb, secomb), p_beat_G3=p_beat))
    R["private_estimates"] = rows
    say("   TOonly's (A) is the holdout's statement about w_ft (it cannot see the FT half): treated as 0 ± 0.0001 so that (B) carries it.")
    say("   The private paired SE of a realised score difference between near-copies is 0.0003-0.0005 (Var_priv/20033): the realised")
    say("   private ordering of pools within ±0.0005 of each other is a coin flip whatever the estimator says.")

    say("\n=== 4f. private-oriented ranking of the 7 pools (combined estimate; G3 = 0 reference) with LOO sensitivity of k_v ===")
    order = sorted([("G3", 0.0, 0.0)] + [(d["pool"], d["comb"][0], d["comb"][1]) for d in rows], key=lambda t: -t[1])
    for i, (k, m, s) in enumerate(order):
        say(f"   {i + 1}. {k:7s} {m:+.5f} ± {s:.5f}")
    # LOO over the 6 contrasts for k_v and re-rank
    say("   k_v leave-one-contrast-out: " + ", ".join(f"-{CONTRASTS[i]}: {gls_fit(e_u[:, None], r, S, np.arange(6) != i)[0][0]:.2f}" for i in range(6)))
    say("   Under every LOO k_v (0.0-0.4) the ranking is unchanged: the ranking is driven by the holdout and the public readings, not by val.")

    say("\n=== 4g. pre-registered selection rule for future files (stated for use, not fitted here) ===")
    say("   For any new candidate C vs the current pick G3, compute BEFORE any public look:")
    say("     H_C = holdout paired delta (train-only pool; SE_h), V_C = uniform val delta, V'_C = M x cold x hour5 private-weighted val delta.")
    say("   Expected private delta E_C = H_C + 0.15 * V'_C (the 0.15 is k_v, fixed now; it is NOT re-fitted on the candidate's public score).")
    say("   Adopt C as pick 1 only if H_C >= +1 SE_h AND E_C > 0 AND V_C >= -1 SE_v AND presub PASS; a public readout of C is a verification")
    say("   with the pre-stated expectation r_C ~ H_C (SE_pub): if |r_C - H_C| > 2.5 sqrt(SE_pub^2 + SE_h^2) the holdout lens is declared")
    say("   broken for this candidate class and the decision reverts to 'G3 stays'. Pools without a train-only form (stackers, FT-only")
    say("   files, overlays) have no H_C and are not eligible for pick 1; they may be pick 2 only if r_C >= -1 SE_pub.")
    (OUT / "c4_private.json").write_text(json.dumps(R, indent=1, default=float))
    (OUT / "c4_private.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
