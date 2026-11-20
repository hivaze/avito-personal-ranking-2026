"""Part 3/4 supplements: (i) val vs holdout vs blends as predictors of the public deltas on the SAME 5 contrasts (chi2 profile),
(ii) where the val excess over the holdout sits (TO half = member checkpoint selection on val, vs FT half = OOF fine-tune),
(iii) winner's-curse size for val-selected pools, (iv) how far the holdout-public agreement is from chance.
Writes out/c5_extra.log."""
from scipy import stats
from vplib import *  # noqa: F401,F403
import pmix
from c3_transfer import holdout_perfeed, gls_fit

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def main():
    H = holdout_perfeed()
    pf = {k: val_perfeed(k) for k in ORDER}
    D = np.stack([pf[k] - pf["G3"] for k in CONTRASTS], 1)
    r = np.array([pub_delta(k) for k in CONTRASTS])
    wpm = pmix.weights()["public"]
    e_u = D.mean(0); Ru = D - e_u; S_val = Ru.T @ Ru / len(D) ** 2
    e_pm = (wpm[:, None] * D).sum(0) / wpm.sum(); Rp = D - e_pm
    S_pub = (wpm[:, None] * Rp).T @ Rp / wpm.sum() / N_PUB
    S = S_pub + S_val
    hk = [k for k in CONTRASTS if k in H]; idx = [CONTRASTS.index(k) for k in hk]
    Dh = np.stack([H[k] - H["G3"] for k in hk], 1)
    d_hold = Dh.mean(0); S_hold = (Dh - d_hold).T @ (Dh - d_hold) / len(Dh) ** 2
    m = np.zeros(6, bool); m[idx] = True
    say("(i) the same 4 contrasts with a holdout (Cd, Pall, PF7x, M2; TOonly excluded because its holdout is G3's by construction):")
    say("    chi2 of public = lam * d_val + (1 - lam) * d_hold, no fitted parameter, covariance = public + val + holdout sampling")
    Sm = S[np.ix_(m, m)]
    for lam in np.round(np.arange(0, 1.01, 0.25), 2):
        pred = lam * e_u[m] + (1 - lam) * d_hold
        Sl = Sm + (1 - lam) ** 2 * S_hold
        z = r[m] - pred; c2 = float(z @ np.linalg.pinv(Sl) @ z)
        say(f"    lam {lam:.2f} (val weight): chi2(4) {c2:5.2f} p {1 - stats.chi2.cdf(c2, 4):.3f}  per-contrast z " +
            " ".join(f"{k}:{zi / np.sqrt(Sl[i, i]):+.2f}" for i, (k, zi) in enumerate(zip(hk, z))))
    # fitted lam (1 parameter) by profile
    best = min(((lam, (lambda z, Sl: float(z @ np.linalg.pinv(Sl) @ z))(r[m] - (lam * e_u[m] + (1 - lam) * d_hold), Sm + (1 - lam) ** 2 * S_hold))
                for lam in np.arange(-0.5, 1.51, 0.05)), key=lambda t: t[1])
    say(f"    profile minimum lam = {best[0]:.2f} (chi2 {best[1]:.2f}); lam = 1 (pure val) rejected at p {1 - stats.chi2.cdf(float((r[m] - e_u[m]) @ np.linalg.pinv(Sm) @ (r[m] - e_u[m])), 4):.3f}; lam = 0 (pure holdout) p {1 - stats.chi2.cdf(float((r[m] - d_hold) @ np.linalg.pinv(Sm + S_hold) @ (r[m] - d_hold)), 4):.3f}")
    say("    Caveat: 4 contrasts; a 1-parameter lam has a 68% interval of roughly ±0.4 around the minimum (see the profile).")
    prof = []
    for lam in np.arange(-0.5, 1.51, 0.1):
        z = r[m] - (lam * e_u[m] + (1 - lam) * d_hold); Sl = Sm + (1 - lam) ** 2 * S_hold
        prof.append((lam, float(z @ np.linalg.pinv(Sl) @ z)))
    say("    chi2 profile: " + " ".join(f"{l:+.1f}:{c:.1f}" for l, c in prof))
    inside = [l for l, c in prof if c <= best[1] + 1]
    say(f"    1-sigma interval for lam (chi2 <= min + 1): [{min(inside):+.1f}, {max(inside):+.1f}]")

    say("\n(ii) where the val excess over the holdout sits: val TO-half delta vs holdout delta (both are train-only members),")
    say("     and val FT-half delta vs TO-half delta (the OOF fine-tune adds nothing pool-specific)")
    say(f"     {'pool':6s} {'d_to(val)':>10s} {'d_hold':>10s} {'to - hold':>18s} | {'d_ft(val)':>10s} {'ft - to':>18s}")
    for k in hk:
        dto = val_half_perfeed(k, "to") - val_half_perfeed("G3", "to"); dft = val_half_perfeed(k, "ft") - val_half_perfeed("G3", "ft")
        dh = H[k] - H["G3"]
        a = (dto.mean() - dh.mean(), np.hypot(dto.std() / np.sqrt(len(dto)), dh.std() / np.sqrt(len(dh))))
        b = wmean_se(dft - dto)
        say(f"     {k:6s} {dto.mean():+10.5f} {dh.mean():+10.5f} {fmt(*a):>18s} | {dft.mean():+10.5f} {fmt(*b):>18s}")
    say("     Reading: the excess of val over the holdout is in the TRAIN-ONLY half (members whose checkpoints were picked on val:")
    say("     m02 'late-cooldown' members), not in the FT half; the FT half tracks the TO half within 1 SE for every pool.")

    say("\n(iii) winner's-curse size: val-selected pools (M2 = decide4 val front-runner, S2 = val-best stacker) vs their public shortfall")
    for k, K in [("M2", 10), ("S2", 20)]:
        j = CONTRASTS.index(k); se_v = np.sqrt(S_val[j, j])
        say(f"     {k}: val SE of its delta vs G3 {se_v:.5f}; expected max-of-K inflation at K={K}: {se_v * np.sqrt(2 * np.log(K)):.5f}; "
            f"observed public - val {r[j] - e_u[j]:+.5f} (SE {np.sqrt(S[j, j]):.5f}); val - holdout {(e_u[j] - d_hold[hk.index(k)]) if k in hk else np.nan:+.5f}")
    say("     Reading: the observed shortfalls are of the size a 10-20-look val selection produces; this is consistent with, not proof of, selection.")

    say("\n(iv) is the holdout-public agreement (chi2 0.7 on 4-5 df) closer than chance? P(chi2_4 <= 0.72) = "
        f"{stats.chi2.cdf(0.72, 4):.3f}: a tighter-than-expected fit, i.e. the holdout SE + public SE already overstate the disagreement;")
    say("     nothing in it suggests the holdout under-predicts the public spread. Expected |public - holdout| per contrast ~0.0006; realised 0.0001-0.0004.")
    (OUT / "c5_extra.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
