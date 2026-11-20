"""Step 7: robustness of the transfer-factor estimate. Three estimators of k in  public_d = k * true_val_d:
 (a) flat-prior marginal likelihood (used in s2/s4): P | X ~ N(kX, S_P + k^2 S_X)  -> ignores regression to the mean of noisy val deltas
 (b) Deming / functional MLE: argmin r' C(k)^-1 r
 (c) random-effects structural model: true level mu_i ~ N(m, tau^2), joint Gaussian likelihood of (X, P) over (k, tau)
and a parametric bootstrap of each under H: k = 1 (true spread tau from (c)).  Writes out/s7_eiv_check.log."""
from lib import *
import io
from scipy import stats
buf = io.StringIO()


def say(*a):
    print(*a, flush=True); print(*a, file=buf)


z = np.load(OUT / "nd_val.npz"); names = list(z["names"]); ND = dict(zip(names, z["nd"]))
zh = np.load(OUT / "nd_hold.npz"); HN = dict(zip(list(zh["names"]), zh["nd"]))
fv, Mv, ov = meta("val"); pub, tday, Mt = test_parts(); wpub = mix_weights(Mv, Mt[pub])
GRID = np.linspace(-2, 4, 241); TAUS = np.array([0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0013, 0.0016, 0.002, 0.003])


def setup(files, pred="val"):
    A = np.stack([ND[n] for n in files]); n = len(files); wn = wpub / wpub.sum(); mA = A @ wn; R = (A - mA[:, None]) * np.sqrt(wn)[None, :]
    SP = R @ R.T / NPUB
    if pred == "val":
        X = A.mean(1); SX = np.cov(A) / NVAL
    else:
        B = np.stack([HN[n] for n in files]); X = B.mean(1); SX = np.cov(B) / B.shape[1]
    D = np.eye(n)[1:] - np.eye(n)[0]
    return D @ X, D @ np.array([PUBLIC[n] for n in files]), D @ SX @ D.T, D @ SP @ D.T, D @ D.T


def est(x, p, SX, SP, DD):
    """returns k_flat, k_deming, (k_re, tau_re), profile nll of the random-effects model on GRID."""
    q = np.empty(len(GRID)); ld = np.empty(len(GRID))
    for i, k in enumerate(GRID):
        r = p - k * x; C = SP + k * k * SX; q[i] = r @ np.linalg.solve(C, r); ld[i] = np.linalg.slogdet(C)[1]
    kf = GRID[(0.5 * (q + ld)).argmin()]; kd = GRID[q.argmin()]
    return kf, kd, np.nan, 0.0, 0.5 * (q + ld), q, 0.5 * (q + ld)


def ci(prof, lvl):
    ins = GRID[prof <= prof.min() + lvl]; return ins.min(), ins.max()


rng = np.random.default_rng(5)
BLOCKS = [("ALL modern 14", MODERN, "val"), ("unselected 11", [n for n in MODERN if n not in ("M2", "S2", "TOonly")], "val"),
          ("11-12 era 7 (Cd,Pall,PF7x,G3,M2,S2,TOonly)", ["Cd", "Pall", "PF7x", "G3", "M2", "S2", "TOonly"], "val"),
          ("11-13/14 era 8 (G3,C2,Q1a,Q2a,C1,C3,W8,S16)", ["G3", "C2", "Q1a", "Q2a", "C1", "C3", "W8", "S16"], "val"),
          ("G3,C3,W8,S16", ["G3", "C3", "W8", "S16"], "val"),
          ("ALL modern 13 (no S2), holdout", [n for n in MODERN if n != "S2"], "holdout"),
          ("11-13/14 era 8, holdout", ["G3", "C2", "Q1a", "Q2a", "C1", "C3", "W8", "S16"], "holdout")]
for title, files, pred in BLOCKS:
    x, p, SX, SP, DD = setup(files, pred)
    kf, kd, kr, tr, prof_re, q, prof_f = est(x, p, SX, SP, DD)
    lo, hi = ci(prof_re, 0.5); lo2, hi2 = ci(prof_re, 1.92); dlo, dhi = ci(q, 1.0)
    i1 = np.abs(GRID - 1).argmin(); i0 = np.abs(GRID).argmin()
    say(f"\n{title} [{pred}]  (n={len(files)})")
    d2lo, d2hi = ci(q, 3.84)
    say(f"   flat-prior (predictive) k {kf:+.2f} [1s {lo:+.2f},{hi:+.2f}] | Deming (attenuation-corrected) k {kd:+.2f} [dQ<=1: {dlo:+.2f},{dhi:+.2f}] [dQ<=3.84: {d2lo:+.2f},{d2hi:+.2f}]"
        f" | pivot chi2(k=1) {q[i1]:.1f}/{len(x)} (p {1 - stats.chi2.cdf(q[i1], len(x)):.2f})  chi2(k=0) {q[i0]:.1f} (p {1 - stats.chi2.cdf(q[i0], len(x)):.2f})")
    obs_sd = np.sqrt(np.mean(x ** 2)); noise = np.sqrt(np.mean(np.diag(SX)))
    say(f"   rms observed {pred} delta vs first file {obs_sd:.5f}; rms sampling SE of those deltas {noise:.5f}  (signal/noise of the predictor ~{obs_sd / noise:.1f})")
    # parametric bootstrap under k = 1 and k = 0.5 with the fitted tau (and with a tau grid) : distribution of the three estimators
    n = len(files); D = np.eye(n)[1:] - np.eye(n)[0]
    def sq(S):
        w, V = np.linalg.eigh(S); return V * np.sqrt(np.clip(w, 0, None))
    Lx = sq(SX); Lp = sq(SP)
    for ktrue in (1.0, 0.5):
        for tau in (0.0004, 0.0008):
            ks = []
            for b in range(400):
                mu = D @ rng.normal(0, tau, n); xs = mu + Lx @ rng.standard_normal(len(x)); ps = ktrue * mu + Lp @ rng.standard_normal(len(x))
                qq = np.empty(len(GRID)); ll = np.empty(len(GRID))
                for i, k in enumerate(GRID):
                    r = ps - k * xs; C = SP + k * k * SX; qq[i] = r @ np.linalg.solve(C, r); ll[i] = np.linalg.slogdet(C)[1]
                ks.append((GRID[(0.5 * (qq + ll)).argmin()], GRID[qq.argmin()]))
            ks = np.array(ks)
            say(f"   bootstrap true k {ktrue}, tau {tau:.4f}: flat-prior k^ median {np.median(ks[:, 0]):+.2f} (P(k^ <= observed {kf:+.2f}) = {(ks[:, 0] <= kf).mean():.3f}) | Deming k^ median {np.median(ks[:, 1]):+.2f} (P(<= {kd:+.2f}) = {(ks[:, 1] <= kd).mean():.3f})")
(OUT / "s7_eiv_check.log").write_text(buf.getvalue())
