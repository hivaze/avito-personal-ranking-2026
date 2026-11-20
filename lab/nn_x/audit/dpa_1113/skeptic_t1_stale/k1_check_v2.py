"""Skeptic check of T1 (stale-TO lens). CPU only. Read-only on the proposer's dpa/validation/out files and project data.
Writes only to this directory.
 (1) per-pool stale drop (stale TO - base TO) with paired SE; diff-in-diff per contrast vs G3 with paired SE
 (2) decomposition S_c = V_TO_c + (drop_c - drop_G3): how much of S is one shared number (G3's drop)
 (3) H0 Monte Carlo: public contrasts = pure noise N(0, SP) -> distribution of the BEST proxy's fit (multiplicity)
 (4) H1 Monte Carlo: public = 0.71 * holdout truth + noise -> P(S beats H by the observed LOO margin / Kendall)
"""
import os, sys
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
VAL = "/tmp/scratchpad/dpa/validation"
sys.path.insert(0, VAL)
from lib import meta, val_y, zmean, zf, ndcg_per_feed, MEMBERS, PUBLIC, N_PUB, FAM  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from pathlib import Path  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

VOUT = Path(VAL) / "out"; ST = VOUT / "stale"
HERE = Path(__file__).resolve().parent
LOG = open(HERE / "k1_check_v2.log", "w")


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


f, M, off = meta("val"); y = val_y()
Ch = ["Cd", "Pall", "PF7x", "M2", "Q1a", "Q2a"]; pools = ["G3"] + Ch


def pool_nd(members, scen):
    return ndcg_per_feed(zmean([np.load(ST / f"nn_{m}_{scen}.npy") for m in members], M).astype(np.float32), y, off, M)


B = {p: pool_nd(MEMBERS[p], "base") for p in pools}
S = {p: pool_nd(MEMBERS[p], "both") for p in pools}
pr("(1) pool-level stale drop (TO half, stale both - base), paired SE over val feeds")
for p in pools:
    d = S[p] - B[p]
    pr(f"  {p:5s} base {B[p].mean():.5f} stale {S[p].mean():.5f} drop {d.mean():+.5f} ± {d.std(ddof=1) / np.sqrt(len(d)):.5f}")
pr("  diff-in-diff per contrast  DiD_c = (S_c - S_G3) - (B_c - B_G3) = drop_c - drop_G3:")
for c in Ch:
    d = (S[c] - S["G3"]) - (B[c] - B["G3"])
    v = B[c] - B["G3"]; s = S[c] - S["G3"]
    pr(f"  {c:5s} valTO {v.mean():+.5f} ± {v.std(ddof=1)/np.sqrt(len(v)):.5f} | S {s.mean():+.5f} ± {s.std(ddof=1)/np.sqrt(len(s)):.5f} | DiD {d.mean():+.5f} ± {d.std(ddof=1) / np.sqrt(len(d)):.5f} (z {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.1f})")
# families inside G3 and the added families
for fam in ("F10", "F7x", "F7x70", "POP1", "POPFREE"):
    b = pool_nd(FAM[fam], "base"); s = pool_nd(FAM[fam], "both")
    pr(f"  family pool {fam:7s} drop {(s - b).mean():+.5f}")
# within-feed corr between pool scores under base and stale for G3 vs Q2a member sets (diversity under stale)
pr("  pop1 s951 both vs base file corr:", f"{np.corrcoef(np.load(ST / 'nn_abl_e10_pop1_s951_both.npy'), np.load(ST / 'nn_abl_e10_pop1_s951_base.npy'))[0, 1]:.7f}",
   "max|d|", f"{np.abs(np.load(ST / 'nn_abl_e10_pop1_s951_both.npy') - np.load(ST / 'nn_abl_e10_pop1_s951_base.npy')).max():.3e}")
pr("  pop1 s952 both vs base file corr:", f"{np.corrcoef(np.load(ST / 'nn_abl_e10_pop1_s952_both.npy'), np.load(ST / 'nn_abl_e10_pop1_s952_base.npy'))[0, 1]:.7f}")

# ---------------- proxies (6-vectors + covariances), as in s5/s8
pv = {p: np.load(VOUT / f"pfv_{p}.npy") for p in pools}
pvto = {p: np.load(VOUT / f"pfv_{p}_to.npy") for p in pools}
ph = {p: np.load(VOUT / f"pfh_{p}.npy") for p in pools}
Pv = np.array([PUBLIC[p] - PUBLIC["G3"] for p in Ch])
D = np.stack([pv[p] - pv["G3"] for p in Ch], 1); w = np.load(VOUT / "w_val_pubmatch.npz")["w_pub"]
mu = (w[:, None] * D).sum(0) / w.sum(); R = D - mu; SP = (w[:, None] * R).T @ R / w.sum() / N_PUB


def vec(get, n):
    Dd = np.stack([get(p) for p in Ch], 1); X = Dd.mean(0); Rr = Dd - X
    return X, Rr.T @ Rr / n ** 2


dh = pl.read_parquet(VOUT / "desc_holdout.parquet")
dd = (f["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400; tue = dd <= 20719
FTval = {p: None for p in pools}
ENS = {"Cd": "Cd", "Pall": "Pall", "PF7x": "PF7x", "G3": "G3", "M2": "M2", "Q1a": "Q1apopfree", "Q2a": "Q2apop1"}
SF = {}
for p in pools:
    ft = zf(np.load(f"<repo>/preds/ens_{ENS[p]}_ft_val.npy"), M)
    to = zmean([np.load(ST / f"nn_{m}_both.npy") for m in MEMBERS[p]], M)
    SF[p] = ndcg_per_feed((0.5 * ft + 0.5 * zf(to, M)).astype(np.float32), y, off, M)
prox = {
    "val": vec(lambda p: pv[p] - pv["G3"], len(M)),
    "val_TO": vec(lambda p: pvto[p] - pvto["G3"], len(M)),
    "holdout": vec(lambda p: ph[p] - ph["G3"], 30000),
    "stale_TO": vec(lambda p: S[p] - S["G3"], len(M)),
    "stale_final": vec(lambda p: SF[p] - SF["G3"], len(M)),
    "val_Thu": vec(lambda p: (pv[p] - pv["G3"])[~tue], int((~tue).sum())),
    "val_Tue": vec(lambda p: (pv[p] - pv["G3"])[tue], int(tue.sum())),
}
for lab, msk in [("hold_weekend", dh["dow"].to_numpy() >= 5), ("hold_late3d", dh["day"].to_numpy() >= 20713),
                 ("hold_luo0>=.3", dh["luo_zero"].to_numpy() >= 0.3)]:
    prox[lab] = vec(lambda p, msk=msk: (ph[p] - ph["G3"])[msk], int(msk.sum()))
Xh, Sh = prox["holdout"]; Xs, Ss = prox["stale_TO"]
for lam in (0.25, 0.5, 0.75):
    prox[f"{lam}H+{1-lam}S"] = (lam * Xh + (1 - lam) * Xs, lam ** 2 * Sh + (1 - lam) ** 2 * Ss)
# S with the G3-specific shared offset removed: S' = valTO + mean-free DiD (keeps contrast-specific staleness only)
Xv, Sv = prox["val_TO"]; did = Xs - Xv
prox["valTO+DiD-meanDiD"] = (Xv + did - did.mean(), Sv)
prox["valTO+meanDiD"] = (Xv + did.mean(), Sv)


def gls_k(P, SPm, X, SX, it=40):
    k = 1.0
    for _ in range(it):
        Vi = np.linalg.inv(SPm + k * k * SX); kn = (X @ Vi @ P) / (X @ Vi @ X)
        if abs(kn - k) < 1e-9:
            k = kn; break
        k = kn
    Vi = np.linalg.inv(SPm + k * k * SX); se = 1 / np.sqrt(X @ Vi @ X)
    r = P - k * X; ch = float(r @ np.linalg.inv(SPm + k * k * SX) @ r)
    c0 = float(P @ np.linalg.inv(SPm) @ P)
    return k, se, ch, c0


def loo(P, SPm, X, SX):
    e = []
    for i in range(len(P)):
        m = np.arange(len(P)) != i
        k, se, _, _ = gls_k(P[m], SPm[np.ix_(m, m)], X[m], SX[np.ix_(m, m)])
        e.append(P[i] - k * X[i])
    return float(np.sqrt(np.mean(np.square(e))))


def kt(P, X):
    return kendalltau(np.r_[0.0, P], np.r_[0.0, X])[0]


pr("\n(2) observed fits on the 6 public contrasts (public = k * proxy; GLS)")
pr(f"  chi2(k=0) of public deltas vs pure public noise SP: {float(Pv @ np.linalg.inv(SP) @ Pv):.2f} on 6 df; public SE (diag) {np.sqrt(np.diag(SP)).round(5)}")
obs = {}
for name, (X, SX) in prox.items():
    k, se, ch, c0 = gls_k(Pv, SP, X, SX); lr = loo(Pv, SP, X, SX); t = kt(Pv, X)
    obs[name] = (c0 - ch, lr, t)
    pr(f"  {name:20s} k {k:+.2f} ± {se:.2f}  dchi2 {c0 - ch:5.2f}  LOOrmse {lr:.5f}  Kendall {t:+.3f}  X={np.round(X * 1e4, 1)}")
pr(f"  public x1e4 = {np.round(Pv * 1e4, 1)}")

names = [n for n in prox]
L = np.linalg.cholesky(SP)
rng = np.random.default_rng(20261113)
NS = 2000


def sim(truth_fn, tag):
    best_d, min_lr, max_t, s_minus_h_lr, s_t, s_d = [], [], [], [], [], []
    for _ in range(NS):
        # proxy noise: redraw each proxy around its observed value is not needed for multiplicity; keep proxies fixed
        P = truth_fn() + L @ rng.standard_normal(6)
        ds, lrs, ts = {}, {}, {}
        for n in names:
            X, SX = prox[n]
            k, se, ch, c0 = gls_k(P, SP, X, SX)
            ds[n] = c0 - ch; ts[n] = kt(P, X)
        for n in ("holdout", "stale_TO"):
            lrs[n] = loo(P, SP, *prox[n])
        best_d.append(max(ds.values())); max_t.append(max(ts.values()))
        s_minus_h_lr.append(lrs["stale_TO"] - lrs["holdout"]); s_t.append(ts["stale_TO"]); s_d.append(ds["stale_TO"])
    best_d, max_t, s_minus_h_lr, s_t, s_d = map(np.array, (best_d, max_t, s_minus_h_lr, s_t, s_d))
    od, olr, ot = obs["stale_TO"]; ohlr = obs["holdout"][1]
    pr(f"  [{tag}] P(max over {len(names)} proxies of dchi2 >= observed S {od:.2f}) = {(best_d >= od).mean():.3f}; "
       f"P(S alone dchi2 >= {od:.2f}) = {(s_d >= od).mean():.3f}")
    pr(f"  [{tag}] P(max Kendall over proxies >= {ot:.3f}) = {(max_t >= ot - 1e-9).mean():.3f}; P(S Kendall >= {ot:.3f}) = {(s_t >= ot - 1e-9).mean():.3f}")
    pr(f"  [{tag}] P(LOOrmse(S) - LOOrmse(H) <= observed {olr - ohlr:+.5f}) = {(s_minus_h_lr <= olr - ohlr).mean():.3f}; P(LOOrmse(S) <= LOOrmse(H)) = {(s_minus_h_lr <= 0).mean():.3f}")


pr("\n(3) Monte Carlo, proxies fixed at observed values, public redrawn; N =", NS)
sim(lambda: np.zeros(6), "H0: all true public contrasts = 0")
kH = gls_k(Pv, SP, Xh, Sh)[0]
sim(lambda: kH * Xh, f"H1: truth = {kH:.2f} * holdout")
LOG.close()
