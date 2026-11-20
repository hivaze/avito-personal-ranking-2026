"""Skeptic check of T1 (stale-TO lens S). CPU only. Reads the proposer's per-feed caches and stale member preds
(read-only); writes only under skeptic_T1/out.
 (1) per-contrast S, val_TO, diff-in-diff (S - val_TO) with paired SE: does the stale shift track popularity exposure?
 (2) LOO comparison of S vs trivial shrink-to-incumbent models (constant, holdout+offset, val_TO+offset) on H6.
 (3) parametric simulation: how often does S beat H by the observed LOO margin / reach Kendall 0.714 when the public
     is generated from the holdout lens (k 0.71) or from the null (all equal), given the public noise covariance.
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import kendalltau
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa

VD = Path("/tmp/scratchpad/dpa/validation/out")
ST = VD / "stale"
OUT = Path(__file__).resolve().parent / "out"
LOG = open(OUT / "k1_check.log", "w")


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


FAM = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1"] + [f"v14_s{i}" for i in range(92, 97)],
    "F7": [f"v17_pre2s_s{i}" for i in range(101, 108)],
    "F9": [f"v22_pre2s_lr1_s{i}" for i in range(401, 407)],
    "F10": [f"v23_gru_pre2s_s{i}" for i in range(501, 507)],
    "F12": [f"v26_d192_pre2s_s{i}" for i in range(701, 708)],
    "F7x": [f"r6_cd_m02_r{i}" for i in range(101, 108)],
    "F7x70": [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)],
    "POPFREE": [f"abl_e10_xdrop_pop_s{i}" for i in range(951, 955)],
    "POP1": [f"abl_e10_pop1_s{i}" for i in range(951, 955)],
}
MEM = {
    "G3": FAM["F10"] + FAM["F7x"] + FAM["F7x70"],
    "Cd": FAM["F5cd"] + FAM["F6"] + FAM["F7"],
    "Pall": FAM["F5cd"] + FAM["F6"] + FAM["F7"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
    "PF7x": FAM["F5cd"] + FAM["F6"] + FAM["F7x"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
    "M2": FAM["F7x"] + FAM["F7x70"],
    "Q1a": FAM["F10"] + FAM["F7x"] + FAM["F7x70"] + FAM["POPFREE"],
    "Q2a": FAM["F10"] + FAM["F7x"] + FAM["F7x70"] + FAM["POP1"],
    "G3+F7x70dup": None,
}
PUBLIC = {"Cd": 0.39208, "Pall": 0.392436, "PF7x": 0.392723, "G3": 0.392820, "M2": 0.392434, "Q1a": 0.392886, "Q2a": 0.393205}
Ch = ["Cd", "Pall", "PF7x", "M2", "Q1a", "Q2a"]
N_PUB = 8562

f = pl.read_parquet("<repo>/data/flat/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load("<repo>/data/flat/val/cand_target.npy").astype(np.float64)
n = len(M); fid = np.repeat(np.arange(n), M)


def zf(x):
    x = np.asarray(x, np.float64)
    mu = np.bincount(fid, x, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, x * x, n) / M - mu ** 2, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]


def zmean(arrs):
    acc = None
    for a in arrs:
        z = zf(a); acc = z if acc is None else acc + z
    return acc / len(arrs)


def nd(s):
    return ndcg_per_feed(np.asarray(s, np.float32), y, off, M)


# ---------- (1) per-pool base / stale per-feed NDCG (TO half) ----------
pools = ["G3"] + Ch
base, stale = {}, {}
for p in pools:
    base[p] = nd(zmean([np.load(ST / f"nn_{m}_base.npy") for m in MEM[p]]))
    stale[p] = nd(zmean([np.load(ST / f"nn_{m}_both.npy") for m in MEM[p]]))
# family-addition decomposition: G3 + one family, base vs stale
extra = {"G3+POPFREE": MEM["Q1a"], "G3+POP1": MEM["Q2a"], "G3+F5cd": MEM["G3"] + FAM["F5cd"], "G3+F6": MEM["G3"] + FAM["F6"],
         "G3+F7": MEM["G3"] + FAM["F7"], "G3+F12": MEM["G3"] + FAM["F12"], "G3+F9": MEM["G3"] + FAM["F9"]}
for k, ms in extra.items():
    if k in ("G3+POPFREE", "G3+POP1"):
        continue
    base[k] = nd(zmean([np.load(ST / f"nn_{m}_base.npy") for m in ms]))
    stale[k] = nd(zmean([np.load(ST / f"nn_{m}_both.npy") for m in ms]))
pv = {p: np.load(VD / f"pfv_{p}.npy") for p in pools}
pvto = {p: np.load(VD / f"pfv_{p}_to.npy") for p in pools}
ph = {p: np.load(VD / f"pfh_{p}.npy") for p in pools}
for p in pools:
    assert abs(base[p].mean() - pvto[p].mean()) < 2e-6, (p, base[p].mean(), pvto[p].mean())


def mse(d):
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


pr("(1) per contrast vs G3 (val TO half): val_TO | S (stale both) | S - val_TO (diff-in-diff) | holdout | public")
for p in list(Ch) + [k for k in extra if k not in ("G3+POPFREE", "G3+POP1")]:
    a = mse(base[p] - base["G3"]); b = mse(stale[p] - stale["G3"]); c = mse((stale[p] - stale["G3"]) - (base[p] - base["G3"]))
    hh = mse(ph[p] - ph["G3"]) if p in ph else (np.nan, np.nan)
    pr(f"  {p:11s} {a[0]:+.5f}±{a[1]:.5f} | {b[0]:+.5f}±{b[1]:.5f} | {c[0]:+.5f}±{c[1]:.5f} (z {c[0] / c[1]:+.1f}) | "
       f"{hh[0]:+.5f} | {PUBLIC.get(p, np.nan) - PUBLIC['G3']:+.5f}")
pr(f"  stale drop of each pool (stale - base): " + ", ".join(f"{p} {(stale[p] - base[p]).mean():+.5f}" for p in base))

# ---------- (2) LOO of trivial models on H6 ----------
Pv = np.array([PUBLIC[p] - PUBLIC["G3"] for p in Ch])
D = np.stack([pv[p] - pv["G3"] for p in Ch], 1)
w = np.load(VD / "w_val_pubmatch.npz")["w_pub"]
mu = (w[:, None] * D).sum(0) / w.sum(); R = D - mu
SP = (w[:, None] * R).T @ R / w.sum() / N_PUB


def vec(get, nn):
    Dd = np.stack([get(p) for p in Ch], 1)
    X = Dd.mean(0); Rr = Dd - X
    return X, Rr.T @ Rr / nn ** 2


prox = {"val": vec(lambda p: pv[p] - pv["G3"], n), "val_TO": vec(lambda p: pvto[p] - pvto["G3"], n),
        "holdout": vec(lambda p: ph[p] - ph["G3"], 30000), "S": vec(lambda p: stale[p] - stale["G3"], n)}
prox["0.5H+0.5S"] = (0.5 * prox["holdout"][0] + 0.5 * prox["S"][0], 0.25 * prox["holdout"][1] + 0.25 * prox["S"][1])
prox["const(-1)"] = (-np.ones(6) * 1e-3, np.zeros((6, 6)))


def gls_k(P, SPm, X, SX, it=80):
    k = 1.0
    for _ in range(it):
        Vi = np.linalg.inv(SPm + k * k * SX + 1e-14 * np.eye(len(P)))
        kn = (X @ Vi @ P) / (X @ Vi @ X)
        if abs(kn - k) < 1e-10:
            break
        k = kn
    return k


def loo_k(P, SPm, X, SX):
    e = []
    for i in range(len(P)):
        m = np.arange(len(P)) != i
        k = gls_k(P[m], SPm[np.ix_(m, m)], X[m], SX[np.ix_(m, m)])
        e.append(P[i] - k * X[i])
    return float(np.sqrt(np.mean(np.square(e))))


def loo_ak(P, X):
    """OLS with intercept: public = a + b X (LOO)."""
    e = []
    for i in range(len(P)):
        m = np.arange(len(P)) != i
        A = np.stack([np.ones(m.sum()), X[m]], 1)
        a, b = np.linalg.lstsq(A, P[m], rcond=None)[0]
        e.append(P[i] - a - b * X[i])
    return float(np.sqrt(np.mean(np.square(e))))


def loo_offset_k1(P, X):
    """public = a + X (slope fixed 1, offset fitted LOO)."""
    e = []
    for i in range(len(P)):
        m = np.arange(len(P)) != i
        a = (P[m] - X[m]).mean(); e.append(P[i] - a - X[i])
    return float(np.sqrt(np.mean(np.square(e))))


pr("\n(2) H6 LOO rmse (public = k*X GLS as in the proposer's s5 | public = a + bX OLS | public = a + X):")
pr(f"  null (predict 0): {np.sqrt((Pv ** 2).mean()):.5f};  constant mean (LOO): "
   f"{np.sqrt(np.mean([(Pv[i] - np.delete(Pv, i).mean()) ** 2 for i in range(6)])):.5f}")
for kname, (X, SX) in prox.items():
    pr(f"  {kname:10s} k*X {loo_k(Pv, SP, X, SX):.5f} | a+bX {loo_ak(Pv, X):.5f} | a+X {loo_offset_k1(Pv, X):.5f}")

# ---------- (3) simulation ----------
rng = np.random.default_rng(7)
XH, SH = prox["holdout"]; XS, SS = prox["S"]; XV, SV = prox["val_TO"]
obs = {"H": loo_k(Pv, SP, XH, SH), "S": loo_k(Pv, SP, XS, SS), "V": loo_k(Pv, SP, XV, SV)}
levels = lambda d: np.concatenate([[0.0], d])  # G3 + 6 contrasts
tau_obs = kendalltau(levels(XS), levels(Pv))[0]
pr(f"\n(3) observed: LOO H {obs['H']:.5f}, S {obs['S']:.5f}, val_TO {obs['V']:.5f}; Kendall(S, public) {tau_obs:+.3f}")
NS = 4000
for scen, mean_fn, cov in [("null: public = noise", lambda: np.zeros(6), SP),
                           ("holdout lens true (k 0.71)", lambda: 0.71 * XH, SP + 0.71 ** 2 * SH),
                           ("val_TO lens true (k 0.35)", lambda: 0.35 * XV, SP + 0.35 ** 2 * SV)]:
    L = np.linalg.cholesky(cov + 1e-14 * np.eye(6))
    cntS_beats = cntS_abs = cntTau = cntTauAny = cntS_v = 0
    for _ in range(NS):
        P = mean_fn() + L @ rng.standard_normal(6)
        lh = loo_k(P, SP, XH, SH); ls = loo_k(P, SP, XS, SS); lv = loo_k(P, SP, XV, SV)
        cntS_beats += (ls - lh) <= (obs["S"] - obs["H"])
        cntS_v += (ls - lv) <= (obs["S"] - obs["V"])
        cntS_abs += ls <= obs["S"]
        t = kendalltau(levels(XS), levels(P))[0]
        cntTau += t >= tau_obs - 1e-9
        ts = [kendalltau(levels(prox[k][0]), levels(P))[0] for k in ("val", "val_TO", "holdout", "S", "0.5H+0.5S")]
        cntTauAny += max(ts) >= tau_obs - 1e-9
    pr(f"  [{scen}] P(LOO_S - LOO_H <= observed {obs['S'] - obs['H']:+.5f}) = {cntS_beats / NS:.3f}; "
       f"P(LOO_S - LOO_valTO <= observed {obs['S'] - obs['V']:+.5f}) = {cntS_v / NS:.3f}; P(LOO_S <= {obs['S']:.5f}) = {cntS_abs / NS:.3f}; "
       f"P(tau_S >= {tau_obs:.3f}) = {cntTau / NS:.3f}; P(max tau over 5 lenses >= it) = {cntTauAny / NS:.3f}")
