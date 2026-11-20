"""S8: combined lenses (holdout + stale-val TO), stale-val under composition weights, proxy inter-correlations, and the
retroactive decisions each lens would have made. Writes out/s8_combo.log."""
import importlib.util, sys
from lib import *  # noqa
spec = importlib.util.spec_from_file_location("s5", str(HERE / "s5_stale_transfer.py")); s5 = importlib.util.module_from_spec(spec)
sys.argv = ["x"]; spec.loader.exec_module(s5)  # functions only (main not run); NOTE: s5 opens its log in 'w' -> re-run s5 after
LOG = open(OUT / "s8_combo.log", "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n")
f, M, off = meta("val"); y = val_y(); ST = OUT / "stale"
Ch = ["Cd", "Pall", "PF7x", "M2", "Q1a", "Q2a"]; pools = ["G3"] + Ch
st = {p: ndcg_per_feed(zmean([np.load(ST / f"nn_{m}_both.npy") for m in MEMBERS[p]], M).astype(np.float32), y, off, M) for p in pools}
ph = {p: np.load(OUT / f"pfh_{p}.npy") for p in pools}
pv = {p: np.load(OUT / f"pfv_{p}.npy") for p in pools}
import pmix; W = pmix.weights(); wz = np.load(OUT / "w_val_pubmatch.npz")
Pv = np.array([PUBLIC[p] - PUBLIC["G3"] for p in Ch])
D = np.stack([pv[p] - pv["G3"] for p in Ch], 1); w = wz["w_pub"]
mu = (w[:, None] * D).sum(0) / w.sum(); R = D - mu; SP = (w[:, None] * R).T @ R / w.sum() / N_PUB
def wvec(get, ww=None, n=None):
    D = np.stack([get(p) for p in Ch], 1); ww = np.ones(len(D)) if ww is None else ww
    X = (ww[:, None] * D).sum(0) / ww.sum(); R = D - X
    return X, (ww[:, None] ** 2 * R).T @ R / ww.sum() ** 2
Xs, Ss = wvec(lambda p: st[p] - st["G3"]); Xh, Sh = wvec(lambda p: ph[p] - ph["G3"])
prox = {"holdout": (Xh, Sh), "stale_TO": (Xs, Ss),
        "stale_TO@EBpub": wvec(lambda p: st[p] - st["G3"], w), "stale_TO@calibM": None, "stale_TO@pmix_pub": wvec(lambda p: st[p] - st["G3"], W["public"])}
EDGES = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250]); ft, Mt, _ = meta("test"); pub = public_mask()
bv = np.searchsorted(EDGES, M, side="right"); bt = np.searchsorted(EDGES, Mt[pub], side="right")
wM = (np.bincount(bt, minlength=10) / len(bt) / (np.bincount(bv, minlength=10) / len(bv)))[bv]
prox["stale_TO@calibM"] = wvec(lambda p: st[p] - st["G3"], wM)
for lam in (0.25, 0.5, 0.75):
    prox[f"{lam}*hold+{1 - lam}*stale"] = (lam * Xh + (1 - lam) * Xs, lam ** 2 * Sh + (1 - lam) ** 2 * Ss)
pr("== H6 fits (public = k * proxy) ==")
pr(f"  {'proxy':22s} {'k':>6s} {'SE':>5s} {'chi2k^':>7s} {'chi2k1':>7s} {'chi2k0':>7s} {'gR2':>5s} {'LOOrmse':>8s} {'LOOz2':>6s} | rmse(public-proxy) at k=1")
for kname, (X, SX) in prox.items():
    k, se, ch, c1, c0 = s5.gls_k(Pv, SP, X, SX); lr, lz = s5.loo(Pv, SP, X, SX)
    pr(f"  {kname:22s} {k:+6.2f} {se:5.2f} {ch:7.2f} {c1:7.2f} {c0:7.2f} {1 - ch / c0:5.2f} {lr:8.5f} {lz:6.2f} | {np.sqrt(((Pv - X) ** 2).mean()):.5f}")
pr("\nper-contrast (public | holdout | stale_TO | 0.5/0.5):")
for i, p in enumerate(Ch):
    pr(f"  {p:5s} {Pv[i]:+.5f} | {Xh[i]:+.5f} | {Xs[i]:+.5f} | {0.5 * Xh[i] + 0.5 * Xs[i]:+.5f}")
pr(f"\ncorr across the 6 contrasts: hold~stale {np.corrcoef(Xh, Xs)[0, 1]:+.3f}; hold~public {np.corrcoef(Xh, Pv)[0, 1]:+.3f}; stale~public {np.corrcoef(Xs, Pv)[0, 1]:+.3f}; "
   f"val~stale {np.corrcoef(np.array([(pv[p] - pv['G3']).mean() for p in Ch]), Xs)[0, 1]:+.3f}")
pr(f"rmse(public - 0) {np.sqrt((Pv ** 2).mean()):.5f}; mean public SE {np.sqrt(np.diag(SP)).mean():.5f}")
