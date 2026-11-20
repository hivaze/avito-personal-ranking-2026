"""Critic check 2 (CPU): the stream reads every correction on ONE 2-way user split (fit on 50%, apply to the other 50%).
Re-read the three items closest to a positive with 5-fold user cross-fitting (fit on 80%) over 4 different user
hashes, so a split-specific sign cannot carry the conclusion:
  (i)  hmean5 top-decile offset on the final: likelihood fit (listwise CE) and direct NDCG grid
  (ii) pop-light weight per tercile of history length H (the stream's only positive: +0.00012 +- 0.00027)
  (iii) staleness set (z f4, freshest decile, cold) likelihood offset (the only item with pmix-private > 0)
Paired per-feed deltas vs the stored best file (m04_02); val, pmix-public, pmix-private, M<50, M>=50.
"""
import sys
import numpy as np
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, "<repo>/lab/nn_x/w12/fixmiss/systematic")
from core import layout, load_table, _feed_lse, zfeed, clogit_fit, HERE  # noqa: E402
sys.path.insert(0, "<repo>/src")
import pmix  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

P = "<repo>/preds/"
F, M, off = layout()
fid = np.repeat(np.arange(len(M)), M)
D = load_table(["label", "hmean5", "f4", "xp_imp_c50", "H"])
y = D["label"].to_numpy().astype(np.float64)
W = pmix.weights()
ft = np.load(P + "ens_FD16P20_m04_02kd_ft_val.npy").astype(np.float64)
to = np.load(P + "ens_FD16P20_m04_02kd_to_val.npy").astype(np.float64)
s = 0.5 * ft + 0.5 * to
nd0 = ndcg_per_feed(s, y, off, M)
print(f"base val {nd0.mean():.5f}")


def folds(seed, K=5):
    u = F["user_id"].to_numpy().astype(np.uint64)
    key = np.where(u == 0, np.arange(len(u), dtype=np.uint64) + np.uint64(10 ** 12), u) + np.uint64(seed * 7919)
    with np.errstate(over="ignore"):
        z = key + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z % np.uint64(K)).astype(np.int64)


def rep(d):
    out = []
    for k, v in [("val", pmix.wmean_se(d)), ("pub", pmix.wmean_se(d, W["public"])), ("priv", pmix.wmean_se(d, W["private"])),
                 ("M<50", pmix.wmean_se(d[M < 50])), ("M>=50", pmix.wmean_se(d[M >= 50]))]:
        out.append(f"{k} {v[0]:+.5f}±{v[1]:.5f}")
    return " | ".join(out)


def nll(x, fm):
    lse = _feed_lse(x, off, M)
    return float((-(x - lse[fid]) * y)[fm[fid]].sum())


SEEDS = [1, 2, 3, 4]
FS = {sd: folds(sd) for sd in SEEDS}

# ---------------- (i) hmean5 top decile ----------------
h5 = D["hmean5"].to_numpy()
x_h = (h5 >= np.nanquantile(h5, 0.9)).astype(float)
grid = np.array([-0.2, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3])
curves = np.array([ndcg_per_feed(s + b * x_h, y, off, M) - nd0 for b in grid])


def run_offset(X, label):
    allD = []
    for sd in SEEDS:
        fo = FS[sd]; d = np.zeros(len(M)); newscore = s.copy()
        for k in range(5):
            tr = fo != k; te = fo == k
            T = float(np.exp(minimize_scalar(lambda lt: nll(s / np.exp(lt), tr), bounds=(-1, 1), method="bounded").x))
            b = clogit_fit(s / T, X, y, off, M, tr, l2=1e-4)
            r = te[fid]
            newscore[r] = s[r] / T + X[r] @ b
        d = ndcg_per_feed(newscore, y, off, M) - nd0
        allD.append(d)
        print(f"  {label} seed {sd}: {rep(d)}")
    print(f"  {label} MEAN over {len(SEEDS)} hashes: {rep(np.mean(allD, 0))}", flush=True)


def run_grid(curves, grid, label):
    allD = []
    for sd in SEEDS:
        fo = FS[sd]; d = np.zeros(len(M)); picks = []
        for k in range(5):
            tr = fo != k; te = fo == k
            j = curves[:, tr].mean(1).argmax(); picks.append(grid[j])
            d[te] = curves[j, te]
        allD.append(d)
        print(f"  {label} seed {sd}: {rep(d)}  picks {picks}")
    print(f"  {label} MEAN over {len(SEEDS)} hashes: {rep(np.mean(allD, 0))}", flush=True)
    j = curves.mean(1).argmax()
    print(f"  {label} in-sample best on ALL val: b {grid[j]}  {curves[j].mean():+.5f}")


print("\n(i) hmean5 top-decile offset on the final")
run_offset(x_h[:, None], "hmean5_q9 likelihood")
run_grid(curves, grid, "hmean5_q9 NDCG grid")

# ---------------- (iii) staleness set ----------------
f4 = D["f4"].to_numpy().astype(np.float64)
ok = ~np.isnan(f4)
f4z = np.zeros_like(f4); f4z[ok] = (f4[ok] - f4[ok].mean()) / f4[ok].std()
Xst = np.column_stack([f4z, (f4 <= np.nanquantile(f4, 0.1)).astype(float), (D["xp_imp_c50"].to_numpy() == 0).astype(float)])
print("\n(iii) staleness set offset on the final")
run_offset(Xst, "age/cold likelihood")

# ---------------- (ii) pop-light weight per tercile of H ----------------
S8 = range(951, 959); S4 = range(951, 955)
FD16 = [f"w5_fd10_{a}_s{sd}" for a in ("kd05w192", "kd05t25w192") for sd in S8]
FDP = [f"abl_fd10_pop1kd05w192_s{sd}" for sd in S4]


def pool(names, suf):
    return zfeed(np.mean([zfeed(np.load(P + f"nn_{m}_{suf}.npy").astype(np.float64), off, M) for m in names], 0), off, M)


Z = {"bft": pool(FD16, "n2cpv3s3_val_oof"), "pft": pool(FDP, "n2cpv3s3_val_oof"),
     "bto": pool(FD16, "trainonly_val"), "pto": pool(FDP, "trainonly_val")}


def build(wrow):
    return 0.5 * zfeed((1 - wrow) * Z["bft"] + wrow * Z["pft"], off, M) + 0.5 * zfeed((1 - wrow) * Z["bto"] + wrow * Z["pto"], off, M)


wref = np.where(M < 50, 0.4, 0.2)
ref = build(wref[fid]); ndr = ndcg_per_feed(ref, y, off, M)
print(f"\n(ii) rebuilt m04_02 val {ndr.mean():.5f} (stored {nd0.mean():.5f}, max|d| per feed {np.abs(ndr - nd0).max():.2e})")
H = D["H"].to_numpy()[off].astype(float)
qs = np.quantile(H, [1 / 3, 2 / 3]); t = np.searchsorted(qs, H, side="right")
print(f"   H tercile cut points {qs}; tercile sizes {np.bincount(t)}")
wg = [0.1, 0.2, 0.3, 0.4, 0.5]
cur = {}
for tt in range(3):
    for w in wg:
        wf = np.where(t == tt, w, wref)
        cur[(tt, w)] = ndcg_per_feed(build(wf[fid]), y, off, M) - ndr
allD = []
for sd in SEEDS:
    fo = FS[sd]; d = np.zeros(len(M)); picks = {}
    for k in range(5):
        tr = fo != k
        for tt in range(3):
            wb = max(wg, key=lambda w: cur[(tt, w)][tr & (t == tt)].mean())
            m = (fo == k) & (t == tt)
            d[m] = cur[(tt, wb)][m]
            picks.setdefault(tt, []).append(wb)
    allD.append(d)
    print(f"  H-tercile grid seed {sd}: {rep(d)}  picks {picks}")
print(f"  H-tercile grid MEAN over {len(SEEDS)} hashes: {rep(np.mean(allD, 0))}")
for tt in range(3):
    print(f"   in-sample tercile {tt}: " + "  ".join(f"w{w} {cur[(tt, w)][t == tt].mean():+.5f}" for w in wg))
# a pre-declared fixed rule implied by the stream's picks (no fitting): 0.2 at the H cap tercile, 0.4 elsewhere (M<50 keeps 0.4)
wfix = np.where(t == 2, np.where(M < 50, 0.4, 0.2), 0.4)
dfix = ndcg_per_feed(build(wfix[fid]), y, off, M) - ndr
print(f"  fixed rule 'pop-light 0.4 unless H top tercile and M>=50' (in-sample, selected after seeing the stream): {rep(dfix)}")
