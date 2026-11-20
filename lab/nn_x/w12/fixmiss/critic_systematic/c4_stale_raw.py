"""Critic check 4 (CPU): two angles the stream did not read.
(a) Test-like staleness. The stream measured staleness bias on plain validation (warm items 5.69 d stale) while public /
    private are 9.34 / 11.23 d (docs/07 'Measurement of popularity staleness'). Using the stored stale-validation replays
    of the 16 held-out train-only students (e10 kd05w192 + kd05t25w192, @last; same recipe as the FD16 TO branch), compare
    actual/expected for young / cold / old slices on plain vs stale inputs, and the 5-fold user cross-fitted NDCG of the
    stream's age/cold offset on each.
(b) Raw-logit blending of the FD16P20 components (z-pool discards per-feed confidence): same weights, raw means.
"""
import sys
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, "<repo>/lab/nn_x/w12/fixmiss/systematic")
from core import layout, load_table, _feed_lse, zfeed, clogit_fit, softmax_p, HERE  # noqa: E402
sys.path.insert(0, "<repo>/src")
import pmix  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

P = "<repo>/preds/"
SV = "<repo>/lab/nn_x/w12/stale/hz2n/preds_stale_full/"
F, M, off = layout()
fid = np.repeat(np.arange(len(M)), M)
D = load_table(["label", "f4", "xp_imp_c50"])
y = D["label"].to_numpy().astype(np.float64)
W = pmix.weights()


def rep(d):
    return " | ".join(f"{k} {v[0]:+.5f}±{v[1]:.5f}" for k, v in [
        ("val", pmix.wmean_se(d)), ("pub", pmix.wmean_se(d, W["public"])), ("priv", pmix.wmean_se(d, W["private"])),
        ("M<50", pmix.wmean_se(d[M < 50])), ("M>=50", pmix.wmean_se(d[M >= 50]))])


def nll(x, fm):
    lse = _feed_lse(x, off, M)
    return float((-(x - lse[fid]) * y)[fm[fid]].sum())


def folds(seed, K=5):
    u = F["user_id"].to_numpy().astype(np.uint64)
    key = np.where(u == 0, np.arange(len(u), dtype=np.uint64) + np.uint64(10 ** 12), u) + np.uint64(seed * 7919)
    with np.errstate(over="ignore"):
        z = key + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z % np.uint64(K)).astype(np.int64)


E10 = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in range(951, 959)]
plain = zfeed(np.mean([zfeed(np.load(P + f"nn_{m}_trainonly_val.npy").astype(np.float64), off, M) for m in E10], 0), off, M)
stale = zfeed(np.mean([zfeed(np.load(SV + f"nn_{m}@last_staleval.npy").astype(np.float64), off, M) for m in E10], 0), off, M)
f4 = D["f4"].to_numpy().astype(np.float64); cold = D["xp_imp_c50"].to_numpy() == 0
ok = ~np.isnan(f4)
f4z = np.zeros_like(f4); f4z[ok] = (f4[ok] - f4[ok].mean()) / f4[ok].std()
q = np.nanquantile(f4, [0.1, 0.5, 0.9])
slices = {"youngest decile": f4 <= q[0], "oldest decile": f4 >= q[2], "cold (exposure 0)": cold, "warm": ~cold,
          "young & warm": (f4 <= q[0]) & ~cold}
Xst = np.column_stack([f4z, (f4 <= q[0]).astype(float), cold.astype(float)])
nd = {}
for name, s in [("plain val inputs", plain), ("stale val inputs", stale)]:
    T = float(np.exp(minimize_scalar(lambda lt: nll(s / np.exp(lt), np.ones(len(M), bool)), bounds=(-1, 1), method="bounded").x))
    p = softmax_p(s / T, off, M)
    npos = np.bincount(fid, y, len(M))
    E = p * npos[fid]           # expected positives (exact for 1-positive feeds; first-order otherwise)
    nd[name] = ndcg_per_feed(s, y, off, M)
    print(f"\n[{name}] TO 16-pool val NDCG {nd[name].mean():.5f}  T {T:.3f}")
    for sn, m in slices.items():
        a, e = y[m].sum(), E[m].sum()
        print(f"   {sn:20s} rows {int(m.sum()):8d}  actual {int(a):6d}  A/E {a / e:.3f}  z {(a - e) / np.sqrt(e):+.1f}")
    allD = []
    for sd in (1, 2, 3, 4):
        fo = folds(sd); new = s.copy()
        for k in range(5):
            tr = fo != k; r = (fo == k)[fid]
            b = clogit_fit(s / T, Xst, y, off, M, tr, l2=1e-4)
            new[r] = s[r] / T + Xst[r] @ b
        allD.append(ndcg_per_feed(new, y, off, M) - nd[name])
    print(f"   age/cold offset, 5-fold user cross-fit, mean of 4 hashes: {rep(np.mean(allD, 0))}  (last b {np.round(b, 3).tolist()})")
print(f"\nstale minus plain inputs (same members): {rep(nd['stale val inputs'] - nd['plain val inputs'])}")

# (b) raw-logit blend of the FD16P20 components with the same weights as m04_02
S8, S4 = range(951, 959), range(951, 955)
FD16 = [f"w5_fd10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8]
FDP = [f"abl_fd10_pop1kd05w192_s{s}" for s in S4]
wl = np.where(M < 50, 0.4, 0.2)[fid]


def comps(form):
    out = {}
    for k, names, suf in [("bft", FD16, "n2cpv3s3_val_oof"), ("pft", FDP, "n2cpv3s3_val_oof"),
                          ("bto", FD16, "trainonly_val"), ("pto", FDP, "trainonly_val")]:
        arr = [np.load(P + f"nn_{m}_{suf}.npy").astype(np.float64) for m in names]
        out[k] = zfeed(np.mean([zfeed(a, off, M) for a in arr], 0), off, M) if form == "z" else np.mean(arr, 0)
    return out


Cz, Cr = comps("z"), comps("raw")
ref = 0.5 * zfeed((1 - wl) * Cz["bft"] + wl * Cz["pft"], off, M) + 0.5 * zfeed((1 - wl) * Cz["bto"] + wl * Cz["pto"], off, M)
ndref = ndcg_per_feed(ref, y, off, M)
raw = 0.5 * ((1 - wl) * Cr["bft"] + wl * Cr["pft"]) + 0.5 * ((1 - wl) * Cr["bto"] + wl * Cr["pto"])
print(f"\nraw-logit blend (same weights) vs m04_02 z blend: {rep(ndcg_per_feed(raw, y, off, M) - ndref)}")
for k in Cr:
    sdv = np.sqrt(np.maximum(np.bincount(fid, Cr[k] ** 2, len(M)) / M - (np.bincount(fid, Cr[k], len(M)) / M) ** 2, 0))
    print(f"   raw {k}: mean per-feed sd {sdv.mean():.3f}")
