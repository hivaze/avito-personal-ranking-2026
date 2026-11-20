"""STUDY A: family weights fitted on the HOLDOUT (train-only), evaluated on VAL (out-of-sample). See plan.txt.
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 python nn_x/stack/hfit/hfit.py POOL [variants]
POOL in P6, P3, P11 (family lists below). Writes logs/<POOL>.json + prints a table."""
import os, sys, json, time
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
os.environ.setdefault("OMP_NUM_THREADS", "12"); os.environ.setdefault("NUMBA_NUM_THREADS", "12")
sys.path.insert(0, "<repo>/src")
import numpy as np, numba as nb
from common import feed_meta
from stack_val import feed_rank_z

ROOT = "<repo>"; H = f"{ROOT}/nn_x/stack/hfit"; C = f"{H}/cache"
FAMS = json.load(open(f"{C}/families.json"))
POOLS = {"P6": ["F5cd", "F6", "F7x", "F9", "F10", "F12"], "P3": ["F10", "F7x", "F7x70"],
         "P9nr": [f for f in ["F5cd", "F6", "F7", "F9", "F10", "F12", "V27", "V28", "V29"] if f in FAMS],
         "P11": [f for f in ["F5cd", "F6", "F7", "F7x", "F7x70", "F9", "F10", "F12", "V27", "V28", "V29"] if f in FAMS]}
pool = sys.argv[1]; FL = POOLS[pool]; F = len(FL)
VARIANTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["W1", "W2", "W3", "W4", "W5"]
NF = np.array([len(FAMS[f]) for f in FL], float)
W_ME = NF / NF.sum(); W_FE = np.ones(F) / F
LAMS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
LOG = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


@nb.njit(parallel=True, cache=True)
def ndcg_pf(s, y, off, M):
    n = len(M); out = np.ones(n)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        v = s[o:o + m]; t = y[o:o + m]
        npos = 0
        for k in range(m):
            npos += t[k] > 0
        if npos == 0:
            continue
        idx = np.argsort(-v)
        dcg = 0.0; i = 0
        while i < m:
            j = i
            while j + 1 < m and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            n1 = 0
            for k in range(i, j + 1):
                n1 += t[idx[k]] > 0
            n0 = (j - i + 1) - n1
            for k in range(n1):
                dcg += 1.0 / np.log2(i + n0 + k + 2.0)
            i = j + 1
        idcg = 0.0
        for k in range(npos):
            idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


def fz(off, M, x):
    return feed_rank_z(off, M, np.ascontiguousarray(x, dtype=np.float32))[:, 1].astype(np.float64)


# ---------------- data
Mh, offh = np.load(f"{ROOT}/preds/holdout/holdout_meta.npy"); Mh = Mh.astype(np.int64); offh = offh.astype(np.int64)
yh = np.load(f"{ROOT}/preds/holdout/holdout_y.npy").astype(np.float64)
uh = np.load(f"{C}/hold_user.npy")
Xh = np.stack([np.load(f"{C}/hold_{f}.npy") for f in FL], 1).astype(np.float32)
fv, Mv, offv = feed_meta("val"); Mv = Mv.astype(np.int64); offv = offv.astype(np.int64)
yv = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
Xv_ft = np.stack([np.load(f"{C}/val_ft_{f}.npy") for f in FL], 1).astype(np.float64)
Xv_to = np.stack([np.load(f"{C}/val_to_{f}.npy") for f in FL], 1).astype(np.float64)
SL = dict(np.load(f"{C}/val_slices.npz"))

# ListNet surrogate in float64 (numba loss + analytic gradient; scipy L-BFGS-B). v1 (torch float32 + softmax
# parametrisation, hfit_v1_torchf32.py) stalled at the initial point on several folds -> superseded.
nh = len(Mh); fidh = np.repeat(np.arange(nh), Mh)
NPOS = np.bincount(fidh, yh, nh)
Xh64 = Xh.astype(np.float64)


@nb.njit(parallel=True, cache=True)
def listnet_lg(v, X, y, off, M, npos, sel):
    n = len(sel); F_ = X.shape[1]
    L = np.zeros(n); G = np.zeros((n, F_))
    for k in nb.prange(n):
        f = sel[k]; o, m = off[f], M[f]
        s = X[o:o + m] @ v
        mx = s.max()
        e = np.exp(s - mx); Z = e.sum()
        lse = mx + np.log(Z)
        l = lse
        for i in range(m):
            yn = y[o + i] / npos[f]
            l -= yn * s[i]
            c = e[i] / Z - yn
            for j in range(F_):
                G[k, j] += c * X[o + i, j]
        L[k] = l
    return L.sum() / n, G.sum(0) / n


def fit(feeds, kind, lam, w0, max_iter=500):
    """ListNet (-sum_i y_i/npos log softmax(s)_i, mean over feeds with positives), s = X v.
    kind 'simplex': v >= 0 (temperature = sum v, simplex weights u = v/sum v); 'free': v unconstrained.
    Penalty lam * ||u - w0||^2 on the sum-normalised weights. Returns u."""
    from scipy.optimize import minimize
    sel = np.asarray(feeds[NPOS[feeds] > 0], np.int64)

    def obj(v):
        l, g = listnet_lg(v, Xh64, yh, offh, Mh, NPOS, sel)
        S = v.sum(); u = v / S; d = u - w0
        l += lam * (d * d).sum()
        g = g + (2 * lam / S) * (d - (d * u).sum())
        return l, g
    v0 = np.asarray(w0, np.float64) * 1.0
    bnds = [(0, None)] * F if kind == "simplex" else None
    r = minimize(obj, v0, jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": max_iter, "ftol": 1e-13, "gtol": 1e-10})
    return r.x / r.x.sum()


def hold_score(w, feeds=None):
    s = Xh.astype(np.float64) @ w
    return s


def hold_ndcg_pf(w):
    return ndcg_pf(hold_score(w), yh, offh, Mh)


def coord_ndcg(feeds, w0, passes=4):
    """direct NDCG coordinate ascent on the simplex, objective = mean NDCG over 'feeds' of the holdout."""
    msk = np.zeros(nh, bool); msk[feeds] = True
    obj = lambda w: hold_ndcg_pf(w)[msk].mean()
    w = w0.copy(); best = obj(w)
    for p in range(passes):
        improved = False
        for f in range(F):
            for fac in [0.0, 0.5, 0.75, 1.33, 2.0]:
                w2 = w.copy(); w2[f] = w2[f] * fac if w2[f] > 0 else (0.05 if fac > 1 else 0.0)
                if w2.sum() <= 0: continue
                w2 /= w2.sum(); o = obj(w2)
                if o > best + 1e-6:
                    best, w, improved = o, w2, True
        if not improved: break
    return w


def group_folds(users, k, seed):
    uu = np.unique(users); r = np.random.default_rng(seed).permutation(len(uu))
    g = dict(zip(uu, r % k)); return np.array([g[u] for u in users])


def fit_variant(var, feeds):
    """returns simplex-normalised weights (W5 may have negative entries; normalised by sum)."""
    if var == "W4":
        return coord_ndcg(feeds, W_ME), None
    kind = "free" if var == "W5" else "simplex"
    w0 = W_FE if var == "W3" else W_ME
    if var == "W2":
        return fit(feeds, kind, 0.0, w0), 0.0
    # lambda by 5-fold user-grouped CV inside 'feeds'
    fo = group_folds(uh[feeds], 5, 7)
    cv = {}
    for lam in LAMS:
        pf = np.zeros(len(feeds))
        for k in range(5):
            tr, te = feeds[fo != k], feeds[fo == k]
            w = fit(tr, kind, lam, w0)
            pf[fo == k] = hold_ndcg_pf(w)[te]
        cv[lam] = pf.mean()
    lam = max(cv, key=cv.get)
    return fit(feeds, kind, lam, w0), {"cv": cv, "lam": lam}


def val_score(w):
    return 0.5 * fz(offv, Mv, Xv_ft @ w) + 0.5 * fz(offv, Mv, Xv_to @ w)


def paired(a, b, msk=None):
    d = a - b if msk is None else (a - b)[msk]
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


t0 = time.time()
ME_v = ndcg_pf(val_score(W_ME), yv, offv, Mv); FE_v = ndcg_pf(val_score(W_FE), yv, offv, Mv)
ME_h = hold_ndcg_pf(W_ME); FE_h = hold_ndcg_pf(W_FE)
P(f"POOL {pool} families {FL} n={NF.astype(int).tolist()} (members {int(NF.sum())})")
P(f"  ME  val {ME_v.mean():.5f}  hold {ME_h.mean():.5f} | FE val {FE_v.mean():.5f} hold {FE_h.mean():.5f} | "
  f"FE-ME val {paired(FE_v, ME_v)[0]:+.5f}+-{paired(FE_v, ME_v)[1]:.5f} hold {paired(FE_h, ME_h)[0]:+.5f}+-{paired(FE_h, ME_h)[1]:.5f}")
res = {"pool": pool, "families": FL, "n": NF.tolist(), "ME_val": ME_v.mean(), "ME_hold": ME_h.mean(),
       "FE_val": FE_v.mean(), "FE_hold": FE_h.mean(), "variants": {}}
fo2 = group_folds(uh, 2, 11)
allf = np.arange(nh)
for var in VARIANTS:
    w, info = fit_variant(var, allf)
    # 2-fold user-grouped holdout OOS
    pf_oos = np.zeros(nh); w2s = []
    for k in range(2):
        wk, _ = fit_variant(var, allf[fo2 != k]); w2s.append(wk)
        pf_oos[fo2 == k] = hold_ndcg_pf(wk)[fo2 == k]
    hin = hold_ndcg_pf(w)
    v = ndcg_pf(val_score(w), yv, offv, Mv)
    dME, sME = paired(v, ME_v); dFE, sFE = paired(v, FE_v)
    dho, sho = paired(pf_oos, ME_h); dhi, shi = paired(hin, ME_h)
    slices = {k: paired(v, ME_v, m) for k, m in SL.items()}
    acc = (dME > 2 * sME) and (dho >= -sho) and all(d >= -2 * s for d, s in slices.values())
    P(f"  {var}: w={np.round(w, 3).tolist()} lam={None if info is None else (info if isinstance(info, float) else info['lam'])}")
    if isinstance(info, dict):
        P(f"      CV ndcg by lam: " + " ".join(f"{k:g}:{v_:.5f}" for k, v_ in info["cv"].items()))
    P(f"      2-fold weights: {[np.round(x, 3).tolist() for x in w2s]}")
    P(f"      val {v.mean():.5f}  vs ME {dME:+.5f}+-{sME:.5f} ({dME / sME:+.1f} SE)  vs FE {dFE:+.5f}+-{sFE:.5f}")
    P(f"      hold in-sample vs ME {dhi:+.5f}+-{shi:.5f} | hold 2-fold OOS vs ME {dho:+.5f}+-{sho:.5f} ({dho / sho:+.1f} SE)")
    P("      slices vs ME: " + "  ".join(f"{k} {d:+.5f}+-{s:.5f}" for k, (d, s) in slices.items()))
    P(f"      ACCEPT={acc}   [{time.time() - t0:.0f}s]")
    res["variants"][var] = {"w": w.tolist(), "info": info if not isinstance(info, dict) else {"lam": info["lam"], "cv": {str(k): x for k, x in info["cv"].items()}},
                            "w_2fold": [x.tolist() for x in w2s], "val": v.mean(), "dME": dME, "sME": sME, "dFE": dFE, "sFE": sFE,
                            "hold_in": dhi, "hold_in_se": shi, "hold_oos": dho, "hold_oos_se": sho,
                            "slices": {k: list(x) for k, x in slices.items()}, "accept": bool(acc)}
os.makedirs(f"{H}/logs", exist_ok=True)
json.dump(res, open(f"{H}/logs/{pool}_{'-'.join(VARIANTS)}.json", "w"), indent=1, default=float)
open(f"{H}/logs/{pool}_{'-'.join(VARIANTS)}.txt", "w").write("\n".join(LOG) + "\n")
