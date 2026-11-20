"""Skeptic re-run of the proposer's T3a nested residual (G3 base, user-grouped 5-fold LIN, holdout fit on full val),
restricted to the P2 spec columns (it_hist_all_luo, seq_cv_invd20) vs the 4-column version the proposer reported,
plus controls (trainer tied rank, the 2 non-spec columns). Reads the proposer's extracted arrays READ-ONLY.
Writes only to this directory. CPU only."""
import os, sys, json, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, numba as nb, torch
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
import pmix
HERE = Path(__file__).resolve().parent
P = Path("/tmp/scratchpad/dpa/pipeline/out")
ROOT = Path("<repo>")
LOG = open(HERE / "k01.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
CK = torch.load(ROOT / "nn_x/r6/ckpt/r6_cd_m02_r101.pt", map_location="cpu", weights_only=False)
XT = CK["cfg"]["xt_names"]; xi = {n: j for j, n in enumerate(XT)}
log("xt_rel", [XT[j] for j in CK["cfg"]["xt_rel"]], "xt_idcols", [XT[j] for j in CK["cfg"].get("xt_idcols", [])])

@nb.njit(parallel=True, cache=True)
def feed_z(x, off, M):
    out = np.zeros(len(x))
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]; s = 0.0
        for k in range(m): s += x[o + k]
        mu = s / m; v = 0.0
        for k in range(m): v += (x[o + k] - mu) ** 2
        sd = np.sqrt(v / m)
        for k in range(m): out[o + k] = (x[o + k] - mu) / (sd + 1e-9) if sd > 1e-12 else 0.0
    return out

@nb.njit(parallel=True, cache=True)
def warm_rz(x, off, M):
    """warm-only midrank pct and warm-only z (trainer formula: (x-mean)/(std+1e-4)); cold rows (x<=0) -> 0, 0"""
    r = np.zeros(len(x)); z = np.zeros(len(x))
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]; nw = 0; s = 0.0
        for k in range(m):
            if x[o + k] > 0: nw += 1; s += x[o + k]
        if nw == 0: continue
        mu = s / nw; v = 0.0
        for k in range(m):
            if x[o + k] > 0: v += (x[o + k] - mu) ** 2
        sd = np.sqrt(v / nw)
        for k in range(m):
            u = x[o + k]
            if u <= 0: continue
            less = 0; eq = 0
            for j in range(m):
                w = x[o + j]
                if w <= 0: continue
                if w < u: less += 1
                elif w == u: eq += 1
            r[o + k] = (less + 0.5 * (eq - 1)) / max(nw - 1, 1)
            z[o + k] = (u - mu) / (sd + 1e-4)
    return r, z

@nb.njit(parallel=True, cache=True)
def tied_rz(x, off, M):
    r = np.zeros(len(x)); z = np.zeros(len(x))
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]; s = 0.0
        for k in range(m): s += x[o + k]
        mu = s / m; v = 0.0
        for k in range(m): v += (x[o + k] - mu) ** 2
        sd = np.sqrt(v / m)
        for k in range(m):
            less = 0; eq = 0
            for j in range(m):
                if x[o + j] < x[o + k]: less += 1
                elif x[o + j] == x[o + k]: eq += 1
            r[o + k] = (less + 0.5 * (eq - 1)) / max(m - 1, 1)
            z[o + k] = (x[o + k] - mu) / (sd + 1e-4)
    return r, z

def ld(sub):
    R = np.load(P / f"{sub}_raw.npy", mmap_mode="r"); mt = dict(np.load(P / f"{sub}_meta.npz")); return R, mt
Rv, mv = ld("val"); Rh, mh = ld("hold")
Mv, offv, yv = mv["M"], mv["off"], mv["y"].astype(np.float64)
Mh, offh, yh = mh["M"], mh["off"], mh["y"].astype(np.float64)
hm = np.load(ROOT / "preds/holdout/holdout_meta.npy"); assert (hm[0] == Mh).all() and (hm[1] == offh).all()
assert np.allclose(np.load(ROOT / "preds/holdout/holdout_y.npy"), yh)
nF = len(Mv); fid = np.repeat(np.arange(nF), Mv); npos_f = np.bincount(fid, weights=yv, minlength=nF)
zz = lambda a, off, M: feed_z(np.asarray(a, np.float64), off, M)
b = zz(0.5 * zz(np.load(ROOT / "preds/ens_G3_ft_val.npy"), offv, Mv) + 0.5 * zz(np.load(ROOT / "preds/ens_G3_to_val.npy"), offv, Mv), offv, Mv)
MEM = [f"v23_gru_pre2s_s50{i}" for i in range(1, 7)] + [f"r6_cd_m02_r10{i}" for i in range(1, 8)] + [f"r6_cd_m02_p70k_r10{i}" for i in range(1, 8)]
bh = zz(np.mean([zz(np.load(ROOT / f"preds/holdout/nn_{m}_holdout.npy"), offh, Mh) for m in MEM], 0), offh, Mh)
dayh = (mh["ts"].astype(np.int64) + 10800) // 86400; dayh = dayh - dayh.min(); late = dayh >= 11
log(f"late holdout feeds (Sep 17-19 Moscow, day idx >= 11): {late.sum()}")
day = (mv["ts"].astype(np.int64) + 10800) // 86400
luo_v = np.nan_to_num(np.asarray(Rv[:, xi["it_hist_all_luo"]], np.float64)); cold = luo_v == 0
coldpos_f = np.bincount(fid, weights=yv * cold, minlength=nF)
SL = {"M<50": Mv < 50, "M>=50": Mv >= 50, "tue": day <= day.min() + 1, "thu": day >= day.min() + 2,
      "coldpos": (coldpos_f == npos_f) & (npos_f > 0), "warmpos": (coldpos_f == 0) & (npos_f > 0)}
WPRIV = pmix.weights()["private"]
folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=mv["user"]))
rowfold = np.empty(len(yv), np.int64)
for k, (trf, vaf) in enumerate(folds): rowfold[np.isin(fid, vaf)] = k
ndv = lambda s: ndcg_per_feed(s, yv, offv, Mv); ndh = lambda s: ndcg_per_feed(s, yh, offh, Mh)

def nll(theta, G, rows):
    s = G @ theta; ff = fid[rows]
    e = np.exp(np.clip(s, -50, 20.0)); Z = np.bincount(ff, weights=e, minlength=nF)
    ok = Z > 0; lZ = np.log(np.where(ok, Z, 1.0)); yy = yv[rows]
    npu = np.bincount(ff, weights=yy, minlength=nF)
    L = -(yy * s).sum() + (npu * lZ).sum(); p = e / np.where(ok, Z, 1.0)[ff]; w = npu[ff] * p - yy
    return L, G.T @ w
def fit_lin(G, tr):
    th0 = np.zeros(G.shape[1]); th0[0] = 0.94; rows = np.flatnonzero(tr)
    return minimize(lambda t: nll(t, G[tr], rows), th0, jac=True, method="L-BFGS-B", options=dict(maxiter=300)).x
G0 = b[:, None]; pred = np.empty(len(yv))
for k in range(5):
    tr = rowfold != k; pred[~tr] = G0[~tr] @ fit_lin(G0, tr)
beta_full = fit_lin(G0, np.ones(len(yv), bool))[0]
C_v = ndv(pred); C_h = ndh(beta_full * bh)
log(f"base LIN control val {C_v.mean():.5f} hold {C_h.mean():.5f}")

def run(name, Gv, Gh):
    G = np.column_stack([b, Gv]); pred = np.empty(len(yv))
    for k in range(5):
        tr = rowfold != k; pred[~tr] = G[~tr] @ fit_lin(G, tr)
    th = fit_lin(G, np.ones(len(yv), bool)); ph = np.column_stack([bh, Gh]) @ th
    dv = ndv(pred) - C_v; dh = ndh(ph) - C_h
    se = dv.std() / np.sqrt(len(dv)); seh = dh.std() / np.sqrt(len(dh)); sel = dh[late].std() / np.sqrt(late.sum())
    pm, pse = pmix.wmean_se(dv, WPRIV)
    line = (f"{name:28s} val {dv.mean():+.5f} ({se:.5f}) z {dv.mean()/se:+.2f} | pmix {pm:+.5f} ({pse:.5f}) z {pm/pse:+.2f} | "
            f"hold {dh.mean():+.5f} ({seh:.5f}) z {dh.mean()/seh:+.2f} | late {dh[late].mean():+.5f} ({sel:.5f}) z {dh[late].mean()/sel:+.2f}")
    for sn, m in SL.items(): line += f" | {sn} {dv[m].mean():+.5f}"
    log(line + f" | w {np.round(th, 3)}")
    with open(HERE / "k01_results.jsonl", "a") as fh:
        fh.write(json.dumps(dict(name=name, val=float(dv.mean()), val_se=float(se), pmix=float(pm), pmix_se=float(pse),
                                 hold=float(dh.mean()), hold_se=float(seh), late=float(dh[late].mean()), late_se=float(sel))) + "\n")

def raw(R, n): return np.nan_to_num(np.asarray(R[:, xi[n]], np.float64), nan=0.0)
def feats(kind, cols):
    out = []
    for R, off, M in ((Rv, offv, Mv), (Rh, offh, Mh)):
        F = []
        for n in cols:
            x = raw(R, n); cf = (x <= 0).astype(np.float64)
            if kind == "warm_r_flag":            # proposer's T3a encoding
                r, _ = warm_rz(x, off, M); F += [r, cf]
            elif kind == "warm_rz_flag":         # P2 spec: warm-only rank AND z, 0 on cold, + cold flag
                r, z = warm_rz(x, off, M); F += [r, z, cf]
            elif kind == "tied_rz_flag":         # control: trainer tied midrank and z + cold flag
                r, z = tied_rz(x, off, M); F += [r, z, cf]
            elif kind == "tied_r_flag":
                r, _ = tied_rz(x, off, M); F += [r, cf]
        out.append(np.column_stack(F))
    return out
POP4 = ["it_hist_all_luo", "seq_cv_invd20", "it_hist_et1_luo", "seq_cv_pop"]
SPEC = ["it_hist_all_luo", "seq_cv_invd20"]; OTHER = ["it_hist_et1_luo", "seq_cv_pop"]
T0 = time.time()
for name, kind, cols in [("A_T3a_repro_4col", "warm_r_flag", POP4), ("B_spec2_warm_r_flag", "warm_r_flag", SPEC),
                         ("C_spec2_warm_rz_flag", "warm_rz_flag", SPEC), ("D_other2_warm_r_flag", "warm_r_flag", OTHER),
                         ("E_spec2_TIED_r_flag", "tied_r_flag", SPEC), ("F_spec2_TIED_rz_flag", "tied_rz_flag", SPEC),
                         ("G_4col_TIED_r_flag", "tied_r_flag", POP4)]:
    Gv, Gh = feats(kind, cols); run(name, Gv, Gh)
log(f"DONE {time.time()-T0:.0f}s")
