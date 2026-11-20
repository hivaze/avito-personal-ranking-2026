"""Skeptic check for P2 HYG-B (warm-only within-feed rank/z for it_hist_all_luo, seq_cv_invd20).
Reads the pipeline lens' extracted arrays (read-only) and writes only to this directory.
 1. parity: trainer rk/z in X vs my recomputation from R (so the spec encoding is built on the trainer's scale)
 2. PSI trn->val / trn->priv / val->priv of trainer rk/z vs the SPEC encoding (warm-only rank/z, cold rows 0)
 3. nested LIN residual on G3 (user-grouped 5-fold on val; full-val weights applied to the holdout):
    S_spec   = spec encoding of the 2 columns (4 feats)
    S_specf  = spec + 2 cold flags (6)
    C_trn    = trainer midrank + z of the same 2 columns (4)  [control: same columns, trainer encoding]
    C_log    = within-feed z of the raw columns (2)           [control: plain popularity re-weighting]
    T3a_rep  = the proposer's 8-feature set (warm rank + cold flag for luo, invd20, et1, seq_cv_pop)
    T3a_other= the T3a features for et1 + seq_cv_pop only (columns NOT in the spec; xrel_more columns)
    T3a_spec2= the T3a features for luo + invd20 only (the spec's columns)
    noise4   = 4 N(0,1) columns
 CPU only.
"""
import os, sys, time, json
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

ROOT = Path("<repo>")
PIN = Path("/tmp/scratchpad/dpa/pipeline/out")
HERE = Path(__file__).resolve().parent
BOUND = 1790497920
LOG = open(HERE / "k01.log", "a")
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


CK = torch.load(ROOT / "nn_x/r6/ckpt/r6_cd_m02_r101.pt", map_location="cpu", weights_only=False)
CFG = CK["cfg"]; XT = CFG["xt_names"]
base = []
for k in range(1, 6):
    base += [f"f{k}", f"f{k}_zf", f"f{k}_rkf"]
base += ["cmiss", "log1p_norm", "cos_u", "dot_u", "smax", "smax8", "smax32", "smean", "lse", "top3"]
base += [f"smax_et{t}" for t in range(1, 11)] + ["log1p_ccnt", "ccnt_share"] + [f"log1p_ccnt_et{t}" for t in range(1, 11)]
base += ["first_inv", "fcs", "logM", "log1pH"]
for nme in ["cos_u", "smax", "lse", "top3", "log1p_ccnt"]:
    base += [f"{nme}_zf", f"{nme}_rkf"]
base += ["user0", "user0_log1pH"]
xt_keep = [n for i, n in enumerate(XT) if i not in set(CFG["xt_relonly"])]
rel = []
for j in CFG["xt_rel"]:
    rel += [f"{XT[j]}_rkf", f"{XT[j]}_zf"]
NAMES = base + ["xt:" + n for n in xt_keep] + ["xt:" + n for n in rel]
assert len(NAMES) == CFG["n_dense"] == 104
xi = {n: j for j, n in enumerate(XT)}
log("xt_rel cols:", [XT[j] for j in CFG["xt_rel"]], "xt_drop", CFG.get("xt_drop"))


def load(sub):
    X = np.load(PIN / f"{sub}_X.npy", mmap_mode="r"); R = np.load(PIN / f"{sub}_raw.npy", mmap_mode="r")
    return X, R, dict(np.load(PIN / f"{sub}_meta.npz"))


@nb.njit(parallel=True, cache=True)
def feed_z(x, off, M):
    out = np.zeros(len(x))
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]; s = 0.0
        for k in range(m):
            s += x[o + k]
        mu = s / m; v = 0.0
        for k in range(m):
            v += (x[o + k] - mu) ** 2
        sd = np.sqrt(v / m)
        for k in range(m):
            out[o + k] = (x[o + k] - mu) / (sd + 1e-9) if sd > 1e-12 else 0.0
    return out


@nb.njit(parallel=True, cache=True)
def trainer_relz(x, off, M):
    """exact trainer formulas: _pctrank over all rows (mid-rank, /(cnt-1)), _zscore with +1e-4."""
    rk = np.zeros(len(x)); z = np.zeros(len(x))
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]; s = 0.0
        for k in range(m):
            s += x[o + k]
        mu = s / m; v = 0.0
        for k in range(m):
            v += (x[o + k] - mu) ** 2
        sd = np.sqrt(v / m)
        den = max(m - 1, 1)
        for k in range(m):
            less = 0; eq = 0
            for j in range(m):
                if x[o + j] < x[o + k]:
                    less += 1
                elif x[o + j] == x[o + k]:
                    eq += 1
            rk[o + k] = (less + 0.5 * (eq - 1)) / den
            z[o + k] = (x[o + k] - mu) / (sd + 1e-4)
    return rk, z


@nb.njit(parallel=True, cache=True)
def spec_relz(x, off, M):
    """SPEC (--rel_warm): wm = x > 0; rank/z over warm rows with cnt_w (rank den clamp(cnt_w-1,1), z den cnt_w clamp 1);
    cold rows -> 0 for both."""
    rk = np.zeros(len(x)); z = np.zeros(len(x))
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]; nw = 0; s = 0.0
        for k in range(m):
            if x[o + k] > 0:
                nw += 1; s += x[o + k]
        if nw == 0:
            continue
        mu = s / nw; v = 0.0
        for k in range(m):
            if x[o + k] > 0:
                v += (x[o + k] - mu) ** 2
        sd = np.sqrt(v / nw)
        den = max(nw - 1, 1)
        for k in range(m):
            if x[o + k] <= 0:
                continue
            less = 0; eq = 0
            for j in range(m):
                u = x[o + j]
                if u <= 0:
                    continue
                if u < x[o + k]:
                    less += 1
                elif u == x[o + k]:
                    eq += 1
            rk[o + k] = (less + 0.5 * (eq - 1)) / den
            z[o + k] = (x[o + k] - mu) / (sd + 1e-4)
    return rk, z


@nb.njit(parallel=True, cache=True)
def warm_rank(x, off, M):
    out = np.full(len(x), -1.0)
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]; nw = 0
        for k in range(m):
            nw += x[o + k] > 0
        for k in range(m):
            v = x[o + k]
            if v <= 0:
                continue
            if nw == 1:
                out[o + k] = 0.5; continue
            less = 0; eq = 0
            for j in range(m):
                u = x[o + j]
                if u <= 0:
                    continue
                if u < v:
                    less += 1
                elif u == v:
                    eq += 1
            out[o + k] = (less + 0.5 * (eq - 1)) / (nw - 1)
    return out


def psi(ref, x, nb_=10):
    ref = ref[np.isfinite(ref)]; x = x[np.isfinite(x)]
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, nb_ + 1)[1:-1]))
    vals, cnt = np.unique(ref[:: max(1, len(ref) // 200000)], return_counts=True)
    mp = vals[cnt / cnt.sum() > 0.05]
    edges = np.unique(np.concatenate([qs, mp - 1e-9, mp + 1e-9]))
    br = np.bincount(np.searchsorted(edges, ref), minlength=len(edges) + 1) / len(ref)
    bx = np.bincount(np.searchsorted(edges, x), minlength=len(edges) + 1) / len(x)
    br = np.clip(br, 1e-4, None); bx = np.clip(bx, 1e-4, None)
    return float(((bx - br) * np.log(bx / br)).sum())


# ------------------------------------------------------------------ 1+2: parity and PSI
D = {}
for sub in ["trn", "hold", "val", "test"]:
    X, R, mt = load(sub)
    if sub == "test":
        pubf = mt["ts"] < BOUND
        for nm, sel in [("pub", pubf), ("priv", ~pubf)]:
            rows = np.repeat(sel, mt["M"]); Ms = mt["M"][sel]
            D[nm] = dict(X=None, R=np.asarray(R[rows][:, [xi["it_hist_all_luo"], xi["seq_cv_invd20"]]]).astype(np.float64),
                         Xr=np.asarray(X[rows][:, [NAMES.index(f"xt:{c}_{s}") for c in ("it_hist_all_luo", "seq_cv_invd20") for s in ("rkf", "zf")]]).astype(np.float64),
                         M=Ms, off=np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64))
    else:
        D[sub] = dict(R=np.asarray(R[:, [xi["it_hist_all_luo"], xi["seq_cv_invd20"]]]).astype(np.float64),
                      Xr=np.asarray(X[:, [NAMES.index(f"xt:{c}_{s}") for c in ("it_hist_all_luo", "seq_cv_invd20") for s in ("rkf", "zf")]]).astype(np.float64),
                      M=mt["M"].astype(np.int64), off=mt["off"].astype(np.int64))
log(f"loaded {time.time()-T0:.0f}s")
ENC = {}
for s, d in D.items():
    out = {}
    for c, cn in enumerate(("luo", "invd20")):
        x = np.nan_to_num(d["R"][:, c])
        trk, tz = trainer_relz(x, d["off"], d["M"])
        srk, sz = spec_relz(x, d["off"], d["M"])
        out[cn] = dict(x=x, trk=trk, tz=tz, srk=srk, sz=sz, Xrk=d["Xr"][:, 2 * c], Xz=d["Xr"][:, 2 * c + 1])
        if s in ("val", "trn"):
            log(f"parity {s} {cn}: |trainer rk recompute - X| max {np.abs(trk - out[cn]['Xrk']).max():.2e}  "
                f"|z| max {np.abs(tz - out[cn]['Xz']).max():.2e}  (raw min {x.min():.3f} max {x.max():.3f}; cold {np.mean(x <= 0):.3f})")
    ENC[s] = out

log("\n== PSI (10 ref-quantile bins + mass points). Relevant drift for a train-fitted NN = trn -> val/pub/priv ==")
for cn in ("luo", "invd20"):
    for enc in ("trk", "tz", "srk", "sz"):
        line = f"  {cn:6s} {enc:4s}"
        for a, bb in (("trn", "val"), ("trn", "pub"), ("trn", "priv"), ("val", "priv"), ("hold", "val"), ("trn", "hold")):
            line += f" | {a}->{bb} {psi(ENC[a][cn][enc], ENC[bb][cn][enc]):.3f}"
        # warm rows only / cold rows only
        wt = ENC["trn"][cn]["x"] > 0; wp = ENC["priv"][cn]["x"] > 0
        line += f" || warm-rows trn->priv {psi(ENC['trn'][cn][enc][wt], ENC['priv'][cn][enc][wp]):.3f}"
        if enc in ("trk", "tz"):
            line += f" cold-rows trn->priv {psi(ENC['trn'][cn][enc][~wt], ENC['priv'][cn][enc][~wp]):.3f}"
        log(line)
    for s in ("trn", "hold", "val", "pub", "priv"):
        e = ENC[s][cn]; w = e["x"] > 0
        log(f"    {cn} {s:5s} cold {1-w.mean():.3f} | trainer rk cold {e['trk'][~w].mean():.3f} warm {e['trk'][w].mean():.3f} | "
            f"spec rk warm {e['srk'][w].mean():.3f} | trainer z cold {e['tz'][~w].mean():+.3f} warm {e['tz'][w].mean():+.3f} | spec z warm sd {e['sz'][w].std():.3f}")
del D
log(f"PSI done {time.time()-T0:.0f}s")

# ------------------------------------------------------------------ 3: nested LIN residuals on G3
Xv, Rv, mv = load("val"); Xh, Rh, mh = load("hold")
Mv, offv, yv = mv["M"].astype(np.int64), mv["off"].astype(np.int64), mv["y"].astype(np.float64)
Mh, offh, yh = mh["M"].astype(np.int64), mh["off"].astype(np.int64), mh["y"].astype(np.float64)
hm = np.load(ROOT / "preds/holdout/holdout_meta.npy"); assert (hm[0] == Mh).all() and (hm[1] == offh).all()
assert np.allclose(np.load(ROOT / "preds/holdout/holdout_y.npy"), yh)
user = mv["user"]; nF = len(Mv); fid = np.repeat(np.arange(nF), Mv); npos_f = np.bincount(fid, weights=yv, minlength=nF)
zz = lambda a, off, M: feed_z(np.asarray(a, np.float64), off, M)
b = zz(0.5 * zz(np.load(ROOT / "preds/ens_G3_ft_val.npy"), offv, Mv) + 0.5 * zz(np.load(ROOT / "preds/ens_G3_to_val.npy"), offv, Mv), offv, Mv)
MEM = [f"v23_gru_pre2s_s50{i}" for i in range(1, 7)] + [f"r6_cd_m02_r10{i}" for i in range(1, 8)] + [f"r6_cd_m02_p70k_r10{i}" for i in range(1, 8)]
bh = zz(np.mean([zz(np.load(ROOT / f"preds/holdout/nn_{m}_holdout.npy"), offh, Mh) for m in MEM], 0), offh, Mh)
ts = mh["ts"].astype(np.int64); hday = (ts + 10800) // 86400; late = np.repeat(hday >= hday.min() + 11, 1)  # Sep 17-19
dayv = (mv["ts"].astype(np.int64) + 10800) // 86400
log(f"late holdout feeds (day >= first+11): {late.sum()}")
WPRIV = pmix.weights()["private"]
folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=user))
rowfold = np.empty(len(yv), np.int64)
for k, (trf, vaf) in enumerate(folds):
    rowfold[np.isin(fid, vaf)] = k
ndv = lambda s: ndcg_per_feed(s, yv, offv, Mv)
ndh = lambda s: ndcg_per_feed(s, yh, offh, Mh)
luo_v = np.nan_to_num(Rv[:, xi["it_hist_all_luo"]]).astype(np.float64)
cold = luo_v <= 0
coldpos_f = np.bincount(fid, weights=yv * cold, minlength=nF)
coldsh = np.bincount(fid, weights=cold.astype(float), minlength=nF) / Mv
SL = {"M<50": Mv < 50, "M>=50": Mv >= 50, "tue": dayv <= dayv.min() + 1, "thu": dayv >= dayv.min() + 2,
      "coldsh>=.45": coldsh >= 0.45, "coldsh<.3": coldsh < 0.3}


def nll(theta, G, rows):
    s = G @ theta; ff = fid[rows]
    e = np.exp(np.clip(s, -50, 20.0)); Z = np.bincount(ff, weights=e, minlength=nF)
    ok = Z > 0; lZ = np.log(np.where(ok, Z, 1.0)); yy = yv[rows]
    npos_used = np.bincount(ff, weights=yy, minlength=nF)
    L = -(yy * s).sum() + (npos_used * lZ).sum()
    p = e / np.where(ok, Z, 1.0)[ff]; w = npos_used[ff] * p - yy
    return L, G.T @ w


def fit_lin(G, tr):
    th0 = np.zeros(G.shape[1]); th0[0] = 0.94
    rows = np.flatnonzero(tr)
    return minimize(lambda t: nll(t, G[tr], rows), th0, jac=True, method="L-BFGS-B", options=dict(maxiter=300)).x


G0 = b[:, None]; pred = np.empty(len(yv))
for k in range(5):
    tr = rowfold != k; pred[~tr] = G0[~tr] @ fit_lin(G0, tr)
beta_full = fit_lin(G0, np.ones(len(yv), bool))[0]
CV = ndv(pred); CH = ndh(beta_full * bh)
log(f"base: G3 val {ndv(b).mean():.5f} hold {ndh(bh).mean():.5f}; LIN control val {CV.mean():.5f} hold {CH.mean():.5f}  {time.time()-T0:.0f}s")


def run_lin(name, Gv, Gh):
    G = np.column_stack([b, Gv]); pred = np.empty(len(yv))
    for k in range(5):
        tr = rowfold != k; pred[~tr] = G[~tr] @ fit_lin(G, tr)
    th = fit_lin(G, np.ones(len(yv), bool))
    dv = ndv(pred) - CV; dh = ndh(np.column_stack([bh, Gh]) @ th) - CH
    se = dv.std() / np.sqrt(len(dv)); seh = dh.std() / np.sqrt(len(dh))
    pm, pse = pmix.wmean_se(dv, WPRIV)
    dl = dh[late]; sel = dl.std() / np.sqrt(len(dl))
    line = (f"{name:12s} val {dv.mean():+.5f} ({se:.5f}) z {dv.mean()/se:+.2f} | pmix {pm:+.5f} ({pse:.5f}) z {pm/pse:+.2f} | "
            f"hold {dh.mean():+.5f} ({seh:.5f}) z {dh.mean()/seh:+.2f} | late {dl.mean():+.5f} z {dl.mean()/sel:+.2f}")
    for sn, m in SL.items():
        line += f" | {sn} {dv[m].mean():+.5f}"
    log(line + f" | w {np.round(th, 3)}")
    with open(HERE / "k01_results.jsonl", "a") as fh:
        fh.write(json.dumps(dict(name=name, val=float(dv.mean()), val_se=float(se), pmix=float(pm), pmix_se=float(pse),
                                 hold=float(dh.mean()), hold_se=float(seh), late=float(dl.mean()), late_se=float(sel))) + "\n")
    return dv, dh


def feats(X, R, off, M, kind):
    off = off.astype(np.int64); M = M.astype(np.int64)
    out = []
    if kind in ("S_spec", "S_specf"):
        for c in ("it_hist_all_luo", "seq_cv_invd20"):
            rk, z = spec_relz(np.nan_to_num(R[:, xi[c]].astype(np.float64)), off, M); out += [rk, z]
        if kind == "S_specf":
            for c in ("it_hist_all_luo", "seq_cv_invd20"):
                out.append((np.nan_to_num(R[:, xi[c]]) <= 0).astype(np.float64))
    elif kind == "C_trn":
        for c in ("it_hist_all_luo", "seq_cv_invd20"):
            out += [X[:, NAMES.index(f"xt:{c}_rkf")].astype(np.float64), X[:, NAMES.index(f"xt:{c}_zf")].astype(np.float64)]
    elif kind == "C_log":
        for c in ("it_hist_all_luo", "seq_cv_invd20"):
            out.append(feed_z(np.log1p(np.maximum(np.nan_to_num(R[:, xi[c]].astype(np.float64)), 0)), off, M))
    else:
        cols = {"T3a_rep": ["it_hist_all_luo", "seq_cv_invd20", "it_hist_et1_luo", "seq_cv_pop"],
                "T3a_other": ["it_hist_et1_luo", "seq_cv_pop"], "T3a_spec2": ["it_hist_all_luo", "seq_cv_invd20"]}[kind]
        for c in cols:
            w = warm_rank(np.nan_to_num(R[:, xi[c]].astype(np.float64)), off, M)
            out += [np.where(w < 0, 0.0, w), (w < 0).astype(np.float64)]
    return np.column_stack(out)


Xv = np.asarray(Xv); Rv = np.asarray(Rv); Xh = np.asarray(Xh); Rh = np.asarray(Rh)
which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["noise4", "T3a_rep", "T3a_spec2", "T3a_other", "S_spec", "S_specf", "C_trn", "C_log"]
rng = np.random.default_rng(5)
for n in which:
    t = time.time()
    if n == "noise4":
        run_lin(n, rng.standard_normal((len(yv), 4)), rng.standard_normal((len(yh), 4)))
    else:
        run_lin(n, feats(Xv, Rv, offv, Mv, n), feats(Xh, Rh, offh, Mh, n))
    log(f"   ({time.time()-t:.0f}s)")
log(f"DONE {time.time()-T0:.0f}s")
