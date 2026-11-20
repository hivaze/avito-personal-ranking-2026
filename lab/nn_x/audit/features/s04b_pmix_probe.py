"""Residual-on-G3 probe (same design as s04) for a short list of features / groups, additionally reporting the
private-mix (src/pmix.py, M x cold-share reweighting of val feeds) and public-mix weighted deltas, and saving the
per-feed deltas (out/pmix_probe/<name>_<kind>.npy)."""
import sys, json, time
sys.path.insert(0, "<repo>/nn_x/audit/features")
from lib_av import *
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
import pmix

HERE = Path(__file__).resolve().parent; NEWF = HERE / "out/feats/val"; OUTD = HERE / "out/pmix_probe"; OUTD.mkdir(exist_ok=True)
LOG = open(HERE / "out/s04b.log", "a")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


f, M, off = meta("val"); user = f["user_id"].to_numpy(); y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
nF = len(M); fid = np.repeat(np.arange(nF), M); npos_f = np.bincount(fid, weights=y, minlength=nF)
day = (f["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
cold = np.load(feat_file("val", "it_hist_all_luo")) == 0; coldpos_f = np.bincount(fid, weights=y * cold, minlength=nF)
SL = {"M<50": M < 50, "M>=50": M >= 50, "tue": day <= day.min() + 1, "thu": day >= day.min() + 2,
      "coldpos": (coldpos_f == npos_f) & (npos_f > 0), "warmpos": (coldpos_f == 0) & (npos_f > 0)}
W = pmix.weights()
zz = lambda a: feed_z_nanmin(np.asarray(a, np.float64), off, M)
b = zz(0.5 * zz(np.load(ROOT / "preds/ens_G3_ft_val.npy")) + 0.5 * zz(np.load(ROOT / "preds/ens_G3_to_val.npy")))
folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=user))
rowfold = np.empty(len(y), np.int64)
for k, (trf, vaf) in enumerate(folds):
    rowfold[np.isin(fid, vaf)] = k


def nll(theta, G, rows):
    s = G @ theta; ff = fid[rows]; e = np.exp(np.clip(s, -50, 20.0)); Z = np.bincount(ff, weights=e, minlength=nF)
    ok = Z > 0; lZ = np.log(np.where(ok, Z, 1.0)); yy = y[rows]
    return -(yy * s).sum() + (npos_f * lZ).sum(), G.T @ (npos_f[ff] * (e / np.where(ok, Z, 1.0)[ff]) - yy)


def fit_lin(G, tr):
    th0 = np.zeros(G.shape[1]); th0[0] = 0.94; rows = np.flatnonzero(tr)
    return minimize(lambda t: nll(t, G[tr], rows), th0, jac=True, method="L-BFGS-B", options=dict(maxiter=200)).x


def cv_lin(G):
    pred = np.empty(len(y))
    for k in range(5):
        tr = rowfold != k; pred[~tr] = G[~tr] @ fit_lin(G, tr)
    return pred


betas = [fit_lin(b[:, None], rowfold != k)[0] for k in range(5)]
PARAMS = dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=7, min_data_in_leaf=2000, lambda_l2=50.0,
              feature_fraction=1.0, bagging_fraction=1.0, lambdarank_truncation_level=1000, max_bin=63, verbose=-1, num_threads=12,
              deterministic=True, force_row_wise=True, seed=0)


def cv_gbm(F):
    pred = np.empty(len(y))
    for k, (trf, vaf) in enumerate(folds):
        tr = rowfold != k; init = betas[k] * b
        bst = lgb.train(PARAMS, lgb.Dataset(F[tr], y[tr], group=M[np.sort(trf)], init_score=init[tr]), 60)
        pred[~tr] = init[~tr] + bst.predict(F[~tr])
    return pred


nd = lambda s: ndcg_per_feed(s, y, off, M)
c_lin = nd(cv_lin(b[:, None])); c_gbm = nd(cv_gbm(b[:, None].astype(np.float32)))


def rep(name, kind, d):
    np.save(OUTD / f"{name}_{kind}.npy", d.astype(np.float32))
    se = d.std() / np.sqrt(len(d)); out = dict(name=name, kind=kind, delta=float(d.mean()), se=float(se))
    line = f"{name:26s} {kind:4s} {d.mean():+.5f} ({se:.5f}) z {d.mean()/se:+.2f}"
    for wn in ("private", "public"):
        m, s2 = pmix.wmean_se(d, W[wn]); out[f"pmix_{wn}"] = [float(m), float(s2)]; line += f" | {wn}-mix {m:+.5f} ({s2:.5f})"
    for sn, mm in SL.items():
        out[sn] = float(d[mm].mean()); line += f" | {sn} {d[mm].mean():+.5f}"
    log(line); return out


def cols_of(x):
    x = np.asarray(x, np.float64); return zz(x), pctrank(x, off, M), np.isnan(x).astype(np.float64)


def fill(x):
    x = np.asarray(x, np.float64); mn = np.nanmin(x)
    return np.nan_to_num(x, nan=(mn - 1.0) if np.isfinite(mn) else 0.0)


def probe(name, xs):
    G = [b]; F = [b]
    for x in xs:
        z, rk, isn = cols_of(x); G += [z, rk]; F += [fill(x), z]
        if isn.any():
            G.append(isn); F.append(isn)
    out = [rep(name, "LIN", nd(cv_lin(np.stack(G, 1))) - c_lin), rep(name, "GBM", nd(cv_gbm(np.stack(F, 1).astype(np.float32))) - c_gbm)]
    with open(HERE / "out/pmix_probe.jsonl", "a") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")


def load_any(n):
    p = NEWF / f"{n}.npy"
    return np.load(p if p.exists() else feat_file("val", n)).astype(np.float64)


JOBS = {"a_age_days": ["a_age_days"], "a_luo_rate": ["a_luo_rate"], "a_luo_resid_split": ["a_luo_resid_split"],
        "a_zero_rate_split": ["a_zero_rate_split"], "a_luo_resid_tr": ["a_luo_resid_tr"],
        "rc_it_hist_all_luo_z": ["rc_it_hist_all_luo_z"], "a_group": ["a_age_days", "a_luo_rate", "a_luo_resid_split", "a_zero_rate_split"],
        "age_x_luo": ["a_age_days", "it_hist_all_luo"], "k_ctr": ["k_ctr"], "k_group": ["k_pop", "k_ctr", "k_coec", "k_top1", "k_n95", "k_pop5"],
        "xs_before": ["xs_cand_before", "xs_hist_before"], "xs_all": ["xs_cand_other", "xs_cand_before", "xs_cand_after", "xs_hist_other", "xs_hist_before", "xs_hist_after"],
        "u_res_cos": ["u_res_cos"], "u_group": ["u_minus_hall", "u_minus_h10", "u_x_maxsim", "u_res_cos", "u_hc_cos", "u_hc_proj"]}
sel = sys.argv[1:] or list(JOBS)
log(f"control lin {c_lin.mean():.5f} gbm {c_gbm.mean():.5f}; private-mix cold share val {W['cs_val']:.3f} -> priv {W['cs_priv']:.3f}")
for n in sel:
    try:
        probe(n, [load_any(m) for m in JOBS[n]])
    except Exception as e:
        log(f"{n}: ERROR {e!r}")
log("DONE")
