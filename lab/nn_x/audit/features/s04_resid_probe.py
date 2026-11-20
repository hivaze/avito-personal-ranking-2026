"""Nested (user-grouped) residual-on-G3 probe on val for new and unused features (CPU, read-only on project files).

Base b = within-feed z of 0.5 z(ens_G3_ft_val) + 0.5 z(ens_G3_to_val) = the G3 submission score (val 0.40032).
Folds: GroupKFold(5) by val user_id, exactly nn/finetune.py (no shuffle). Two residual models per feature (design of
featrev inventory c07 / feat_new/hz_resid.py):
  LIN: listwise softmax s = beta b + w1 z(x) + w2 (pctrank(x) - 0.5) + w3 isnan(x); control s = beta b.
  GBM: LightGBM lambdarank on [b, x, z(x), isnan(x)] with init_score beta_k b, 60 rounds, 7 leaves,
       min_data_in_leaf 2000, l2 50; control = the same on [b].
Paired per-feed NDCG delta vs the control, SE over 14,423 feeds; slices M<50 / M>=50, Tue / Thu, feeds whose
positives are all LUO-cold (coldpos) / all warm (warmpos). Group GBMs per family. 4 noise columns calibrate the null.

  python s04_resid_probe.py <shard k of n>   e.g. "0/2"
Outputs: out/resid_<k>.jsonl (one line per feature / group), out/s04_<k>.log.
"""
import sys, json, time
sys.path.insert(0, "<repo>/nn_x/audit/features")
from lib_av import *
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

HERE = Path(__file__).resolve().parent
NEWF = HERE / "out/feats/val"
shard = sys.argv[1] if len(sys.argv) > 1 else "0/1"
ks, ns = map(int, shard.split("/"))
LOG = open(HERE / f"out/s04_{ks}.log", "a")
OUTJ = HERE / f"out/resid_{ks}.jsonl"
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


f, M, off = meta("val")
user = f["user_id"].to_numpy(); y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
nF = len(M); fid = np.repeat(np.arange(nF), M); npos_f = np.bincount(fid, weights=y, minlength=nF)
day = (f["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
luo = np.load(feat_file("val", "it_hist_all_luo")); cold = luo == 0
coldpos_f = np.bincount(fid, weights=y * cold, minlength=nF)
SL = {"M<50": M < 50, "M>=50": M >= 50, "tue": day <= day.min() + 1, "thu": day >= day.min() + 2,
      "coldpos": (coldpos_f == npos_f) & (npos_f > 0), "warmpos": (coldpos_f == 0) & (npos_f > 0)}
zz = lambda a: feed_z_nanmin(np.asarray(a, np.float64), off, M)
b = zz(0.5 * zz(np.load(ROOT / "preds/ens_G3_ft_val.npy")) + 0.5 * zz(np.load(ROOT / "preds/ens_G3_to_val.npy")))
folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=user))
rowfold = np.empty(len(y), np.int64)
for k, (trf, vaf) in enumerate(folds):
    rowfold[np.isin(fid, vaf)] = k


def nll(theta, G, rows):
    s = G @ theta; ff = fid[rows]
    e = np.exp(np.clip(s, -50, 20.0)); Z = np.bincount(ff, weights=e, minlength=nF)
    ok = Z > 0; lZ = np.log(np.where(ok, Z, 1.0)); yy = y[rows]
    L = -(yy * s).sum() + (npos_f * lZ).sum()
    p = e / np.where(ok, Z, 1.0)[ff]; w = npos_f[ff] * p - yy
    return L, G.T @ w


def fit_lin(G, tr):
    th0 = np.zeros(G.shape[1]); th0[0] = 0.94
    rows = np.flatnonzero(tr)
    r = minimize(lambda t: nll(t, G[tr], rows), th0, jac=True, method="L-BFGS-B", options=dict(maxiter=200))
    return r.x


def cv_lin(G):
    pred = np.empty(len(y))
    for k in range(5):
        tr = rowfold != k; pred[~tr] = G[~tr] @ fit_lin(G, tr)
    return pred


betas = [fit_lin(b[:, None], rowfold != k)[0] for k in range(5)]
PARAMS = dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=7, min_data_in_leaf=2000,
              lambda_l2=50.0, feature_fraction=1.0, bagging_fraction=1.0, lambdarank_truncation_level=1000,
              max_bin=63, verbose=-1, num_threads=12, deterministic=True, force_row_wise=True, seed=0)


def cv_gbm(F):
    pred = np.empty(len(y))
    for k, (trf, vaf) in enumerate(folds):
        tr = rowfold != k; init = betas[k] * b
        ds = lgb.Dataset(F[tr], y[tr], group=M[np.sort(trf)], init_score=init[tr])
        bst = lgb.train(PARAMS, ds, 60); pred[~tr] = init[~tr] + bst.predict(F[~tr])
    return pred


nd = lambda s: ndcg_per_feed(s, y, off, M)
c_lin = nd(cv_lin(b[:, None])); c_gbm = nd(cv_gbm(b[:, None].astype(np.float32)))
log(f"control: base {nd(b).mean():.5f} lin {c_lin.mean():.5f} gbm {c_gbm.mean():.5f} betas {np.round(betas, 3)} {time.time()-T0:.0f}s")


def rep(name, kind, d):
    se = d.std() / np.sqrt(len(d)); out = dict(name=name, kind=kind, delta=float(d.mean()), se=float(se), z=float(d.mean() / se))
    line = f"{name:26s} {kind:4s} {d.mean():+.5f} ({se:.5f}) z {d.mean()/se:+.2f}"
    for sn, m in SL.items():
        s2 = d[m].std() / np.sqrt(m.sum()); out[sn] = [float(d[m].mean()), float(s2)]; line += f" | {sn} {d[m].mean():+.5f}"
    log(line); return out


def cols_of(x):
    x = np.asarray(x, np.float64); isn = np.isnan(x).astype(np.float64)
    return zz(x), pctrank(x, off, M), isn


def probe(name, x):
    z, rk, isn = cols_of(x)
    G = np.stack([b, z, rk] + ([isn] if isn.any() else []), 1)
    d_lin = nd(cv_lin(G)) - c_lin
    F = np.stack([b, np.nan_to_num(np.asarray(x, np.float64), nan=np.nanmin(x) - 1.0 if np.isfinite(np.nanmin(x)) else 0.0), z] + ([isn] if isn.any() else []), 1).astype(np.float32)
    d_gbm = nd(cv_gbm(F)) - c_gbm
    return rep(name, "LIN", d_lin), rep(name, "GBM", d_gbm)


def probe_group(name, xs):
    F = [b]
    for x in xs:
        z, rk, isn = cols_of(x)
        F += [np.nan_to_num(np.asarray(x, np.float64), nan=np.nanmin(x) - 1.0 if np.isfinite(np.nanmin(x)) else 0.0), z]
        if isn.any():
            F.append(isn)
    d = nd(cv_gbm(np.stack(F, 1).astype(np.float32))) - c_gbm
    return rep(name, "GGBM", d)


def load_any(n):
    p = NEWF / f"{n}.npy"
    if p.exists():
        return np.load(p).astype(np.float64)
    return np.load(feat_file("val", n)).astype(np.float64)


NEW = sorted(p.stem for p in NEWF.glob("*.npy"))
ONDISK = {p.stem for d in (FEATS, FEATS_NEW) for p in (d / "val").glob("*.npy")}
UNUSED_PREFIX = ("trd_", "clu_", "tte_", "als_", "u_het", "ixh_", "hzcv_", "src_", "ud_", "seq_cvlag", "it_tr_", "u_tr_")
UNUSED = sorted(n for n in ONDISK if n.startswith(UNUSED_PREFIX)) + [
    "hist_n_uniq_cat", "hour", "it_cand_cnt_split", "it_cand_cnt_train", "pre2_fe_noid", "pre2_hmax8", "pre2_idmax32",
    "pre2_id_contrib", "prew_fe_z", "prew_new_z", "prew_hmax32", "prew_idmax32", "prew_fe_noid", "ctx_hs_d7", "ctx_hs_e1k64",
    "ctx_hs_k128t10", "ctx_hs_k256", "ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_mean", "ctx_nd_dupb_h", "ctx_nd_dupb_u",
    "ctx_nd_dupb_f1", "seq_cos_last1", "seq_dcos_deep", "seq_dcos_hl2", "seq_dcos_nt_hl10", "seq_dcos_rep", "seq_dcos_s1",
    "seq_dcos_s2", "seq_dcos_t1_hl5", "seq_dmax_hl10", "seq_max2_gap", "seq_max_new8", "seq_max_nt", "seq_max_old",
    "seq_max_s1", "seq_max_t1", "cos_h50", "cos_het5", "cos_het6", "cos_het7", "cos_het8", "cos_het9", "cos_het10", "it_hist_all"]
UNUSED = [n for n in UNUSED if n in ONDISK]
rng = np.random.default_rng(0)
NOISE = {f"noise{i}": rng.standard_normal(len(y)) for i in range(4)}
GROUPS = {"h_": [n for n in NEW if n.startswith("h_")], "rc_": [n for n in NEW if n.startswith("rc_")],
          "a_": [n for n in NEW if n.startswith("a_")], "u_": [n for n in NEW if n.startswith("u_")],
          "xs_": [n for n in NEW if n.startswith("xs_")], "k_": [n for n in NEW if n.startswith("k_")],
          "xs_causal": [n for n in NEW if n in ("xs_cand_before", "xs_hist_before")],
          "trd_": [n for n in UNUSED if n.startswith("trd_")], "clu_": [n for n in UNUSED if n.startswith("clu_")],
          "tte_": [n for n in UNUSED if n.startswith("tte_")], "als_": [n for n in UNUSED if n.startswith("als_")],
          "seq_unused": [n for n in UNUSED if n.startswith("seq_") and not n.startswith("seq_cvlag")],
          "ctx_hs_unused": [n for n in UNUSED if n.startswith("ctx_hs_")], "prew_": [n for n in UNUSED if n.startswith("prew_")],
          "ixh_": [n for n in UNUSED if n.startswith("ixh_")], "ud_": [n for n in UNUSED if n.startswith("ud_")],
          "NEW_all": NEW, "NEW_noxs": [n for n in NEW if not n.startswith("xs_")]}
jobs = [("single", n) for n in list(NOISE) + NEW + UNUSED] + [("group", g) for g in GROUPS if GROUPS[g]]
done = set()
if OUTJ.exists():
    done = {(json.loads(l)["name"], json.loads(l)["kind"]) for l in OUTJ.read_text().splitlines() if l.strip()}
log(f"{len(NEW)} new, {len(UNUSED)} unused, {len(GROUPS)} groups; shard {shard}; {len(done)} done")
for i, (kind, n) in enumerate(jobs):
    if i % ns != ks:
        continue
    try:
        if kind == "single":
            if (n, "LIN") in done and (n, "GBM") in done:
                continue
            x = NOISE[n] if n in NOISE else load_any(n)
            if np.isnan(x).all() or np.nanstd(x) == 0:
                log(f"{n}: constant / all-NaN, skipped"); continue
            res = probe(n, x)
        else:
            if (n, "GGBM") in done:
                continue
            res = [probe_group(n, [load_any(m) for m in GROUPS[n]])]
        with open(OUTJ, "a") as fh:
            for r in res:
                fh.write(json.dumps(r) + "\n")
    except Exception as e:  # keep going; report
        log(f"{n}: ERROR {e!r}")
log(f"DONE {time.time()-T0:.0f}s")
