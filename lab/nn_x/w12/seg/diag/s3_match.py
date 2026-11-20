"""A1: how much of the H=256 deficit remains after matching on label-free feed composition (and, separately, n_pos)."""
from lib import *
import json, lightgbm as lgb, warnings
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
cov = pl.read_parquet(OUT / "feed_cov.parquet")
e = np.load(OUT / "final_ndcg.npy"); au = np.load(OUT / "rk_auc.npz")["FINAL"]; E0 = np.load(OUT / "rand_ndcg.npy")
# range-restriction covariates (label-free)
smw = np.load(OUT / "cos_smaxw.npy"); s64 = smw[:, 3].astype(np.float64); s64[s64 < -1] = np.nan
cu = np.load(OUT / "cos_u.npy").astype(np.float64)
f = lambda n: np.nan_to_num(np.load(FC / f"{n}.npy").astype(np.float64), nan=0.0)
lp, lx = np.log1p(np.maximum(f("it_hist_all_luo"), 0)), np.log1p(np.maximum(f("xp_imp_c50"), 0))
def fm(x):
    ok = ~np.isnan(x); n = np.maximum(np.bincount(fid, weights=ok, minlength=NF), 1); xx = np.where(ok, x, 0)
    mu = np.bincount(fid, weights=xx, minlength=NF) / n; sd = np.sqrt(np.maximum(np.bincount(fid, weights=xx * xx, minlength=NF) / n - mu ** 2, 0)); return mu, sd
rr = {}
for nm, x in (("smax64", s64), ("cos_u", cu), ("logpop", lp), ("logexp", lx)):
    rr[nm + "_mean"], rr[nm + "_sd"] = fm(x)
d = ROOT / "data/competition/val/item_embeddings"; T = np.fromfile(d / "data.memmap", dtype=np.int8).reshape(json.loads((d / "data.meta").read_text())["shape"])
ce_i = np.load(FLAT / "val/cand_emb_idx.npy"); pc = np.zeros(NF)
for k in range(NF):
    ci = ce_i[ov[k]:ov[k] + Mv[k]]; ci = ci[ci >= 0]
    if len(ci) < 2: continue
    C = T[ci].astype(np.float32); C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-3); s = C.sum(0); pc[k] = (s @ s - len(ci)) / (len(ci) * (len(ci) - 1))
rr["cand_pair_cos"] = pc
X1n = ["M", "logM", "day", "hour", "uid0", "cold_share", "emb_miss_share", "n_ccat", "top_ccat_share", "ccat_in_hist_share", "h64_ncat", "h64_topcat", "h64_noemb",
       "h64_distinct", "h64_src_top"] + [f"h64_et{t}" for t in range(1, 11)]
C = {c: cov[c].to_numpy().astype(np.float64) for c in cov.columns}; C["logM"] = np.log(Mv); C.update(rr)
X2n = X1n + list(rr) + ["cand_in_hist_share"]
pl.DataFrame({k: C[k] for k in X2n}).write_parquet(OUT / "feed_cov2.parquet")
SETS = {"X1 (declared label-free set)": X1n, "X2 = X1 + range restriction": X2n, "X1 + n_pos (NOT label-free)": X1n + ["npos"], "X2 + n_pos (NOT label-free)": X2n + ["npos"],
        "M + n_pos only (NOT label-free)": ["M", "npos"], "M only": ["M", "logM"]}
C["npos"] = NPOS.astype(np.float64)
gkf = list(GroupKFold(5).split(np.arange(NF), groups=clu))
def crossfit(y, names, ref):
    X = np.stack([C[n] for n in names], 1); pred = np.zeros(NF)
    for tr, te in gkf:
        trr = tr[ref[tr]]
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10,
                              n_jobs=12, verbose=-1, random_state=0).fit(X[trr], y[trr])
        pred[te] = m.predict(X[te])
    return pred
print("### covariate shift, H=256 vs NEAR and vs REST (standardised mean difference, sd of ALL feeds)")
rows = []
for n in X2n + ["npos"]:
    x = C[n]; sd = x.std() + 1e-12
    rows.append([n, f"{x[CAP].mean():.3f}", f"{x[NEAR].mean():.3f}", f"{x[REST].mean():.3f}", f"{(x[CAP].mean() - x[NEAR].mean()) / sd:+.2f}", f"{(x[CAP].mean() - x[REST].mean()) / sd:+.2f}"])
print(tab(rows, ["covariate", "mean H=256", "mean NEAR", "mean REST", "SMD vs NEAR", "SMD vs REST"]))
for yname, y in (("NDCG", e), ("per-feed AUC", au), ("skill", (e - E0) / (1 - E0))):
    for refname, ref in (("REST", REST), ("NEAR", NEAR)):
        d0, s0 = cdiff(y, CAP, ref); rows = [["raw (no matching)", f"{d0:+.5f}", f"{s0:.5f}", f"{d0 / s0:+.1f}", "100%", ""]]
        for sn, names in SETS.items():
            yy = np.where(np.isnan(y), np.nanmean(y), y)
            p = crossfit(yy, names, ref); r = y - p
            da, sa = cdiff(r, CAP, ref); okr = ref & ~np.isnan(y); r2 = 1 - ((y[okr] - p[okr]) ** 2).sum() / ((y[okr] - y[okr].mean()) ** 2).sum()
            rows.append([sn, f"{da:+.5f}", f"{sa:.5f}", f"{da / sa:+.1f}", f"{100 * da / d0:.0f}%", f"{r2:.4f}"])
        print(f"\n### outcome {yname}; reference {refname}: D = mean(H=256) - mean({refname}), cross-fitted LightGBM on {refname} feeds (5-fold by user)\n" + tab(rows, ["matching set", "D", "SE", "z", "share of raw left", "OOF R2 on ref"]))
# OLS cross-check (declared): NDCG ~ X + 1[H=256] on ALL feeds, cluster-robust
def ols(y, names):
    X = np.stack([C[n] for n in names] + [CAP.astype(float), np.ones(NF)], 1); X = (X - X.mean(0)) / (X.std(0) + 1e-12); X[:, -1] = 1; X[:, -2] = CAP
    b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b; XtXi = np.linalg.pinv(X.T @ X)
    G = np.zeros((NCL, X.shape[1])); np.add.at(G, clu, X * r[:, None]); V = XtXi @ (G.T @ G) @ XtXi
    return b[-2], np.sqrt(V[-2, -2])
print("\n### OLS cross-check: coefficient of 1[H=256] on NDCG (all feeds, cluster-robust SE)")
print(tab([[sn, "%+.5f" % ols(e, names)[0], "%.5f" % ols(e, names)[1]] for sn, names in SETS.items()], ["covariates", "coef 1[H=256]", "SE"]))
# exact-cell matching (declared): M decile x cold-share tercile, reference REST re-weighted to the H=256 cell mix
mq = np.digitize(Mv, np.quantile(Mv, np.linspace(0.1, 0.9, 9))); cq = np.digitize(C["cold_share"], np.quantile(C["cold_share"], [1 / 3, 2 / 3])); cell = mq * 3 + cq
for extra, nm in ((None, "M decile x cold tercile"), (np.minimum(NPOS, 3), "M decile x cold tercile x n_pos(1,2,3+) (NOT label-free)")):
    cl = cell if extra is None else cell * 4 + extra; w = np.zeros(NF)
    for c_ in np.unique(cl):
        a, b_ = CAP & (cl == c_), REST & (cl == c_)
        if a.sum() and b_.sum(): w[b_] = a.sum() / b_.sum()
    wm = (w * e).sum() / w.sum(); print(f"exact-cell matching [{nm}]: NDCG H=256 {e[CAP].mean():.5f} vs re-weighted REST {wm:.5f} -> D {e[CAP].mean() - wm:+.5f} (raw {e[CAP].mean() - e[REST].mean():+.5f}); ESS of REST {w.sum() ** 2 / (w ** 2).sum():.0f}")
# within-cap splits
seen = np.load(OUT / "seen_train.npy")
print("\n### inside H=256: NDCG / AUC by label-free splits")
rows = []
for nm, m in (("user seen in train", CAP & seen), ("user not in train", CAP & ~seen), ("distinct-item share of newest 64 < median", CAP & (C["h64_distinct"] < np.median(C["h64_distinct"][CAP]))),
              ("distinct-item share >= median", CAP & (C["h64_distinct"] >= np.median(C["h64_distinct"][CAP]))), ("history categories (all 256) <= 5", CAP & (C["hall_ncat"] <= 5)), ("history categories 6-10", CAP & (C["hall_ncat"] > 5) & (C["hall_ncat"] <= 10)),
              ("history categories > 10", CAP & (C["hall_ncat"] > 10)), ("NEAR: user seen in train", NEAR & seen), ("NEAR: user not in train", NEAR & ~seen)):
    rows.append([nm, int(m.sum()), "%.4f+-%.4f" % cse(e, m)[:2], "%.4f+-%.4f" % cse(au, m)[:2], f"{Mv[m].mean():.0f}", f"{NPOS[m].mean():.2f}"])
print(tab(rows, ["split", "feeds", "NDCG", "AUC", "mean M", "n_pos"]))
