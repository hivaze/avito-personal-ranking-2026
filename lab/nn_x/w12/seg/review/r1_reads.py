"""R1 (three decision numbers of part A, own code) and R2 (the same reads on the held-out twin and on hard-label pools)."""
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from rv_lib import *

C = np.load(OUT / "cos.npz")
ER = e_rand()
W = boot_weights(2000)
R = {}
R["FINAL (FD16P20)"] = np.load(OUT / "final_score.npy").astype(np.float64)
R["W16P20w (held-out twin)"] = final_from(W16, P1W, 0.2)
R["W16 (held-out, no pop-light)"] = final_from(W16, [], 0.0)
R["TEACHER 11 hard-label (TO)"] = pool(T11, "trainonly_val")
R["G3 file (hard labels, FT+TO)"] = 0.5 * z(np.load(P / "ens_G3_ft_val.npy")) + 0.5 * z(np.load(P / "ens_G3_to_val.npy"))
R["single hard-label w5_e10_base_s951 (TO)"] = z(np.load(P / "nn_w5_e10_base_s951_trainonly_val.npy"))
R["single student w5_fd10_kd05w192_s951 (TO)"] = z(np.load(P / "nn_w5_fd10_kd05w192_s951_trainonly_val.npy"))
R["pre2_fe"] = feat("pre2_fe")
R["max cosine, all stored history"] = C["smax_all"].astype(np.float64)
R["max cosine, newest 64"] = C["smax_64"].astype(np.float64)
R["max cosine, newest 16"] = C["smax_16"].astype(np.float64)
R["max cosine, newest 128"] = C["smax_n128"].astype(np.float64)
R["mean of top-16 cosines"] = C["top16"].astype(np.float64)
R["mean of top-32 cosines"] = C["top32"].astype(np.float64)
R["popularity only (it_hist_all_luo)"] = feat("it_hist_all_luo")
R["exposure only (xp_imp_c50)"] = feat("xp_imp_c50")
x = np.load(FLAT / "val/cand_f1.npy").astype(np.float64)
R["f1 column"] = x if np.nanmean(auc(x)) >= 0.5 else -x
ND = {k: ndcg(v) for k, v in R.items()}
AU = {k: auc(v) for k, v in R.items()}
SK = {k: (v - ER) / (1 - ER) for k, v in ND.items()}
np.savez(OUT / "nd.npz", **{k.split(" (")[0].replace(" ", "_").replace(",", ""): v for k, v in ND.items()})


def ratio(x, a, b, w):
    """bootstrap draws of mean(x|a)/mean(x|b)"""
    sa, na = np.bincount(CLU, weights=x * a, minlength=NCL), np.bincount(CLU, weights=a.astype(float), minlength=NCL)
    sb, nb = np.bincount(CLU, weights=x * b, minlength=NCL), np.bincount(CLU, weights=b.astype(float), minlength=NCL)
    return (sa.sum() / na.sum()) / (sb.sum() / nb.sum()), (w @ sa) / (w @ na) / ((w @ sb) / (w @ nb))


print("### R1(a) / R2: gap and skill ratio by ranker (own code). Paired contrast = [NDCG_X - NDCG_FINAL](cap) - [..](NEAR)")
rows = []
rF, bF = ratio(SK["FINAL (FD16P20)"], CAP, NEAR, W)
for k in R:
    d1, s1 = cdiff(ND[k], CAP, NEAR)
    d2, s2 = cdiff(ND[k], CAP, REST)
    r, b = ratio(SK[k], CAP, NEAR, W)
    pc, ps = cdiff(ND[k] - ND["FINAL (FD16P20)"], CAP, NEAR)
    a1, a2 = np.nanmean(AU[k][NEAR]), np.nanmean(AU[k][CAP])
    rows.append([k, f"{ND[k].mean():.5f}", f"{ND[k][NEAR].mean():.4f}", f"{ND[k][CAP].mean():.4f}", f"{d1:+.4f}+-{s1:.4f}", f"{d2:+.4f}+-{s2:.4f}", f"{a1:.4f}", f"{a2:.4f}",
                 f"{r:.3f} ({b.std():.3f})", "" if k.startswith("FINAL") else f"{rF - r:+.3f}+-{(bF - b).std():.3f} (z {(rF - r) / (bF - b).std():+.2f})",
                 "" if k.startswith("FINAL") else f"{pc:+.4f}+-{ps:.4f} (z {pc / ps:+.2f})", f"{(a2 - 0.5) / (a1 - 0.5):.3f}"])
print(tab(rows, ["ranker", "ALL", "NEAR", "cap", "D(cap-NEAR)", "D(cap-REST)", "AUC NEAR", "AUC cap", "skill ratio (boot SE)", "R_FINAL - R_X", "paired contrast vs FINAL", "AUC-edge ratio"]))

print("\n### R1(c1/c2): cosine rankers at the cap (paired per feed)")
for a_, b_, met in (("max cosine, newest 64", "max cosine, all stored history", "NDCG"), ("max cosine, newest 16", "max cosine, all stored history", "NDCG"),
                    ("max cosine, newest 128", "max cosine, all stored history", "NDCG"), ("max cosine, newest 64", "max cosine, all stored history", "AUC"),
                    ("mean of top-32 cosines", "mean of top-16 cosines", "AUC"), ("mean of top-32 cosines", "mean of top-16 cosines", "NDCG")):
    D = (ND if met == "NDCG" else AU)
    dl = D[a_] - D[b_]
    for nm, m in (("cap", CAP), ("NEAR", NEAR), ("H 20-99", (H >= 20) & (H <= 99))):
        mu, se, n = cmean(dl, m)
        print(f"  {a_} minus {b_} [{met}] @ {nm}: {mu:+.4f} +- {se:.4f} (n {n})")

print("\n### R1(c3): stored history-drop predictions (HDROP-ALL), cost of one 20% / 10% pass, 2 hard-label members")
HD = ROOT / "nn_x/tricks/HDROP-ALL/preds/val"
for rate in ("d20", "d10"):
    cost = np.zeros(NF)
    for v in ("w5_e10_base_s951", "w5_e10_base_s952"):
        ref = ndcg(np.load(HD / f"{v}~ref.npy"))
        cost += np.mean([ndcg(np.load(HD / f"{v}~{rate}_k{k}.npy")) - ref for k in range(4)], 0) / 2
    for nm, m in (("cap", CAP), ("NEAR", NEAR), ("H 20-99", (H >= 20) & (H <= 99))):
        mu, se, n = cmean(cost, m)
        print(f"  {rate} @ {nm}: {mu:+.5f} +- {se:.5f}")
    d, se = cdiff(cost, CAP, NEAR)
    print(f"  {rate} contrast cap - NEAR: {d:+.5f} +- {se:.5f}")

print("\n### R1(b): odds-weighted residual (propensity cap vs reference, LightGBM, 5 folds by user, weights clipped at p99)")
cv = pl.read_parquet(OUT / "cov.parquet")
XA = [c for c in cv.columns if c not in ("seen", "log_ntr", "hall_distinct")]
XB = XA + ["seen", "log_ntr"]
SETS = {"M only": ["M", "logM"], "feed only": ["M", "logM", "day", "hour", "uid0", "cold_share", "n_ccat", "top_ccat", "emb_miss"],
        "XA (newest-64 composition, no activity)": XA, "XB (XA + seen in train, n train feeds)": XB}
PW = {}
for refn, ref in (("NEAR", NEAR), ("REST", REST)):
    sel = np.where(CAP | ref)[0]
    for sn, cols in SETS.items():
        X = cv.select(cols).to_numpy()[sel]
        t = CAP[sel].astype(int)
        p = np.zeros(len(sel))
        for tr_i, te_i in GroupKFold(5).split(X, t, CLU[sel]):
            mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                     reg_lambda=5.0, n_jobs=12, verbose=-1, random_state=0)
            mdl.fit(X[tr_i], t[tr_i])
            p[te_i] = mdl.predict_proba(X[te_i])[:, 1]
        from sklearn.metrics import roc_auc_score
        w = np.zeros(NF)
        odds = p / np.clip(1 - p, 1e-3, None)
        wr = odds[t == 0]
        wr = np.minimum(wr, np.quantile(wr, 0.99))
        w[sel[t == 0]] = wr
        w[sel[t == 1]] = 1.0
        PW[(refn, sn)] = w
        ess = wr.sum() ** 2 / (wr ** 2).sum()
        line = f"  ref {refn:4s} | {sn:42s} | propensity AUC {roc_auc_score(t, p):.3f} | ESS of the reference {ess:.0f} of {len(wr)}"
        for k in ("FINAL (FD16P20)", "W16P20w (held-out twin)", "TEACHER 11 hard-label (TO)", "G3 file (hard labels, FT+TO)"):
            d, se = cdiff(ND[k], CAP, ref, w)
            d0, _ = cdiff(ND[k], CAP, ref)
            line += f" | {k.split(' (')[0]}: {d:+.4f}+-{se:.4f} ({100 * d / d0:.0f}% left)"
        print(line)
np.savez(OUT / "pw.npz", **{f"{a}|{b}": v for (a, b), v in PW.items()})

print("\n### balance after weighting (XB, ref NEAR): standardised mean differences cap vs weighted NEAR for the largest raw gaps")
w = PW[("NEAR", "XB (XA + seen in train, n train feeds)")]
rows = []
for c in XB + ["hall_distinct"]:
    xx = cv[c].to_numpy().astype(np.float64)
    sd = xx[CAP | NEAR].std() + 1e-9
    raw = (xx[CAP].mean() - xx[NEAR].mean()) / sd
    adj = (xx[CAP].mean() - (w[NEAR] * xx[NEAR]).sum() / w[NEAR].sum()) / sd
    rows.append([c, f"{xx[CAP].mean():.3f}", f"{xx[NEAR].mean():.3f}", f"{raw:+.2f}", f"{adj:+.2f}"])
rows.sort(key=lambda r: -abs(float(r[3])))
print(tab(rows[:14], ["covariate", "cap", "NEAR", "SMD raw", "SMD weighted"]))

print("\n### regression cross-check (LightGBM of per-feed NDCG on XB fitted on NEAR feeds, 5 folds by user; residual cap minus residual NEAR)")
for k in ("FINAL (FD16P20)", "W16P20w (held-out twin)", "TEACHER 11 hard-label (TO)"):
    for sn in ("XA (newest-64 composition, no activity)", "XB (XA + seen in train, n train feeds)"):
        X = cv.select(SETS[sn]).to_numpy()
        pred = np.zeros(NF)
        for tr_i, te_i in GroupKFold(5).split(X, None, CLU):
            fit = tr_i[NEAR[tr_i]]
            mdl = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0,
                                    n_jobs=12, verbose=-1, random_state=0)
            mdl.fit(X[fit], ND[k][fit])
            pred[te_i] = mdl.predict(X[te_i])
        d, se = cdiff(ND[k] - pred, CAP, NEAR)
        print(f"  {k.split(' (')[0]:10s} | {sn:42s} | residual D(cap - NEAR) {d:+.4f} +- {se:.4f}")
