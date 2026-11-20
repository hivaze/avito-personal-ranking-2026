"""(5) Adversarial validation at the ensemble level: can val feeds be told apart from (private) test feeds using
label-free per-feed ensemble statistics, and do the ensemble-design conclusions survive importance weighting?

Per-feed features: M, H, user0, cold_share, hour-of-day (MSK), plus ensemble statistics:
  corr(F5_ft, F5_to), corr(F5 mix, F5cd mix), corr(F5 mix, V13 mix), member disagreement (mean over candidates of the
  std of the 6 train-only member z's), v11 top-1 margin (z), v11 top-5 z spread.
Classifier: LightGBM binary (shallow), 5-fold stratified CV AUC. Then val importance weights w = p/(1-p) (clipped
[0.2, 5], mean 1) from the val-vs-private classifier, applied to paired per-feed deltas of the main candidates.
"""
import json
import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import BOUND, ROOT, Split  # noqa: E402
from exp3_diversity import F5, F5CD, V13  # noqa: E402


def p(n):
    return np.load(f"{ROOT}/preds/{n}.npy").astype(np.float64)


def fam(S, split, ms, ft_suf):
    s_ft, s_to = ("val_oof", "trainonly_val") if split == "val" else ("test", "trainonly_test")
    zf = [S.z(p(f"nn_{m}_{ft_suf}_{s_ft}")) for m in ms]
    zt = [S.z(p(f"nn_{m}_{s_to}")) for m in ms]
    return zf, zt


def feed_stats(S, split):
    zf, zt = fam(S, split, F5, "n2cpf")
    Zf, Zt = S.z(np.mean(zf, 0)), S.z(np.mean(zt, 0))
    v11 = S.z(0.5 * Zf + 0.5 * Zt)
    cf, ct = fam(S, split, F5CD, "n2cpf")
    cd = S.z(0.5 * S.z(np.mean(cf, 0)) + 0.5 * S.z(np.mean(ct, 0)))
    hf, ht = fam(S, split, V13, "n2cpf")
    v13 = S.z(0.5 * S.z(np.mean(hf, 0)) + 0.5 * S.z(np.mean(ht, 0)))
    pc = lambda a, b: np.bincount(S.fid, a * b, S.n) / S.M  # noqa: E731
    dis = np.bincount(S.fid, np.std(zt, 0), S.n) / S.M
    # top-1 margin and top-5 spread of v11
    order = np.lexsort((-v11, S.fid))
    sv = v11[order]
    pos = np.arange(S.N) - np.repeat(S.off, S.M)
    top1 = sv[S.off]
    top2 = sv[S.off + 1]
    top5 = np.bincount(S.fid[order], np.where(pos < 5, sv, 0), S.n) / np.minimum(S.M, 5)
    hour = ((S.ts + 3 * 3600) % 86400) / 3600
    X = np.stack([S.M, S.H, S.user == 0, S.cold_share, hour, pc(Zf, Zt), pc(v11, cd), pc(v11, v13), dis, top1 - top2, top1 - top5], 1)
    names = ["M", "H", "user0", "cold_share", "hour", "corr_ft_to", "corr_v11_F5cd", "corr_v11_V13", "member_dis", "top1_margin", "top1_minus_top5"]
    return X.astype(np.float64), names, (Zf, Zt, cd, v13)


def cv_auc(X, y, cols, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    imp = np.zeros(len(cols))
    for tr, te in skf.split(X, y):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=16, verbose=-1)
        m.fit(X[tr][:, cols], y[tr])
        oof[te] = m.predict_proba(X[te][:, cols])[:, 1]
        imp += m.booster_.feature_importance("gain")
    return roc_auc_score(y, oof), oof, imp / imp.sum()


def main():
    Sv, St = Split("val"), Split("test")
    Xv, names, (Zf, Zt, cd, v13) = feed_stats(Sv, "val")
    Xt, _, _ = feed_stats(St, "test")
    priv = St.ts >= BOUND
    rep = {"names": names}
    print("feature means  val | public | private")
    for i, n in enumerate(names):
        print(f"  {n:16s} {Xv[:, i].mean():9.4f} | {Xt[~priv, i].mean():9.4f} | {Xt[priv, i].mean():9.4f}")
    rep["means"] = {n: [float(Xv[:, i].mean()), float(Xt[~priv, i].mean()), float(Xt[priv, i].mean())] for i, n in enumerate(names)}
    meta = [0, 1, 2, 3]
    ens = list(range(len(names)))
    ens_nohour = [i for i in ens if names[i] != "hour"]
    for tag, Xb, yb in [("val_vs_private", np.vstack([Xv, Xt[priv]]), np.r_[np.zeros(len(Xv)), np.ones(priv.sum())]),
                        ("val_vs_public", np.vstack([Xv, Xt[~priv]]), np.r_[np.zeros(len(Xv)), np.ones((~priv).sum())]),
                        ("public_vs_private", Xt, priv.astype(float))]:
        for ctag, cols in [("meta", meta), ("meta+ens(no hour)", ens_nohour), ("all", ens)]:
            auc, oof, imp = cv_auc(Xb, yb, cols)
            rep[f"{tag}:{ctag}"] = {"auc": float(auc), "imp": {names[c]: float(v) for c, v in zip(cols, imp)}}
            top = sorted(zip([names[c] for c in cols], imp), key=lambda t: -t[1])[:4]
            print(f"{tag:18s} {ctag:18s} AUC {auc:.4f}  top gain {[(a, round(b, 3)) for a, b in top]}")
            if tag == "val_vs_private" and ctag == "meta+ens(no hour)":
                pv = np.clip(oof[: len(Xv)], 1e-3, 1 - 1e-3)
                w = np.clip(pv / (1 - pv), 0.2, 5)
                w_priv = w / w.mean()
    # importance-weighted deltas of the main candidates
    S = Sv
    v11 = 0.5 * Zf + 0.5 * Zt
    base = S.per_feed(v11)
    cands = {
        "eq(F5,F5cd)": S.z(v11) + cd,
        "eq(F5,F5cd,V13)": S.z(v11) + cd + v13,
        "eq(F5,V13)": S.z(v11) + v13,
        "w_ft=0.67": 0.67 * Zf + 0.33 * Zt,
        "w_ft by M (0.33/0.67)": np.where((S.M >= 50)[S.fid], 0.67 * Zf + 0.33 * Zt, 0.33 * Zf + 0.67 * Zt),
        "train-only F5": Zt,
        "fine-tuned F5": Zf,
    }
    ess = w_priv.sum() ** 2 / (w_priv ** 2).sum()
    print(f"\nadversarial weights: ESS {ess:.0f} of {S.n}")
    rep["ess"] = float(ess)
    rep["weighted"] = {}
    for k, s in cands.items():
        d = S.per_feed(s) - base
        u = (d.mean(), d.std() / np.sqrt(S.n))
        x = w_priv * d
        wv = (x.mean(), x.std() / np.sqrt(S.n))
        rep["weighted"][k] = {"unweighted": [float(u[0]), float(u[1])], "adv_weighted": [float(wv[0]), float(wv[1])]}
        print(f"  {k:24s} unweighted {u[0]:+.5f}+-{u[1]:.5f}   adversarially weighted (private-like) {wv[0]:+.5f}+-{wv[1]:.5f}")
    np.save(f"{ROOT}/solvers/out/ensemble/adv_w_private_val.npy", w_priv.astype(np.float32))
    json.dump(rep, open(f"{ROOT}/solvers/out/ensemble/exp5.json", "w"), indent=1)


if __name__ == "__main__":
    main()
