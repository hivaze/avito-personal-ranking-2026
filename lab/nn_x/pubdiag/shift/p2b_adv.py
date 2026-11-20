"""Part 2: adversarial validation val vs PUBLIC (and vs PRIVATE) test feeds on label-free per-feed descriptors,
importance weights for val feeds (OOF LightGBM, 5 folds), weighted val deltas of the pools."""
from shiftlib import *  # noqa: F401,F403
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

dv = pl.read_parquet(OUT / "desc_val.parquet"); dt = pl.read_parquet(OUT / "desc_test.parquet")
pub = dt["timestamp"].to_numpy() < BOUND
R = dict(np.load(OUT / "p2_val_perfeed.npz"))
PUBLIC = {"PF7x": 0.392723, "G3": 0.392819, "M2": 0.392434, "Cd": 0.39208}

cols = dv.columns
INTR = [c for c in cols if c in ("M", "logM", "H", "uid0", "n_cat", "max_cat_share", "hist_n_cat", "hist_top_cat_share",
                                 "cand_cat_in_hist_share", "f1_atmin", "top_cat")
        or c.startswith(("cc", "f1_", "f2_", "f3_", "f4_", "f5_", "het", "hes", "hc")) and c != "ccnt_logmean"
        and c != "ccnt_zero_share"]
OVL = ["cold_share", "coldcand_share", "luo_cold_share", "luo_logmean", "luo_logmax", "ccnt_logmean", "ccnt_zero_share",
       "et1_logmean", "n_train_feeds"]
HOUR = ["minute_of_day"]
SETS = {
    "intrinsic": INTR,
    "intrinsic+hour": INTR + HOUR,
    "overlap": OVL + ["M", "H", "uid0"],
    "full": INTR + HOUR + OVL,
    "M+cold": ["M", "cold_share"],
}
PRM = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=200, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=12, verbose=-1, seed=7)
NR = 400


def adv(target_mask, feats, tag):
    Xv = dv.select(feats).to_numpy().astype(np.float32); Xt = dt.filter(pl.Series(target_mask)).select(feats).to_numpy().astype(np.float32)
    X = np.vstack([Xv, Xt]); yy = np.r_[np.zeros(len(Xv)), np.ones(len(Xt))]
    oof = np.zeros(len(X)); imp = np.zeros(len(feats))
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=11).split(X, yy)):
        m = lgb.train(PRM, lgb.Dataset(X[tr], yy[tr], feature_name=feats,
                                       categorical_feature=["top_cat"] if "top_cat" in feats else "auto"), NR)
        oof[te] = m.predict(X[te]); imp += m.feature_importance("gain")
    auc = roc_auc_score(yy, oof)
    p = np.clip(oof[:len(Xv)], 1e-4, 1 - 1e-4)
    w = p / (1 - p); w = np.minimum(w, np.quantile(w, 0.999)); w = w / w.mean()
    neff = w.sum() ** 2 / (w ** 2).sum()
    order = np.argsort(-imp)
    top = [(feats[i], imp[i] / imp.sum()) for i in order[:8]]
    return dict(auc=auc, w=w, neff=neff, top=top, wmax=w.max())


CMPS = [("PF7x", "Cd"), ("G3", "Cd"), ("M2", "Cd"), ("M2", "PF7x"), ("G3", "PF7x"), ("F7x", "F7b"), ("F7x", "F7")]
rows = []
W = {}
for target, tmask in [("public", pub), ("private", ~pub)]:
    print(f"\n######## val vs {target.upper()} ({int(tmask.sum())} test feeds) ########")
    for sname, feats in SETS.items():
        r = adv(tmask, feats, f"{target}_{sname}")
        W[f"{target}_{sname}"] = r["w"]
        lvl = np.average(R["Cd"], weights=r["w"])
        print(f"[{sname}] AUC {r['auc']:.4f}  n_eff {r['neff']:.0f}  w_max {r['wmax']:.1f}  Cd level {lvl:.5f} "
              f"(val {R['Cd'].mean():.5f}{', public 0.39208' if target == 'public' else ''})")
        print("   top gain: " + ", ".join(f"{a} {b:.2f}" for a, b in r["top"]))
        line = []
        for a, b in CMPS:
            m, se = mse(R[a] - R[b], r["w"])
            rows.append(dict(target=target, set=sname, cmp=f"{a}-{b}", d=m, se=se, auc=r["auc"], neff=r["neff"]))
            line.append(f"{a}-{b} {m:+.5f}±{se:.5f}")
        print("   " + " | ".join(line))
pl.DataFrame(rows).write_csv(OUT / "p2_adv_weighted.csv")
np.savez(OUT / "p2_adv_weights.npz", **W)

print("\nunweighted val: " + " | ".join(f"{a}-{b} {fmt(*mse(R[a] - R[b]))}" for a, b in CMPS))
print("public LB (context only): " + " | ".join(f"{a}-Cd {PUBLIC[a] - PUBLIC['Cd']:+.5f}" for a in ("PF7x", "G3", "M2")))
