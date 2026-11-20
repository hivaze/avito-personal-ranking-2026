"""Feed-level adversarial validation: AUC with user-grouped CV for several split pairs and feature sets.
Writes OUT/feed_adv.json (AUCs, importances, PSI) and OUT/feed_oof_<comp>_<set>.npz (OOF probabilities)."""
import json, time
from common_av import *
from adv_lib import *

t0 = time.time()
def log(*a): print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

v = pl.read_parquet(f"{OUT}/feed_val.parquet")
t = pl.read_parquet(f"{OUT}/feed_test.parquet")
r = pl.read_parquet(f"{OUT}/feed_train_rl.parquet")
pub, pri = t.filter(pl.col("ts") < 0), t.filter(pl.col("ts") >= 0)
day = lambda d: ((d["ts"] * 86400 + BOUND + MSK) // 86400)
v_tue = v.filter(pl.col("weekday") == 1); v_thu = v.filter(pl.col("weekday") == 3)

TIME = ["ts", "hour", "weekday"]
COVERAGE = ["cold_item_share", "hist10_cold_share", "hist_cold_share", "it_hist_all_luo_zero_share",
            "it_hist_all_luo_mlog", "seq_cv_invd20_zero_share", "seq_cv_invd20_mlog", "seq_cv_pop_zero_share",
            "seq_cv_pop_mlog", "prew_fe_has_share", "prew_fe_z_mean", "days_since_prior", "it_tr_ctr_mean",
            "it_tr_coec_mean", "it_cand_cnt_train_mean", "pre_idmax32_mean", "pre2_fe_has_share"]
DROP_TRAIN = ["pre2_fe_has_share", "pre2_fe_s_mean"]  # train NaN on Sep 6-8 only
# per-feed means of within-feed z-scores are 0 up to float32 rounding (|mean| ~2e-7); their rounding noise is a
# split fingerprint (AUC 0.997 val vs private on these 5 columns alone, diag1.py), so they are excluded
ZM = [c for c in v.columns if c.endswith("_z_mean")]
COMMON = [c for c in v.columns if c not in ("fidx", "user_id") and c in r.columns and c not in ZM]
VT_ONLY = [c for c in v.columns if c not in ("fidx", "user_id") and c not in r.columns and c not in ZM]

COMPS = {
    "a_val_vs_private": (v, pri, False),
    "b_val_vs_public": (v, pub, False),
    "c_public_vs_private": (pub, pri, False),
    "d_trainrl_vs_val50": (r, v.filter(pl.col("M") >= 50), True),
    "e_trainrl_vs_private50": (r, pri.filter(pl.col("M") >= 50), True),
    "f_valtue_vs_valthu": (v_tue, v_thu, False),
    "g_val50_vs_private50": (v.filter(pl.col("M") >= 50), pri.filter(pl.col("M") >= 50), False),
}

res_all = {}
for comp, (A, B, is_train) in COMPS.items():
    allf = [c for c in (COMMON if is_train else COMMON + VT_ONLY) if not (is_train and c in DROP_TRAIN)]
    sets = {"all": allf, "notime": [c for c in allf if c not in TIME],
            "intrinsic": [c for c in allf if c not in TIME and c not in COVERAGE]}
    D = pl.concat([A.select(["fidx", "user_id"] + allf).with_columns(pl.lit(0).alias("_y"), pl.lit(0).alias("_src")),
                   B.select(["fidx", "user_id"] + allf).with_columns(pl.lit(1).alias("_y"), pl.lit(1).alias("_src"))],
                  how="vertical_relaxed")
    y = D["_y"].to_numpy()
    feed_key = np.arange(len(D))  # rows are unique feeds
    g = groups_of(D["user_id"].to_numpy(), feed_key)
    res_all[comp] = {"nA": int((y == 0).sum()), "nB": int((y == 1).sum())}
    for sname, fl in sets.items():
        X = D.select(fl).to_numpy().astype(np.float32)
        res = run_adv(X, y, g, fl, P_FEED, 400)
        uni = univariate(X, y, fl)
        res_all[comp][sname] = dict(auc=res["auc"], fold_auc=res["fold_auc"].tolist(),
                                    imp={n: dict(shap=float(res["shap"][j]), gain=float(res["gain"][j]), **uni[n])
                                         for j, n in enumerate(fl)},
                                    table=top_table(res, uni, 15))
        np.savez(f"{OUT}/feed_oof_{comp}_{sname}.npz", oof=res["oof"], y=y, fidx=D["fidx"].to_numpy())
        log(comp, sname, f"AUC {res['auc']:.4f} folds {np.round(res['fold_auc'], 4)}  nA {res_all[comp]['nA']} nB {res_all[comp]['nB']}")
json.dump(res_all, open(f"{OUT}/feed_adv.json", "w"), indent=1, default=float)
for comp in COMPS:
    print(f"\n## {comp}  (A n={res_all[comp]['nA']}, B n={res_all[comp]['nB']})")
    for s in ["all", "notime", "intrinsic"]:
        print(f"{s}: AUC {res_all[comp][s]['auc']:.4f}")
    print("notime"); print(res_all[comp]["notime"]["table"])
    print("intrinsic"); print(res_all[comp]["intrinsic"]["table"])
