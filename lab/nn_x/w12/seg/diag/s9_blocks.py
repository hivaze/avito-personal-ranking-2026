from lib import *
import lightgbm as lgb, warnings
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
cov = pl.read_parquet(OUT / "feed_cov.parquet"); e = np.load(OUT / "final_ndcg.npy"); seen = np.load(OUT / "seen_train.npy")
C = {c: cov[c].to_numpy().astype(np.float64) for c in cov.columns}; C["logM"] = np.log(Mv); C["seen"] = seen.astype(float); C["npos"] = NPOS.astype(float)
tr = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["user_id"]).group_by("user_id").len()
cnt = dict(zip(tr["user_id"].to_list(), tr["len"].to_list())); C["log_ntrain"] = np.log1p(np.array([0 if u == 0 else cnt.get(int(u), 0) for u in uid], np.float64))
F = ["M", "logM", "day", "hour", "uid0", "cold_share", "emb_miss_share", "n_ccat", "top_ccat_share"]; T = [f"h64_et{t}" for t in range(1, 11)]
R = ["ccat_in_hist_share", "h64_ncat", "h64_topcat", "h64_noemb", "h64_distinct", "h64_src_top"]; A = ["seen", "log_ntrain"]
SETS = {"F (feed only)": F, "F + T (event-type mix)": F + T, "F + T + R (= X1)": F + T + R, "X1 + A (user activity markers)": F + T + R + A, "F + A": F + A, "A only": A, "X1 + A + n_pos (NOT label-free)": F + T + R + A + ["npos"]}
gkf = list(GroupKFold(5).split(np.arange(NF), groups=clu))
def crossfit(y, names, ref):
    X = np.stack([C[n] for n in names], 1); pred = np.zeros(NF)
    for tr_, te in gkf:
        trr = tr_[ref[tr_]]
        pred[te] = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, n_jobs=12, verbose=-1, random_state=0).fit(X[trr], y[trr]).predict(X[te])
    return pred
for refname, ref in (("REST", REST), ("NEAR", NEAR)):
    d0, s0 = cdiff(e, CAP, ref); rows = [["raw", f"{d0:+.5f}", f"{s0:.5f}", "100%"]]
    for sn, names in SETS.items():
        r = e - crossfit(e, names, ref); da, sa = cdiff(r, CAP, ref); rows.append([sn, f"{da:+.5f}", f"{sa:.5f}", f"{100 * da / d0:.0f}%"])
    print(f"\n### NDCG, reference {refname}\n" + tab(rows, ["matching block", "D after matching", "SE", "share of raw left"]))
print("\nuser-activity markers: share seen in train H=256 %.3f, NEAR %.3f, REST %.3f; mean train feeds per user (uid != 0) H=256 %.2f, NEAR %.2f" % (seen[CAP].mean(), seen[NEAR].mean(), seen[REST].mean(), np.expm1(C["log_ntrain"][CAP]).mean(), np.expm1(C["log_ntrain"][NEAR]).mean()))
rows = []
for nm, m in (("H=256, not in train", CAP & ~seen), ("H=256, 1-3 train feeds", CAP & seen & (C["log_ntrain"] < np.log1p(3.5))), ("H=256, 4+ train feeds", CAP & (C["log_ntrain"] >= np.log1p(3.5))),
              ("NEAR, not in train", NEAR & ~seen), ("NEAR, 1-3 train feeds", NEAR & seen & (C["log_ntrain"] < np.log1p(3.5))), ("NEAR, 4+ train feeds", NEAR & (C["log_ntrain"] >= np.log1p(3.5)))):
    rows.append([nm, int(m.sum()), "%.4f+-%.4f" % cse(e, m)[:2], f"{NPOS[m].mean():.2f}", f"{Mv[m].mean():.0f}"])
print(tab(rows, ["cell", "feeds", "FINAL NDCG", "n_pos", "M"]))
