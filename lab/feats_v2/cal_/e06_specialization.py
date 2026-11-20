"""Does the weekend change the label-feature relationship? Size-matched specialization test (difference in differences).

Train BASE-feature LightGBM on N random-like weekday feeds (users U1) and on N random-like weekend feeds (U1), evaluate
both on held-out weekend and weekday feeds (users U2). If the relation differs, each model wins on its own day type:
DiD = [ndcg(we_model, ev_we) - ndcg(wd_model, ev_we)] - [ndcg(we_model, ev_wd) - ndcg(wd_model, ev_wd)] > 0.
Same feeds / days / user split as e03.
"""
from calcommon import *
import lightgbm as lgb, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--n", type=int, default=29000)
ap.add_argument("--neval", type=int, default=40000)
ap.add_argument("--seeds", type=int, default=2)
args = ap.parse_args()
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1000], learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              lambdarank_truncation_level=1000, verbose=-1, num_threads=args.threads)
BASE = json.load(open(f"{ROOT}/data/eval_cache/base_features.json"))
t0 = time.time()
F = feeds("train"); M = F["M"].to_numpy().astype(np.int64); off = F["cand_off"].to_numpy().astype(np.int64)
tp = time_parts(F["timestamp"].to_numpy())
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"]
ud = np.unique(tp["day"]); dstr = np.array([str(np.datetime64(int(d), "D")) for d in ud]); day_s = dstr[np.searchsorted(ud, tp["day"])]
uid = F["user_id"].to_numpy().astype(np.int64)
uhalf = np.where(uid > 0, (uid * 2654435761 % 2**32) % 2, np.arange(len(uid)) % 2)
ok = (pr > 0.5) & (M >= 20)
WD_TR = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14", "2026-09-17", "2026-09-18"]
WE = ["2026-09-12", "2026-09-13", "2026-09-19"]
WD_EV = ["2026-09-14", "2026-09-17", "2026-09-18"]
rng = np.random.default_rng(1)


def pick(days, half, n):
    s = np.where(ok & np.isin(day_s, days) & (uhalf == half))[0]
    return np.sort(s[rng.permutation(len(s))[:n]])


sets = {"tr_wd": pick(WD_TR, 0, args.n), "tr_we": pick(WE, 0, args.n), "ev_we": pick(WE, 1, args.neval), "ev_wd": pick(WD_EV, 1, args.neval)}
for k, v in sets.items():
    print(k, len(v), int(M[v].sum()), flush=True)


def gather(fs):
    rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in fs])
    X = np.empty((len(rows), len(BASE)), np.float32)
    for j, n in enumerate(BASE):
        X[:, j] = (np.asarray(cand("train", "category_id")[rows], np.float32) if n == "category_id"
                   else np.load(f"{FEATS}/train/{n}.npy", mmap_mode="r")[rows])
    return X, np.asarray(cand("train", "target")[rows], np.float64), M[fs]


D = {k: gather(v) for k, v in sets.items()}
log("gathered", time.time() - t0)
res = {}
for seed in range(args.seeds):
    for tr in ["tr_wd", "tr_we"]:
        X, y, Mg = D[tr]
        b = lgb.train(dict(PARAMS, seed=seed), lgb.Dataset(X, y, group=Mg, feature_name=BASE, categorical_feature=["category_id"]), 350)
        for ev in ["ev_we", "ev_wd"]:
            Xe, ye, Me = D[ev]
            nd = ndcg_pf(b.predict(Xe), ye, np.r_[0, np.cumsum(Me)[:-1]], Me, mode=0)
            res.setdefault((tr, ev), []).append(nd)
            log(f"seed{seed} {tr}->{ev}: {nd.mean():.4f}")
avg = {k: np.mean(v, 0) for k, v in res.items()}
d_we = avg[("tr_we", "ev_we")] - avg[("tr_wd", "ev_we")]
d_wd = avg[("tr_we", "ev_wd")] - avg[("tr_wd", "ev_wd")]
se = lambda d: d.std() / np.sqrt(len(d))
out = {f"{k[0]}->{k[1]}": float(v.mean()) for k, v in avg.items()}
out.update(we_model_minus_wd_model_on_weekend=float(d_we.mean()), se_on_weekend=float(se(d_we)),
           we_model_minus_wd_model_on_weekday=float(d_wd.mean()), se_on_weekday=float(se(d_wd)),
           DiD=float(d_we.mean() - d_wd.mean()), DiD_se=float(np.sqrt(se(d_we) ** 2 + se(d_wd) ** 2)))
print(json.dumps(out, indent=1))
json.dump(out, open(f"{OUT}/e06_specialization.json", "w"), indent=1)
