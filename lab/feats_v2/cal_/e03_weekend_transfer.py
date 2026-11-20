"""Train-based weekend test (val has no weekend days).

Data: train feeds that look randomly ordered (p_random > 0.5, M >= 20; labels not driven by display order), days >= Sep 8
(every cal_ label statistic for a train row uses strictly earlier days only).
Experiments (LightGBM lambdarank, harness params, fixed rounds), each run with BASE vs BASE + cal_ group:
  A  train weekday (Sep 8-11,14,17,18; users in half U1)   -> eval weekend (Sep 12,13,19; users in U2)
  B  train weekend (Sep 12,13,19; U1)                      -> eval weekday (Sep 14,17,18; U2)
  C  control: train weekday (U1)                           -> eval weekday (U2, same days)
  D  mixed: train weekday+weekend (U1)                     -> eval weekend (U2) and weekday (U2)
Users are split by hash so no user is in both train and eval. Reports paired per-feed delta (cand - base) and SE.
"""
from calcommon import *
import lightgbm as lgb, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--groups", default="lab,dem,all")
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--ntrain", type=int, default=90000)
ap.add_argument("--neval", type=int, default=50000)
ap.add_argument("--seeds", type=int, default=2)
args = ap.parse_args()

PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1000], learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              lambdarank_truncation_level=1000, verbose=-1, num_threads=args.threads)
ROUNDS = 350
BASE = json.load(open(f"{ROOT}/data/eval_cache/base_features.json"))
GROUPS = {
    "lab": ["cal_lab_cat_ctx", "cal_lab_cat_we", "cal_lab_cat_hb", "cal_lab_emb_ctx"],
    "dem": ["cal_dem_cat_ctx", "cal_dem_emb_ctx"],
    "feed": ["cal_weekend", "cal_mod"],
}
GROUPS["all"] = GROUPS["lab"] + GROUPS["dem"] + ["cal_sup_cat_ctx", "cal_ds_cat_ctx", "cal_drift_cat_day"] + GROUPS["feed"]
GROUPS["keep"] = ["cal_lab_cat_ctx", "cal_lab_cat_we", "cal_lab_emb_ctx", "cal_dem_cat_ctx", "cal_dem_emb_ctx"]

t0 = time.time()
F = feeds("train"); M = F["M"].to_numpy().astype(np.int64); off = F["cand_off"].to_numpy().astype(np.int64)
tp = time_parts(F["timestamp"].to_numpy())
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"]
dstr = np.array([str(np.datetime64(int(d), "D")) for d in np.unique(tp["day"])])
day_s = dstr[np.searchsorted(np.unique(tp["day"]), tp["day"])]
uid = F["user_id"].to_numpy().astype(np.int64)
uhalf = np.where(uid > 0, (uid * 2654435761 % 2**32) % 2, np.arange(len(uid)) % 2)
ok = (pr > 0.5) & (M >= 20)
WD_TR = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14", "2026-09-17", "2026-09-18"]
WE = ["2026-09-12", "2026-09-13", "2026-09-19"]
WD_EV = ["2026-09-14", "2026-09-17", "2026-09-18"]
rng = np.random.default_rng(0)


def pick(days, half, n):
    s = np.where(ok & np.isin(day_s, days) & (uhalf == half))[0]
    s = s[rng.permutation(len(s))[:n]]
    return np.sort(s)


sets = {
    "tr_wd": pick(WD_TR, 0, args.ntrain), "tr_we": pick(WE, 0, args.ntrain),
    "ev_we": pick(WE, 1, args.neval), "ev_wd": pick(WD_EV, 1, args.neval),
}
sets["tr_mix"] = np.sort(np.r_[sets["tr_wd"][: args.ntrain // 2], sets["tr_we"][: args.ntrain // 2]])
for k, v in sets.items():
    print(k, len(v), "rows", int(M[v].sum()))
allf = sorted(set(BASE) | set(sum([GROUPS[g] for g in args.groups.split(",")], [])))
need = np.unique(np.concatenate(list(sets.values())))
pos_in_need = {k: np.searchsorted(need, v) for k, v in sets.items()}
Mn = M[need]; offn = np.r_[0, np.cumsum(Mn)[:-1]]
rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in need])
log("gathering", len(allf), "features for", len(rows), "rows")
X = np.empty((len(rows), len(allf)), np.float32)
for j, n in enumerate(allf):
    if n == "category_id":
        X[:, j] = np.asarray(cand("train", "category_id")[rows], np.float32)
    else:
        X[:, j] = np.load(f"{FEATS}/train/{n}.npy", mmap_mode="r")[rows]
y = np.asarray(cand("train", "target")[rows], np.float32)
log("gathered", time.time() - t0)
col = {n: j for j, n in enumerate(allf)}


def rows_of(fsel):
    return np.concatenate([np.arange(offn[i], offn[i] + Mn[i]) for i in fsel])


R = {k: rows_of(v) for k, v in pos_in_need.items()}


def fit_pred(tr, evs, names, seed):
    cols = [col[n] for n in names]
    cat = ["category_id"] if "category_id" in names else []
    d = lgb.Dataset(X[R[tr]][:, cols], y[R[tr]], group=Mn[pos_in_need[tr]], feature_name=names, categorical_feature=cat)
    b = lgb.train(dict(PARAMS, seed=seed), d, ROUNDS)
    return {ev: b.predict(X[R[ev]][:, cols]) for ev in evs}


def ndcg_set(p, ev):
    Me = Mn[pos_in_need[ev]]
    return ndcg_pf(p, y[R[ev]].astype(np.float64), np.r_[0, np.cumsum(Me)[:-1]], Me, mode=0)


EXPS = [("A_wd->we", "tr_wd", "ev_we"), ("B_we->wd", "tr_we", "ev_wd"), ("C_wd->wd", "tr_wd", "ev_wd"),
        ("D_mix->we", "tr_mix", "ev_we"), ("D_mix->wd", "tr_mix", "ev_wd")]
TR_EVS = {"tr_wd": ["ev_we", "ev_wd"], "tr_we": ["ev_wd"], "tr_mix": ["ev_we", "ev_wd"]}
results = []
for seed in range(args.seeds):
    for tr, evs in TR_EVS.items():
        pb = fit_pred(tr, evs, BASE, seed)
        nbs = {ev: ndcg_set(pb[ev], ev) for ev in evs}
        for g in args.groups.split(","):
            names = BASE + [n for n in GROUPS[g] if n not in BASE]
            pc = fit_pred(tr, evs, names, seed)
            for ev in evs:
                ename = [e for e, a, b in EXPS if a == tr and b == ev][0]
                nb_ = nbs[ev]; nc = ndcg_set(pc[ev], ev); d = nc - nb_
                results.append(dict(seed=seed, exp=ename, group=g, base=float(nb_.mean()), cand=float(nc.mean()),
                                    delta=float(d.mean()), se=float(d.std() / np.sqrt(len(d))), dvec=d))
                log(ename, g, f"seed{seed} base={nb_.mean():.4f} cand={nc.mean():.4f} delta={d.mean():+.5f} se={d.std() / np.sqrt(len(d)):.5f}")
# aggregate over seeds (paired per-feed mean over seeds)
agg = []
for ename, tr, ev in EXPS:
    for g in args.groups.split(","):
        rs = [r for r in results if r["exp"] == ename and r["group"] == g]
        dv = np.mean([r["dvec"] for r in rs], 0)
        agg.append(dict(exp=ename, group=g, base=float(np.mean([r["base"] for r in rs])), cand=float(np.mean([r["cand"] for r in rs])),
                        delta=float(dv.mean()), se=float(dv.std() / np.sqrt(len(dv))), n_eval_feeds=len(dv)))
df = pl.DataFrame(agg)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
print(df)
df.write_csv(f"{OUT}/e03_weekend_transfer_{args.groups.replace(',', '_')}.csv")
log("done", time.time() - t0)
