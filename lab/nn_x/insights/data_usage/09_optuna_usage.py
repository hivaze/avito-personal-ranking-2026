"""Optuna search over TRAIN-DATA-USAGE knobs on the CPU GBDT proxy (07_gbdt_arms.py machinery), pre-declared protocol:

- Val split by user hash: TUNE = user_id % 2 == 0 (user 0: by feed index parity), CONFIRM = the rest. The objective is
  the TUNE-half NDCG at a fixed 400 rounds (seed 0). CONFIRM is never read during the search.
- Knobs: alpha_exam (fixed RandTopN examination offset on production rows) in [0, 1.5]; alpha_trunc (end-of-feed offset)
  in [0, 1.5]; ips_clip in {1 (off), 1.5, 2, 3}; w_pool (random-like feed weight) log-uniform [0.5, 4]; drop_hi (share of
  production feeds with the highest RHO excess removed) in [0, 0.3].
- 36 TPE trials (3 worker processes x 12 trials, 12 threads each), seed 0 always (same bagging as the base arm).
- Confirmation (09 --confirm): the best TUNE trial and the base arm re-trained with seeds 1 and 2; paired per-feed
  CONFIRM delta averaged over the two seeds; accept iff delta > 2 SE AND holdout delta >= 0.

usage: python 09_optuna_usage.py --worker K   |   python 09_optuna_usage.py --confirm
"""
import argparse
import json
import sys
import numpy as np
import lightgbm as lgb
import optuna
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import importlib
A = importlib.import_module("07_gbdt_arms")
from metrics import ndcg_per_feed

G = A.G
DB = f"sqlite:///{G}/optuna_usage.db"
STUDY = "usage_v1"


def splits():
    _, mv = A.load("val")
    u = mv["user_id"].astype(np.int64)
    idx = np.arange(len(u))
    tune = np.where(u == 0, idx % 2 == 0, u % 2 == 0)
    return tune


def arm_of(p):
    toks = ["base"]
    if p["alpha_exam"] > 0.02:
        toks.append(f"exam{p['alpha_exam']:.3f}")
    if p["alpha_trunc"] > 0.02:
        toks.append(f"trunc{p['alpha_trunc']:.3f}")
    if p["ips_clip"] > 1:
        toks.append(f"ips{p['ips_clip']}")
    if abs(p["w_pool"] - 1) > 0.02:
        toks.append(f"pool{p['w_pool']:.3f}")
    if p["drop_hi"] > 0.005:
        toks.append(f"rhoinv{100 * (1 - p['drop_hi']):.2f}")
    return "+".join(toks)


def fit_eval(arm, seed, rounds=400, threads=12):
    Xv, mv = A.load("val"); Xh, mh = A.load("HO")
    Mv = mv["M"].astype(np.int64); offv = np.concatenate([[0], np.cumsum(Mv)[:-1]])
    Mh = mh["M"].astype(np.int64); offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
    Xt, yt, gt, it, wt, pt, addinf = A.build(arm, seed)
    assert not addinf, "09 does not use dom/posfeat"
    params = dict(objective="lambdarank", lambdarank_truncation_level=1000, label_gain=[0, 1], learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=10.0, num_threads=threads, verbose=-1, seed=seed, max_bin=255)
    ds = lgb.Dataset(Xt, yt, group=gt, init_score=it if np.any(it != 0) else None, weight=wt if np.any(wt != 1) else None,
                     position=pt, free_raw_data=True)
    b = lgb.train(params, ds, rounds)
    nv = ndcg_per_feed(b.predict(np.asarray(Xv), num_threads=threads), mv["y"], offv, Mv)
    nh = ndcg_per_feed(b.predict(np.asarray(Xh), num_threads=threads), mh["y"], offh, Mh)
    return nv, nh


def worker(k, n_trials):
    tune = splits()
    st = optuna.create_study(study_name=STUDY, storage=DB, direction="maximize", load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=100 + k, n_startup_trials=8))

    def obj(tr):
        p = dict(alpha_exam=tr.suggest_float("alpha_exam", 0.0, 1.5), alpha_trunc=tr.suggest_float("alpha_trunc", 0.0, 1.5),
                 ips_clip=tr.suggest_categorical("ips_clip", [1, 1.5, 2, 3]), w_pool=tr.suggest_float("w_pool", 0.5, 4.0, log=True),
                 drop_hi=tr.suggest_float("drop_hi", 0.0, 0.3))
        arm = arm_of(p)
        nv, nh = fit_eval(arm, 0)
        tr.set_user_attr("arm", arm); tr.set_user_attr("ho", float(nh.mean())); tr.set_user_attr("val_all", float(nv.mean()))
        tr.set_user_attr("confirm", float(nv[~tune].mean()))  # stored, not used for selection
        np.save(G / f"optuna_trial{tr.number}_val.npy", nv.astype(np.float32))
        return float(nv[tune].mean())

    st.optimize(obj, n_trials=n_trials)


def confirm():
    tune = splits()
    st = optuna.load_study(study_name=STUDY, storage=DB)
    best = st.best_trial
    arm = best.user_attrs["arm"]
    print("best TUNE trial", best.number, best.value, arm, best.params, flush=True)
    rows = {}
    for nm, a in [("base", "base"), ("best", arm)]:
        vs, hs = [], []
        for s in (1, 2):
            nv, nh = fit_eval(a, s)
            vs.append(nv); hs.append(nh)
        rows[nm] = (np.mean(vs, 0), np.mean(hs, 0))
    d = rows["best"][0] - rows["base"][0]; dh = rows["best"][1] - rows["base"][1]
    c = ~tune
    res = dict(best_trial=best.number, arm=arm, params=best.params, tune_value=best.value,
               confirm_delta=float(d[c].mean()), confirm_se=float(d[c].std() / np.sqrt(c.sum())),
               tune_delta=float(d[tune].mean()), tune_se=float(d[tune].std() / np.sqrt(tune.sum())),
               val_delta=float(d.mean()), val_se=float(d.std() / np.sqrt(len(d))),
               ho_delta=float(dh.mean()), ho_se=float(dh.std() / np.sqrt(len(dh))))
    res["accept"] = bool(res["confirm_delta"] > 2 * res["confirm_se"] and res["ho_delta"] >= 0)
    print(json.dumps(res, indent=1))
    json.dump(res, open(G / "optuna_confirm.json", "w"), indent=1)
    df = st.trials_dataframe()
    df.to_csv(G / "optuna_trials.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--n_trials", type=int, default=12)
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if a.confirm:
        confirm()
    else:
        worker(a.worker, a.n_trials)
