# Finds out why the first train-row encoder scores raised the holdout and lowered val. It fits a
# LightGBM ranker on random-order train feeds with and without the 9 encoder features, with
# ablations (content part only, ID part only, single windows, popularity inputs removed), and reads
# paired deltas on val and on held-out train feeds.
# How it was run: `python pretrain/diag_trainstage.py --exp all --threads 24` (CPU, 2026-11-11).
# What it produced: lab/logs/pretrain_diag_trainstage.{log,json}; the log is
# results/train_rows_leak_diagnosis.txt (all 9 features: val -0.0090 +- 0.0013, held-out train
# +0.0106 +- 0.0011). The cause: a train candidate had a content vector only if it occurs in some
# history, and among candidates with zero popularity that marks positives.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/pretrain/diag_trainstage.py
"""CPU diagnosis of why train-row pre_* hurt val when used in a TRAIN-fitted ranker.

Train-fit LightGBM lambdarank on in-window random-like train feeds (p_random > 0.9), frozen base features
(eval_cache/base_features.json with it_hist_* -> *_luo), +/- pre_* variants; evaluate on val (paired per-feed deltas)
and on a held-out set of in-window random-like train feeds.
Usage: python pretrain/diag_trainstage.py --exp all --threads 24
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

sys.path.insert(0, "<repo>/src")
from common import swap_luo, FLAT, FEATS
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path("<repo>")
PT = ROOT / "data/pretrain"
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1000], learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              lambdarank_truncation_level=1000, verbose=-1)
ROUNDS = 350
PRE9 = ["pre_fe", "pre_fe_z", "pre_new", "pre_new_z", "pre_hmax32", "pre_hmax8", "pre_fe_noid", "pre_id_contrib", "pre_idmax32"]


def rows_of(off, M, feeds):
    Mf = M[feeds]
    return np.repeat(off[feeds] - np.concatenate([[0], np.cumsum(Mf)[:-1]]), Mf) + np.arange(Mf.sum()), Mf


def load(split, names, rows, Mf):
    cols = []
    for n in names:
        if n == "pre_fe_noid_z":
            x = np.load(FEATS / split / "pre_fe_noid.npy", mmap_mode="r")
            x = np.asarray(x if rows is None else x[rows], np.float32)
            x = feed_rank_z(np.concatenate([[0], np.cumsum(Mf)[:-1]]).astype(np.int64), Mf, x)[:, 1]
        else:
            x = np.load(FEATS / split / f"{n}.npy", mmap_mode="r")
            x = np.asarray(x if rows is None else x[rows], np.float32)
        cols.append(x)
    return np.stack(cols, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_window", type=int, default=25000, help="train feeds per window (random-like)")
    ap.add_argument("--holdout", type=int, default=8000, help="held-out random-like feeds per window")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--exp", default="all")
    a = ap.parse_args()
    t0 = time.time()
    base = swap_luo(json.loads((ROOT / "data/eval_cache/base_features.json").read_text()))
    f = pl.read_parquet(FLAT / "train/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64); users = f["user_id"].to_numpy()
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"]
    rng = np.random.default_rng(0)
    tr_w, ho_w = {}, {}
    for k in [1, 2, 3]:
        wf = np.load(PT / f"win{k}/window_feeds.npy")
        wf = rng.permutation(wf[pr[wf] > 0.9])
        tr_w[k] = np.sort(wf[:a.per_window]); ho_w[k] = np.sort(wf[a.per_window:a.per_window + a.holdout])
    y_tr_all = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
    fv = pl.read_parquet(FLAT / "val/feeds.parquet")
    Mv = fv["M"].to_numpy().astype(np.int64); offv = fv["cand_off"].to_numpy().astype(np.int64)
    yv = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    feat_sets = {"base": base, "pre9": base + PRE9, "noid": base + ["pre_fe_noid", "pre_fe_noid_z"],
                 "idonly": base + ["pre_id_contrib", "pre_idmax32"], "zonly": base + ["pre_fe_z", "pre_new_z"]}
    noith = [n for n in base if not n.startswith("it_hist_")]
    nopop = [n for n in noith if not n.startswith(("it_cand_cnt", "tte_it_"))]
    feat_sets.update({"base_noith": noith, "pre9_noith": noith + PRE9, "base_nopop": nopop, "pre9_nopop": nopop + PRE9})
    allnames = list(dict.fromkeys(sum(feat_sets.values(), [])))
    # load everything once
    def block(feeds):
        r, Mf = rows_of(off, M, feeds)
        return load("train", allnames, r, Mf), y_tr_all[r].astype(np.float32), Mf, users[feeds]
    data = {}
    for k in [1, 2, 3]:
        data[("tr", k)] = block(tr_w[k]); data[("ho", k)] = block(ho_w[k])
    Xv = load("val", allnames, None, Mv)
    print(f"[data] loaded in {time.time()-t0:.0f}s; train rows {sum(data[('tr', k)][0].shape[0] for k in [1,2,3])}", flush=True)
    col = {n: i for i, n in enumerate(allnames)}

    def fit_eval(fs, wins, drop_user0=False, seed=0):
        idx = [col[n] for n in feat_sets[fs]]
        Xs, ys, Ms = [], [], []
        for k in wins:
            X, y, Mf, u = data[("tr", k)]
            if drop_user0:
                keepf = u != 0
                keepr = np.repeat(keepf, Mf)
                X, y, Mf = X[keepr], y[keepr], Mf[keepf]
            Xs.append(X[:, idx]); ys.append(y); Ms.append(Mf)
        p = dict(PARAMS, num_threads=a.threads, seed=seed)
        cat = ["category_id"] if "category_id" in feat_sets[fs] else []
        ds = lgb.Dataset(np.vstack(Xs), np.concatenate(ys), group=np.concatenate(Ms), feature_name=feat_sets[fs], categorical_feature=cat)
        b = lgb.train(p, ds, ROUNDS)
        pv = ndcg_per_feed(b.predict(Xv[:, idx], num_threads=a.threads), yv, offv, Mv)
        ph = []
        for k in [1, 2, 3]:
            X, y, Mf, _ = data[("ho", k)]
            ph.append(ndcg_per_feed(b.predict(X[:, idx], num_threads=a.threads), y, np.concatenate([[0], np.cumsum(Mf)[:-1]]), Mf))
        imp = dict(zip(feat_sets[fs], b.feature_importance("gain")))
        tot = sum(imp.values())
        pre_imp = {n: round(v / tot, 4) for n, v in imp.items() if n.startswith("pre_")}
        return pv, np.concatenate(ph), pre_imp

    exps = {
        "E1_all_windows": dict(wins=[1, 2, 3], arms=["base", "pre9", "noid", "idonly", "zonly"]),
        "E2_no_user0": dict(wins=[1, 2, 3], arms=["base", "pre9"], drop_user0=True),
        "E3_W1_only": dict(wins=[1], arms=["base", "pre9"]),
        "E4_W3_only": dict(wins=[3], arms=["base", "pre9"]),
        "E6_no_it_hist": dict(wins=[1, 2, 3], arms=["base_noith", "pre9_noith"], ref="base_noith"),
        "E7_no_item_pop": dict(wins=[1, 2, 3], arms=["base_nopop", "pre9_nopop"], ref="base_nopop"),
    }
    out = {}
    for name, e in exps.items():
        if a.exp != "all" and name not in a.exp.split(","):
            continue
        res = {}
        for arm in e["arms"]:
            pvs, phs, imps = [], [], []
            for s in range(a.seeds):
                pv, ph, imp = fit_eval(arm, e["wins"], e.get("drop_user0", False), seed=s)
                pvs.append(pv); phs.append(ph); imps.append(imp)
            res[arm] = (np.mean(pvs, 0), np.mean(phs, 0), imps[0])
            print(f"  {name} {arm}: val {res[arm][0].mean():.4f} holdout {res[arm][1].mean():.4f} pre-gain {imps[0]} ({time.time()-t0:.0f}s)", flush=True)
        r = {}
        ref = e.get("ref", "base")
        for arm in e["arms"]:
            if arm == ref:
                continue
            dv = res[arm][0] - res[ref][0]; dh = res[arm][1] - res[ref][1]
            r[arm] = dict(val=round(float(res[arm][0].mean()), 4), val_delta=round(float(dv.mean()), 5), val_se=round(float(dv.std() / np.sqrt(len(dv))), 5),
                          holdout=round(float(res[arm][1].mean()), 4), holdout_delta=round(float(dh.mean()), 5), holdout_se=round(float(dh.std() / np.sqrt(len(dh))), 5),
                          pre_gain_share=res[arm][2])
            print(f"{name} {arm} vs {ref}: VAL {dv.mean():+.5f} ({dv.std()/np.sqrt(len(dv)):.5f}) | HOLDOUT {dh.mean():+.5f} ({dh.std()/np.sqrt(len(dh)):.5f})", flush=True)
        r[ref] = dict(val=round(float(res[ref][0].mean()), 4), holdout=round(float(res[ref][1].mean()), 4))
        out[name] = r
        with open(ROOT / "logs/pretrain_diag_trainstage.json", "a") as fh:
            fh.write(json.dumps({name: r}) + "\n")


if __name__ == "__main__":
    main()
