"""Gate checks for the section-8c rebuild (pre2_*), all CPU:
 G1 train-fit LightGBM (8b setup: 25K random-like feeds per window, frozen LUO base, 2 seeds): base vs base + pre2 (9),
    and base + prew (8b interim fix) for reference; val and train-holdout paired deltas.
 G2 mean within-feed z of the main score on LUO-cold candidates (it_hist_all_luo == 0), train windows vs val.
 G3 has-later split extended to ANY later feed of the same user (any train window, val or test), pre2_final vs references.
Usage: python pretrain/gate_pre2.py [--skip_lgb]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/pretrain")
from common import swap_luo, FLAT, FEATS
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from diag_trainstage import PARAMS, ROUNDS, rows_of, load

ROOT = Path("<repo>"); PT = ROOT / "data/pretrain"
NAMES = ["fe", "fe_z", "new", "new_z", "hmax32", "hmax8", "fe_noid", "id_contrib", "idmax32"]


def g1(threads, wins=(1, 2, 3)):
    base = swap_luo(json.loads((ROOT / "data/eval_cache/base_features.json").read_text()))
    f = pl.read_parquet(FLAT / "train/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"]
    y_all = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    tr, ho = [], []
    for k in wins:
        wf = np.load(PT / f"win{k}/window_feeds.npy"); wf = rng.permutation(wf[pr[wf] > 0.9])
        tr.append(wf[:25000]); ho.append(wf[25000:33000])
    tr, ho = np.sort(np.concatenate(tr)), np.sort(np.concatenate(ho))
    sets = {"base": base, "pre2": base + [f"pre2_{n}" for n in NAMES], "prew": base + [f"prew_{n}" for n in NAMES]}
    alln = list(dict.fromkeys(sum(sets.values(), [])))
    col = {n: i for i, n in enumerate(alln)}
    def block(feeds):
        r, Mf = rows_of(off, M, feeds)
        return load("train", alln, r, Mf), y_all[r].astype(np.float32), Mf, np.concatenate([[0], np.cumsum(Mf)[:-1]]).astype(np.int64)
    X, y, Mt, _ = block(tr); H, yh, Mh, oh = block(ho)
    fv = pl.read_parquet(FLAT / "val/feeds.parquet")
    Mv = fv["M"].to_numpy().astype(np.int64); offv = fv["cand_off"].to_numpy().astype(np.int64)
    yv = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    V = load("val", alln, None, Mv)
    res = {}
    for arm, names in sets.items():
        idx = [col[n] for n in names]
        pv, ph = [], []
        for s in range(2):
            b = lgb.train(dict(PARAMS, num_threads=threads, seed=s), lgb.Dataset(X[:, idx], y, group=Mt, feature_name=names, categorical_feature=["category_id"]), ROUNDS)
            pv.append(ndcg_per_feed(b.predict(V[:, idx], num_threads=threads), yv, offv, Mv)); ph.append(ndcg_per_feed(b.predict(H[:, idx], num_threads=threads), yh, oh, Mh))
        res[arm] = (np.mean(pv, 0), np.mean(ph, 0))
        print(f"  G1 {arm}: val {res[arm][0].mean():.4f} holdout {res[arm][1].mean():.4f}", flush=True)
    out = {}
    for arm in ["pre2", "prew"]:
        dv = res[arm][0] - res["base"][0]; dh = res[arm][1] - res["base"][1]
        out[arm] = dict(val_delta=round(float(dv.mean()), 5), val_se=round(float(dv.std() / np.sqrt(len(dv))), 5),
                        holdout_delta=round(float(dh.mean()), 5), holdout_se=round(float(dh.std() / np.sqrt(len(dh))), 5))
        print(f"G1 {arm} vs base: VAL {dv.mean():+.5f} ({dv.std()/np.sqrt(len(dv)):.5f}) | HOLDOUT {dh.mean():+.5f} ({dh.std()/np.sqrt(len(dh)):.5f})", flush=True)
    dd = res["pre2"][0] - res["prew"][0]
    out["pre2_minus_prew_val"] = (round(float(dd.mean()), 5), round(float(dd.std() / np.sqrt(len(dd))), 5))
    print(f"G1 pre2 - prew: VAL {dd.mean():+.5f} ({dd.std()/np.sqrt(len(dd)):.5f})", flush=True)
    return out


def g2_g3(wins=(1, 2, 3)):
    out = {}
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"]
    f = pl.read_parquet(FLAT / "train/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    ts = f["timestamp"].to_numpy().astype(np.int64); users = f["user_id"].to_numpy()
    # last feed time of each user over train + val + test
    allf = pl.concat([pl.read_parquet(FLAT / s / "feeds.parquet").select("user_id", "timestamp") for s in ["train", "val", "test"]])
    tmax = allf.filter(pl.col("user_id") != 0).group_by("user_id").agg(pl.col("timestamp").max().alias("tmax"))
    # G2 on val
    fv = pl.read_parquet(FLAT / "val/feeds.parquet"); Mv = fv["M"].to_numpy().astype(np.int64); offv = fv["cand_off"].to_numpy().astype(np.int64)
    cold_v = np.load(FEATS / "val/it_hist_all_luo.npy") == 0
    zv = np.load(FEATS / "val/pre2_fe_z.npy")
    out["G2_val"] = dict(mean_z_cold=round(float(zv[cold_v].mean()), 3), mean_z_warm=round(float(zv[~cold_v].mean()), 3))
    print("G2 val", out["G2_val"], flush=True)
    y_all = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
    score_all = np.load(ROOT / "preds/pre2_final_train.npy", mmap_mode="r")
    for k in wins:
        feeds = np.load(PT / f"win{k}/window_feeds.npy")
        r, Mf = rows_of(off, M, feeds)
        o = np.concatenate([[0], np.cumsum(Mf)[:-1]]).astype(np.int64)
        y = y_all[r].astype(np.float64)
        cold = np.asarray(np.load(FEATS / "train/it_hist_all_luo.npy", mmap_mode="r")[r]) == 0
        z = np.asarray(np.load(FEATS / "train/pre2_fe_z.npy", mmap_mode="r")[r])
        g2 = dict(mean_z_cold=round(float(z[cold].mean()), 3), mean_z_warm=round(float(z[~cold].mean()), 3), cold_share=round(float(cold.mean()), 3))
        wf = pl.DataFrame({"u": users[feeds], "t": ts[feeds]}).join(tmax, left_on="u", right_on="user_id", how="left")
        later_any = ((wf["u"] != 0) & (wf["tmax"].fill_null(0) > wf["t"])).to_numpy()
        rnd = pr[feeds] > 0.9
        g3 = {}
        for name, x in [("pre2_final", np.asarray(score_all[r], np.float64)),
                        ("maxsim32", np.asarray(np.load(FEATS / "train/maxsim32.npy", mmap_mode="r")[r], np.float64)),
                        ("cos_u", np.asarray(np.load(FEATS / "train/cos_u.npy", mmap_mode="r")[r], np.float64)),
                        ("f1", np.asarray(np.load(FEATS / "train/f1.npy", mmap_mode="r")[r], np.float64))]:
            pf = ndcg_per_feed(x, y, o, Mf)
            row = {}
            for sub, m in [("all", np.ones(len(feeds), bool)), ("random", rnd)]:
                a_, b_ = pf[m & later_any], pf[m & ~later_any]
                row[sub] = dict(n_later=int(len(a_)), has_later=round(float(a_.mean()), 4), no_later=round(float(b_.mean()), 4),
                                gap=round(float(a_.mean() - b_.mean()), 4), se=round(float(np.sqrt(a_.var() / len(a_) + b_.var() / len(b_))), 4))
            g3[name] = row
        out[f"W{k}"] = dict(G2=g2, G3=g3)
        print(f"W{k} G2 {g2}", flush=True)
        for name, row in g3.items():
            print(f"W{k} G3 {name} {json.dumps(row)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip_lgb", action="store_true"); ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--windows", default="1,2,3"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    wins = [int(x) for x in a.windows.split(",")]
    res = {} if a.skip_lgb else {"G1": g1(a.threads, wins)}
    res.update(g2_g3(wins))
    (ROOT / f"logs/pre2_gate{a.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
