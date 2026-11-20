"""cb26 E3 XM: CatBoost member trained on the train-period EXPLORATION subset (rule: cbrule.py docstring, 14:10 UTC).

Train feeds: p_random > 0.5 and M >= 50 (nn/train.py definition), minus the 30K holdout feeds.
Features: strongest leak-safe tabular set = swap_luo(base) - it_cand_cnt_* - hour + seq_ + ctx_ + clu_ + pre2s.
Early stopping on val user-half A (GroupKFold(2) by user, user0 singletons); report on half B, full val and trH.
Outputs data/<tag>_{val,test,trH}.npy (row order = preds/ and preds/holdout/), data/<tag>.json.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
from cbdata import meta, hold_index, trH_rows, load_tab, D, ROOT  # noqa: E402
from common import swap_luo, FEATS  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def member_features():
    base = json.loads((ROOT / "data/eval_cache/base_features.json").read_text())
    names = [n for n in swap_luo(base) if not n.startswith(("it_tr_", "u_tr_", "it_cand_cnt_", "hour"))]
    tr = {p.stem for p in (FEATS / "train").glob("*.npy")}
    va = {p.stem for p in (FEATS / "val").glob("*.npy")}
    extra = sorted(n for n in tr & va if n.startswith(("seq_", "ctx_", "clu_")))
    names += [n for n in extra if n not in names]
    names += ["pre2_fe_z", "pre2_new_z", "pre2_fe_s", "pre2_new_s", "pre2_hmax32"]
    return names


def to_df(X, fn):
    import pandas as pd
    df = pd.DataFrame(X, columns=fn, copy=False)
    df["category_id"] = np.nan_to_num(df["category_id"].to_numpy(), nan=-1).astype(np.int32)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="YetiRank")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--od_wait", type=int, default=200)
    ap.add_argument("--max_feeds", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", None) and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
    from catboost import CatBoostRanker, Pool
    t0 = time.time()
    names = member_features()
    log(f"{len(names)} features")
    hold, ftr = hold_index()
    Mtr = ftr["M"].to_numpy().astype(np.int64); otr = ftr["cand_off"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[Mtr < 50] = 0.0
    sel = np.setdiff1d(np.where((p > 0.5) & (Mtr >= 50))[0], hold)
    if a.max_feeds and len(sel) > a.max_feeds:
        sel = np.sort(np.random.default_rng(0).choice(sel, a.max_feeds, replace=False))
    rows = np.repeat(otr[sel] - np.r_[0, np.cumsum(Mtr[sel])[:-1]], Mtr[sel]) + np.arange(Mtr[sel].sum())
    log(f"train feeds {len(sel)} rows {len(rows)} (holdout overlap {np.isin(sel, hold).sum()})")
    Xtr = load_tab("train", names, rows)
    ytr = np.load(ROOT / "data/flat/train/cand_target.npy", mmap_mode="r")[rows].astype(np.float32)
    gtr = np.repeat(np.arange(len(sel)), Mtr[sel])
    log(f"train X {Xtr.shape} {time.time() - t0:.0f}s")
    fv, Mv, ov, yv = meta("val")
    Xv = load_tab("val", names)
    u = fv["user_id"].to_numpy().astype(np.int64); g = np.where(u == 0, -1 - np.arange(len(u)), u)
    hA, hB = [np.sort(x) for _, x in GroupKFold(n_splits=2).split(np.arange(len(Mv)), groups=g)]
    fidv = np.repeat(np.arange(len(Mv)), Mv)
    rA = np.isin(fidv, hA)
    ptr = Pool(to_df(Xtr, names), label=ytr, group_id=gtr, cat_features=["category_id"])
    del Xtr
    pA = Pool(to_df(Xv[rA], names), label=yv[rA], group_id=fidv[rA], cat_features=["category_id"])
    log(f"pools ready {time.time() - t0:.0f}s")
    m = CatBoostRanker(loss_function=a.loss, eval_metric="NDCG:type=Base", iterations=a.iters, learning_rate=a.lr,
                       depth=a.depth, l2_leaf_reg=5, border_count=254, task_type="GPU", devices="0", gpu_ram_part=0.2,
                       random_seed=0, metric_period=50, od_type="Iter", od_wait=a.od_wait, use_best_model=True,
                       allow_writing_files=False, thread_count=12)
    m.fit(ptr, eval_set=pA, verbose=100)
    del ptr
    out = {}
    out["val"] = m.predict(Pool(to_df(Xv, names), group_id=fidv, cat_features=["category_id"])).astype(np.float32)
    del Xv
    ft_, Mt, ot, _ = meta("test")
    out["test"] = m.predict(Pool(to_df(load_tab("test", names), names), group_id=np.repeat(np.arange(len(Mt)), Mt),
                                 cat_features=["category_id"])).astype(np.float32)
    fh, Mh, oh, yh = meta("trH")
    out["trH"] = m.predict(Pool(to_df(load_tab("trH", names), names), group_id=np.repeat(np.arange(len(Mh)), Mh),
                                cat_features=["category_id"])).astype(np.float32)
    for k, v in out.items():
        np.save(D / f"{a.tag}_{k}.npy", v)
    pv = ndcg_per_feed(out["val"], yv, ov, Mv)
    ph = ndcg_per_feed(out["trH"], yh, oh, Mh)
    imp = sorted(zip(m.get_feature_importance(type="PredictionValuesChange"), names), reverse=True)[:25]
    res = dict(args=vars(a), n_train_feeds=int(len(sel)), best_iter=int(m.get_best_iteration()),
               val_full=float(pv.mean()), val_halfA_ES=float(pv[hA].mean()), val_halfB=float(pv[hB].mean()),
               trH=float(ph.mean()), time=time.time() - t0, top=[(n, float(x)) for x, n in imp], features=names)
    (D / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
    np.save(D / f"{a.tag}_halfB_feeds.npy", hB)
    log("done", json.dumps({k: v for k, v in res.items() if k not in ("top", "features")}))


if __name__ == "__main__":
    main()
