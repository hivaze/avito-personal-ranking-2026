"""W25 ranker: CatBoost YetiRank (GPU) on exploration-like train feeds, cb26 e3 harness generalised to the W25 feature groups.

Arms (plan.txt): A0 = cb26 e3 reproduction (p_random>0.5 & M>=50 minus trH, base 177 features, lr 0.08).
                 A1 = base on full XM (all M).  A2 = A1 + groups.  Ablations = A1 + one group (fast config).
Row sets: train rows restricted to the p_random>0.5 feeds (nn_x/w25/data/train/rows.npy holds the flat row ids of those feeds, sorted);
W25 columns live in nn_x/w25/data/<split>/<name>.npy aligned to those rows (train) or to the flat order (val/test); trH rows are mapped by searchsorted.
Early stopping on val user-half A (GroupKFold(2) by user_id, user-0 singletons); report full val, half B, trH.
Outputs: out/<tag>_{val,test,trH}.npy (row order = preds/ for val/test, preds/holdout/ for trH), out/<tag>.json.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
from cbdata import meta, hold_index, trH_rows, load_tab, ROOT  # noqa: E402
from cb_member import member_features, to_df  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

W = ROOT / "nn_x/w25"
WD = W / "data"
OUT = W / "out"; OUT.mkdir(exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def w25_groups():
    p = WD / "groups.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_w25(split, names, rows=None):
    """W25 columns. split in train/val/test/trH. rows = flat row ids (train) ; trH -> mapped onto the stored p>0.5 rows."""
    if not names:
        return None
    if split in ("train", "trH"):
        stored = np.load(WD / "train/rows.npy")
        if split == "trH":
            rows = trH_rows()
        pos = np.searchsorted(stored, rows)
        assert (stored[pos] == rows).all(), "requested train rows outside the stored p>0.5 rows"
        sp = "train"
    else:
        pos = None; sp = split
    cols = []
    for n in names:
        x = np.load(WD / sp / f"{n}.npy", mmap_mode="r")
        cols.append(np.asarray(x if pos is None else x[pos], dtype=np.float32))
    return np.stack(cols, 1)


def load_all(split, base, extra, rows=None):
    X = load_tab(split, base, rows)
    if extra:
        X = np.concatenate([X, load_w25(split, extra, rows)], 1)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="xm", choices=["cb26", "xm"], help="cb26 = p>0.5 & M>=50 minus trH; xm = p>0.5 all M minus trH")
    ap.add_argument("--groups", default="", help="comma list of W25 groups (keys of data/groups.json) added to the base")
    ap.add_argument("--drop_groups", default="", help="comma list of groups removed from --groups (leave-one-out ablations)")
    ap.add_argument("--no_base", action="store_true")
    ap.add_argument("--loss", default="YetiRank")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--l2", type=float, default=5.0)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--od_wait", type=int, default=200)
    ap.add_argument("--sub", type=float, default=1.0, help="fraction of train feeds (seeded) for fast ablations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", None) and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
    from catboost import CatBoostRanker, Pool
    t0 = time.time()
    base = [] if a.no_base else member_features()
    G = w25_groups()
    gs = [g for g in a.groups.split(",") if g]
    dg = [g for g in a.drop_groups.split(",") if g]
    extra = []
    for g in gs:
        if g in dg:
            continue
        extra += [n for n in G[g] if n not in extra]
    names = base + extra
    log(f"{len(base)} base + {len(extra)} w25 features (groups {gs} minus {dg})")
    hold, ftr = hold_index()
    Mtr = ftr["M"].to_numpy().astype(np.int64); otr = ftr["cand_off"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    if a.rows == "cb26":
        p[Mtr < 50] = 0.0
        sel = np.setdiff1d(np.where((p > 0.5) & (Mtr >= 50))[0], hold)
    else:
        sel = np.setdiff1d(np.where(p > 0.5)[0], hold)
    if a.sub < 1.0:
        rng = np.random.default_rng(1000 + a.seed)
        sel = np.sort(rng.choice(sel, int(len(sel) * a.sub), replace=False))
    rows = np.repeat(otr[sel] - np.r_[0, np.cumsum(Mtr[sel])[:-1]], Mtr[sel]) + np.arange(Mtr[sel].sum())
    log(f"train feeds {len(sel)} rows {len(rows)} (holdout overlap {np.isin(sel, hold).sum()}; M<50 feeds {(Mtr[sel] < 50).sum()})")
    Xtr = load_all("train", base, extra, rows)
    ytr = np.load(ROOT / "data/flat/train/cand_target.npy", mmap_mode="r")[rows].astype(np.float32)
    gtr = np.repeat(np.arange(len(sel)), Mtr[sel])
    log(f"train X {Xtr.shape} {time.time() - t0:.0f}s; NaN share {np.isnan(Xtr).mean():.4f}")
    fv, Mv, ov, yv = meta("val")
    Xv = load_all("val", base, extra)
    u = fv["user_id"].to_numpy().astype(np.int64); g = np.where(u == 0, -1 - np.arange(len(u)), u)
    hA, hB = [np.sort(x) for _, x in GroupKFold(n_splits=2).split(np.arange(len(Mv)), groups=g)]
    fidv = np.repeat(np.arange(len(Mv)), Mv)
    rA = np.isin(fidv, hA)
    ptr = Pool(to_df(Xtr, names), label=ytr, group_id=gtr, cat_features=["category_id"])
    del Xtr
    pA = Pool(to_df(Xv[rA], names), label=yv[rA], group_id=fidv[rA], cat_features=["category_id"])
    log(f"pools ready {time.time() - t0:.0f}s")
    m = CatBoostRanker(loss_function=a.loss, eval_metric="NDCG:type=Base", iterations=a.iters, learning_rate=a.lr,
                       depth=a.depth, l2_leaf_reg=a.l2, border_count=254, task_type="GPU", devices="0", gpu_ram_part=0.25,
                       random_seed=a.seed, metric_period=50, od_type="Iter", od_wait=a.od_wait, use_best_model=True,
                       allow_writing_files=False, thread_count=12)
    m.fit(ptr, eval_set=pA, verbose=100)
    del ptr
    out = {}
    out["val"] = m.predict(Pool(to_df(Xv, names), group_id=fidv, cat_features=["category_id"])).astype(np.float32)
    del Xv
    ft_, Mt, ot, _ = meta("test")
    out["test"] = m.predict(Pool(to_df(load_all("test", base, extra), names), group_id=np.repeat(np.arange(len(Mt)), Mt),
                                 cat_features=["category_id"])).astype(np.float32)
    fh, Mh, oh, yh = meta("trH")
    out["trH"] = m.predict(Pool(to_df(load_all("trH", base, extra), names), group_id=np.repeat(np.arange(len(Mh)), Mh),
                                cat_features=["category_id"])).astype(np.float32)
    for k, v in out.items():
        np.save(OUT / f"{a.tag}_{k}.npy", v)
    pv = ndcg_per_feed(out["val"], yv, ov, Mv)
    ph = ndcg_per_feed(out["trH"], yh, oh, Mh)
    imp = sorted(zip(m.get_feature_importance(type="PredictionValuesChange"), names), reverse=True)
    res = dict(args=vars(a), n_train_feeds=int(len(sel)), n_train_rows=int(len(rows)), best_iter=int(m.get_best_iteration()),
               val_full=float(pv.mean()), val_se=float(pv.std() / np.sqrt(len(pv))), val_halfA_ES=float(pv[hA].mean()),
               val_halfB=float(pv[hB].mean()), val_Mlt50=float(pv[Mv < 50].mean()), val_Mge50=float(pv[Mv >= 50].mean()),
               trH=float(ph.mean()), trH_se=float(ph.std() / np.sqrt(len(ph))), time=time.time() - t0,
               top=[(n, round(float(x), 3)) for x, n in imp[:40]], imp_w25=[(n, round(float(x), 3)) for x, n in imp if n in extra],
               features=names)
    (OUT / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
    np.save(OUT / f"{a.tag}_halfB_feeds.npy", hB)
    log("done", json.dumps({k: v for k, v in res.items() if k not in ("top", "features", "imp_w25", "args")}))
    log("top w25:", res["imp_w25"][:20])


if __name__ == "__main__":
    main()
