"""cb26 E1/E2: residual CatBoost on the NN pool (rule and design: cbrule.py docstring, pre-declared 14:10 UTC).

  E1 (--mode hr): train on trH (30K random-like train holdout), baseline beta*z(pool_to); 5 user folds (3 train / 1 ES /
                  1 predict) -> honest trH OOF; val/test Delta = mean of the 5 fold models (val untouched).
  E2 (--mode vr): train on val, baseline beta*z(pool mix); n2cpv3 folds (GroupKFold(5), user0 singletons); fold k model
                  trains on 3 folds, early-stops on fold k+1, predicts fold k; test Delta = mean of the 5 fold models.
Outputs (data/): <tag>_delta_{val,test[,trH]}.npy (Delta only, row order = preds/), <tag>_beta.json, <tag>.log
GPU: CUDA_VISIBLE_DEVICES=<one id>, devices='0', gpu_ram_part 0.25; thread_count 12.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, "<repo>/nn_x/cb26")
from cbdata import meta, ens_feats, resid_feature_names, load_tab, zf, D  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fit_beta(z, y, M, off):
    """1-D conditional logit (listwise softmax, multi-positive) for the baseline scale."""
    fid = np.repeat(np.arange(len(M)), M)
    npos = np.bincount(fid, y, len(M))
    def nll(b):
        s = b * z.astype(np.float64)
        mx = np.maximum.reduceat(s, off)
        lse = np.log(np.add.reduceat(np.exp(s - mx[fid]), off)) + mx
        return -(np.bincount(fid, y * s, len(M)) - npos * lse).sum()
    lo, hi = 0.1, 8.0
    for _ in range(60):  # golden section
        a = hi - (hi - lo) / 1.618; b = lo + (hi - lo) / 1.618
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


def user_groups(f):
    u = f["user_id"].to_numpy().astype(np.int64)
    return np.where(u == 0, -1 - np.arange(len(u)), u)


def build_X(split, pool, mode, names):
    f, M, off, y = meta(split)
    cx = D / f"X_{mode}_{pool}_{split}.npy"
    if cx.exists():
        fn = json.loads((D / f"X_{mode}_{pool}_fn.json").read_text())
        return np.load(cx), fn, np.load(D / f"base_{mode}_{pool}_{split}.npy"), f, M, off, y
    e = ens_feats(pool, split)
    fid = np.repeat(np.arange(len(M)), M)
    if mode == "hr":
        base = e["to"]
        ens_cols = {"to": e["to"], "to_sd": e["to_sd"], **{k: v for k, v in e.items() if k.startswith("fam_") and k.endswith("_to")}}
    else:
        base = 0.5 * zf(off, M, e["ft"]) + 0.5 * zf(off, M, e["to"])
        ens_cols = {"pool": base, "to": e["to"], "ft": e["ft"], "to_sd": e["to_sd"], "ft_sd": e["ft_sd"],
                    "ftmto": zf(off, M, e["ft"]) - zf(off, M, e["to"]),
                    **{k: v for k, v in e.items() if k.startswith("fam_")}}
    base = zf(off, M, base)
    # within-feed rank pct and gap to the feed max of the baseline
    mx = np.maximum.reduceat(base, off)
    rk = np.empty(len(base), np.float32)
    for i in range(len(M)):
        o, m = off[i], M[i]
        rk[o:o + m] = np.argsort(np.argsort(-base[o:o + m])) / max(m - 1, 1)
    ens_cols["base_rk"] = rk; ens_cols["base_gap"] = (mx[fid] - base).astype(np.float32)
    # leak-free feed descriptors
    hl = load_tab(split, ["it_hist_all_luo"])[:, 0]
    cold = (np.nan_to_num(hl, nan=0.0) <= 0).astype(np.float32)
    ens_cols["is_cold"] = cold
    ens_cols["cold_share"] = (np.bincount(fid, cold, len(M)) / M)[fid].astype(np.float32)
    ens_cols["user0"] = (f["user_id"].to_numpy() == 0)[fid].astype(np.float32)
    T = load_tab(split, names)
    X = np.hstack([np.stack(list(ens_cols.values()), 1).astype(np.float32), T])
    fn = list(ens_cols) + names
    np.save(cx, X); np.save(D / f"base_{mode}_{pool}_{split}.npy", base.astype(np.float32))
    (D / f"X_{mode}_{pool}_fn.json").write_text(json.dumps(fn))
    return X, fn, base.astype(np.float32), f, M, off, y


def make_pool(X, fn, y, M, sel, base, beta):
    import pandas as pd
    from catboost import Pool
    fid = np.repeat(np.arange(len(M)), M)
    rows = np.isin(fid, sel)
    df = pd.DataFrame(X[rows], columns=fn)
    cat = []
    if "category_id" in fn:
        df["category_id"] = np.nan_to_num(df["category_id"].to_numpy(), nan=-1).astype(np.int32); cat = ["category_id"]
    gid = fid[rows]
    return Pool(df, label=None if y is None else y[rows], group_id=gid, cat_features=cat,
                baseline=None if base is None else (beta * base[rows]).astype(np.float64)), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hr", "vr"], required=True)
    ap.add_argument("--pool", default="G3")
    ap.add_argument("--loss", default="QuerySoftMax")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--od_wait", type=int, default=150)
    ap.add_argument("--metric_period", type=int, default=25)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--nfold_run", type=int, default=5, help="<5 only for timing tests")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.device == "gpu":
        assert os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", None) and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
    from catboost import CatBoostRanker
    t0 = time.time()
    names = [n for n in resid_feature_names() if not n.startswith("it_hist_")]  # it_hist_*_luo: parity skew (INVENTORY)
    tr_split = "trH" if a.mode == "hr" else "val"
    X, fn, base, f, M, off, y = build_X(tr_split, a.pool, a.mode, names)
    log(f"{tr_split} X {X.shape} built {time.time() - t0:.0f}s; features {len(fn)}")
    groups = user_groups(f)
    folds = [np.sort(v) for _, v in GroupKFold(n_splits=5).split(np.arange(len(M)), groups=groups)]
    fid = np.repeat(np.arange(len(M)), M)
    oof = np.zeros(len(y), np.float32)
    others = {}
    if a.mode == "hr":
        for sp in ["val", "test"]:
            others[sp] = build_X(sp, a.pool, a.mode, names)
            log(f"{sp} X built {time.time() - t0:.0f}s")
    else:
        others["test"] = build_X("test", a.pool, a.mode, names)
        log(f"test X built {time.time() - t0:.0f}s")
    dsum = {sp: np.zeros(len(v[0]), np.float64) for sp, v in others.items()}
    betas, iters = [], []
    for k in range(a.nfold_run):
        pr_f, es_f = folds[k], folds[(k + 1) % 5]
        tr_f = np.sort(np.concatenate([folds[j] for j in range(5) if j not in (k, (k + 1) % 5)]))
        trm = np.isin(fid, tr_f)
        offk = np.concatenate([[0], np.cumsum(M[tr_f])[:-1]])
        beta = fit_beta(base[trm], y[trm], M[tr_f], offk)
        betas.append(beta)
        ptr, _ = make_pool(X, fn, y, M, tr_f, base, beta)
        pes, _ = make_pool(X, fn, y, M, es_f, base, beta)
        dev = dict(task_type="GPU", devices="0", gpu_ram_part=0.15) if a.device == "gpu" else dict(task_type="CPU")
        m = CatBoostRanker(loss_function=a.loss, eval_metric="NDCG:type=Base", iterations=a.iters, learning_rate=a.lr,
                           depth=a.depth, l2_leaf_reg=a.l2, border_count=128, random_seed=k, metric_period=a.metric_period,
                           od_type="Iter", od_wait=a.od_wait, use_best_model=True, allow_writing_files=False,
                           thread_count=12, bootstrap_type="Bernoulli", subsample=0.8, **dev)
        m.fit(ptr, eval_set=pes, verbose=a.metric_period * 8)
        bi = m.get_best_iteration(); iters.append(int(bi))
        ppr, prm = make_pool(X, fn, None, M, pr_f, None, beta)
        oof[prm] = m.predict(ppr) + beta * base[prm]  # full score on the predicted fold (beta_k * base + Delta_k)
        for sp, (Xs, fns, bs, fs, Ms, offs, ys) in others.items():
            from catboost import Pool
            import pandas as pd
            df = pd.DataFrame(Xs, columns=fns)
            if "category_id" in fns:
                df["category_id"] = np.nan_to_num(df["category_id"].to_numpy(), nan=-1).astype(np.int32)
            dsum[sp] += m.predict(Pool(df, group_id=np.repeat(np.arange(len(Ms)), Ms),
                                       cat_features=["category_id"] if "category_id" in fns else []))
        imp = sorted(zip(m.get_feature_importance(type="PredictionValuesChange"), fn), reverse=True)[:12]
        log(f"fold {k}: beta {beta:.3f} best_iter {bi} | top {[(n, round(float(g), 1)) for g, n in imp]} | {time.time() - t0:.0f}s")
    np.save(D / f"{a.tag}_oofscore_{tr_split}.npy", oof)
    for sp in dsum:
        np.save(D / f"{a.tag}_delta_{sp}.npy", (dsum[sp] / a.nfold_run).astype(np.float32))
    res = dict(args=vars(a), betas=betas, beta_mean=float(np.mean(betas)), iters=iters, n_feat=len(fn), feats=fn,
               time=time.time() - t0)
    (D / f"{a.tag}_beta.json").write_text(json.dumps(res, indent=1))
    log("done", json.dumps({k: v for k, v in res.items() if k != "feats"}))


if __name__ == "__main__":
    main()
