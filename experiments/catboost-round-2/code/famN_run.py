# Family N runner (file name changed from run.py): CatBoost's own feature processing on this data:
# embedding features with the LDA and KNN calcers, text features as token sets, finer
# quantisation, one-hot categories, feature selection. Arms run on 30,000 or 6,000 feeds, each
# against a reference of the same size and loss.
# How it was run (docstring), from the project root: CUDA_VISIBLE_DEVICES="" python -u
# nn_x/w12/cb/famN/scripts/run.py --arms Yref Lref --seeds 0 1 [--nfit 30000 --nsel 5000]
# Its helper fn.py is not copied.
# What it produced: logs/results.jsonl and one screen line per run (results/famN_nats_lines.txt):
# 34 runs, none moved the information gain by more than 0.0001.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/cb/famN/scripts/run.py
"""famN arm runner. usage:
  cd <repo> && CUDA_VISIBLE_DEVICES="" python -u nn_x/w12/cb/famN/scripts/run.py --arms Yref Lref --seeds 0 1 [--nfit 30000 --nsel 5000] [--dry]
One process runs its arms one after another (seeds inside an arm share the pools). Every (arm, seed) appends one line to logs/results.jsonl and
writes preds/<tag>_{val,hold}.npy (+ _test.npy with --test), <tag>.eval.json, <tag>.curve.json, <tag>.cbm (dropped when larger than 300 MB).
tag = <arm>_n<nfit in K>k_s<seed>. `touch preds/<tag>.STOP` ends that fit cleanly.

Common model (the CPU reference of the track, nn_x/w12/cb/infra/out/ts_lr020_s0): depth 8, l2_leaf_reg 5, border_count 254, od Iter / 100,
eval_metric NDCG:type=Base on the train-side selection feeds (click labels), prediction at the best iteration.
Order of the training rows: feeds in file order (= time order), rows shuffled inside each feed, has_time=True (one permutation in that order),
so every ORDERED statistic CatBoost computes (category CTR, LDA, KNN) only sees earlier feeds and a label-independent part of the own feed.
Arms ending in _nt use CatBoost's default random permutations instead (control).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn  # noqa: E402
from fn import cbx, np, log  # noqa: E402

TSET = dict(s=dict(lb=50, size=500), l=dict(lb=20, size=2000))           # text dictionaries: occurrence_lower_bound, max_dictionary_size = top_tokens_count
LR = dict(YetiRank=0.2, RMSE=0.2, Logloss=0.1)
ITERS = dict(YetiRank=600, RMSE=800, Logloss=600)      # FIXED budgets (protocol v2, 01:59Z): no overfitting detector, global best on the selection feeds


def A(loss, label, **kw):
    d = dict(loss=loss, label=label, has_time=True, emb=None, calcers=None, text=None, tset=None, extra_cols=None, cb={}, cols_file=None)
    d.update(kw)
    return d


ARMS = {
    # ---- references ----
    "Yref": A("YetiRank", "y"),
    "Yref_nt": A("YetiRank", "y", has_time=False),
    "Lref": A("Logloss", "y"),
    "Rref": A("RMSE", "oof"),
    # ---- N1: LDA under Logloss (full-dimension vectors) ----
    "L_lda_pair": A("Logloss", "y", emb=[("e_pair", "pair", None)], calcers=["LDA:components=1"]),
    "L_lda_ehm": A("Logloss", "y", emb=[("e_ehm", "ehm", None)], calcers=["LDA:components=1"]),
    "L_lda_ue": A("Logloss", "y", emb=[("e_ue", "ue", None)], calcers=["LDA:components=1"]),
    "L_lda_all": A("Logloss", "y", emb=[("e_pair", "pair", None), ("e_ehm", "ehm", None), ("e_ue", "ue", None)], calcers=["LDA:components=1"]),
    # ---- N1: default calcers (LDA + KNN:k=5) under YetiRank ----
    "Y_def_pair": A("YetiRank", "y", emb=[("e_pair", "pair", None)]),
    "Y_def_ehm": A("YetiRank", "y", emb=[("e_ehm", "ehm", None)]),
    "Y_def_ue": A("YetiRank", "y", emb=[("e_ue", "ue", None)]),
    "Y_lda_all": A("YetiRank", "y", emb=[("e_pair", "pair", None), ("e_ehm", "ehm", None), ("e_ue", "ue", None)], calcers=["LDA:components=1"]),
    "Y_def_pair32": A("YetiRank", "y", emb=[("e_pair", "pair", 32)]),
    "Y_def_all32": A("YetiRank", "y", emb=[("e_pair", "pair", 32), ("e_ehm", "ehm", 32), ("e_ue", "ue", 32)]),   # default calcers on the three vectors together
    "Y_def_ehm32": A("YetiRank", "y", emb=[("e_ehm", "ehm", 32)]),
    "Y_def_ue32": A("YetiRank", "y", emb=[("e_ue", "ue", 32)]),
    # ---- N1: KNN on the out-of-fold teacher target under RMSE (32-d PCA) ----
    "R_knn50_pair": A("RMSE", "oof", emb=[("e_pair", "pair", 32)], calcers=["KNN:k=50"]),
    "R_knn200_pair": A("RMSE", "oof", emb=[("e_pair", "pair", 32)], calcers=["KNN:k=200"]),
    "R_knn50_ehm": A("RMSE", "oof", emb=[("e_ehm", "ehm", 32)], calcers=["KNN:k=50"]),
    "R_knn200_ehm": A("RMSE", "oof", emb=[("e_ehm", "ehm", 32)], calcers=["KNN:k=200"]),
    "R_knn50_ue": A("RMSE", "oof", emb=[("e_ue", "ue", 32)], calcers=["KNN:k=50"]),
    "R_knn200_ue": A("RMSE", "oof", emb=[("e_ue", "ue", 32)], calcers=["KNN:k=200"]),
    "R_knn5_pair": A("RMSE", "oof", emb=[("e_pair", "pair", 32)], calcers=["KNN:k=5"]),
    # ---- N2: text features as token sets (BoW) ----
    "Y_txt_s": A("YetiRank", "y", text=["cat", "clu"], tset="s"),
    "Y_txt_l": A("YetiRank", "y", text=["cat", "clu"], tset="l"),
    "R_txt_s": A("RMSE", "oof", text=["cat", "clu"], tset="s"),
    "R_txt_l": A("RMSE", "oof", text=["cat", "clu"], tset="l"),
    "Y_txtcat_l": A("YetiRank", "y", text=["cat"], tset="l"),
    "Y_txtclu_l": A("YetiRank", "y", text=["clu"], tset="l"),
    # ---- N3: quantisation, one-hot ----
    "Y_q1024": A("YetiRank", "y", cb=dict(per_float_feature_quantization=["78:border_count=1024", "172:border_count=1024"])),
    "Y_oh64": A("YetiRank", "y", cb=dict(one_hot_max_size=64)),
    "Y_sel": A("YetiRank", "y", cols_file="n3_select.json"),          # the feature set kept by select_features (n3.py select)
    # ---- N1x: hand-made columns (cache/hx/<col>_<set>.npy, built by hx.py; out-of-fold by user on train rows) ----
    "Y_hlda": A("YetiRank", "y", extra_cols=["hlda_pair", "hlda_ehm", "hlda_ue"]),
    "Y_hknn": A("YetiRank", "y", extra_cols=["hknn50_pair", "hknn200_pair"]),
    "Y_hx": A("YetiRank", "y", extra_cols=["hlda_pair", "hlda_ehm", "hlda_ue", "hknn50_pair", "hknn200_pair"]),
    "R_hknn": A("RMSE", "oof", extra_cols=["hknn50_pair", "hknn200_pair"]),
    "Y_hknn3": A("YetiRank", "y", extra_cols=["hknn50_pair", "hknn200_pair", "hknn50_ehm", "hknn200_ehm", "hknn50_ue", "hknn200_ue"]),
    "R_hknn3": A("RMSE", "oof", extra_cols=["hknn50_pair", "hknn200_pair", "hknn50_ehm", "hknn200_ehm", "hknn50_ue", "hknn200_ue"]),
    "Y_hall": A("YetiRank", "y", extra_cols=["hlda_pair", "hlda_ehm", "hlda_ue", "hknn50_pair", "hknn200_pair", "hknn50_ehm", "hknn200_ehm",
                                               "hknn50_ue", "hknn200_ue"]),
}
assert cbx.feature_names()[78] == "it_hist_et1_luo" and cbx.feature_names()[172] == "pre2_fe_z"


def hx_col(c, set_name, b, nk):
    p = fn.CACHE / "hx" / f"n{nk}k" / f"{c}_{set_name}.npy"
    a = np.load(p, mmap_mode="r")
    v = b.take(a).astype(np.float32)
    assert np.isfinite(v).all(), f"{p}: non-finite values on the rows of this block (column not built for these feeds)"
    return v


def make_params(cfg, seed, iters=None, lr=None, od_wait=0):
    loss = cfg["loss"]
    p = dict(loss_function=loss, eval_metric="NDCG:type=Base", iterations=iters or ITERS[loss], learning_rate=lr or LR[loss], depth=8, l2_leaf_reg=5,
             border_count=254, random_seed=seed, od_type="Iter", od_wait=od_wait or (iters or ITERS[loss]) + 1, use_best_model=False, allow_writing_files=False, thread_count=cbx.THREADS,
             metric_period=10, task_type="CPU")
    if cfg["has_time"]:
        p["has_time"] = True
    if cfg["calcers"]:
        p["embedding_calcers"] = list(cfg["calcers"])
    if cfg["text"]:
        t = TSET[cfg["tset"]]
        p["tokenizers"] = [{"tokenizer_id": "Space", "separator_type": "ByDelimiter", "delimiter": " "}]
        p["dictionaries"] = [{"dictionary_id": "W", "occurrence_lower_bound": str(t["lb"]), "max_dictionary_size": str(t["size"]), "gram_order": "1"}]
        p["feature_calcers"] = [f"BoW:top_tokens_count={t['size']}"]
    p.update(cfg["cb"])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--nfit", type=int, default=30000)      # 0 = all train-side fit feeds (full scale)
    ap.add_argument("--nsel", type=int, default=5000)       # 0 = all train-side selection feeds
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--test", action="store_true", help="also predict test (members)")
    ap.add_argument("--budget_min", type=float, default=0.0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--od_wait", type=int, default=0, help="0 = no early stopping (fixed budget, best iteration on the selection feeds)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for arm in a.arms:
        assert arm in ARMS, arm
    out_dir = fn.PRED / ("dry" if a.dry else "")
    tr = cbx.load("xm")
    fit, sel = fn.subset(a.nfit or None, a.nsel or None)
    nk = int(fit.sum()) // 1000
    bf, bs = fn.block("xm", fit, f"fit{a.nfit}"), fn.block("xm", sel, f"sel{a.nsel}")
    bv, bh = fn.block("val"), fn.block("hold")
    bt = fn.block("test") if a.test else None
    ev = cbx.Evaluator()
    log(f"fit feeds {int(fit.sum())} rows {bf.n} | selection feeds {int(sel.sum())} rows {bs.n} | arms {a.arms} seeds {a.seeds}")
    for arm in a.arms:
        cfg = ARMS[arm]
        t0 = time.time()
        kw = dict(emb=cfg["emb"], text=cfg["text"])
        if cfg["cols_file"]:
            keep = set(json.load(open(fn.LOGS / cfg["cols_file"]))["selected"]); kw["cols"] = [n for n in tr.names if n in keep]
            log(f"[{arm}] {len(kw['cols'])} of {len(tr.names)} features")
        ex = (lambda b, sn: None if not cfg["extra_cols"] else {c: hx_col(c, sn, b, nk) for c in cfg["extra_cols"]})
        ptr = fn.build_pool(bf, label=cfg["label"], shuffle=cfg["has_time"], extra=ex(bf, "xm"), **kw)
        pes = fn.build_pool(bs, label="y", extra=ex(bs, "xm"), **kw)
        pp = dict(val=fn.build_pool(bv, extra=ex(bv, "val"), **kw), hold=fn.build_pool(bh, extra=ex(bh, "hold"), **kw))
        pt = None if bt is None else fn.build_pool(bt, extra=ex(bt, "test"), **kw)
        pool_s = time.time() - t0
        log(f"[{arm}] pools {pool_s:.0f}s rss {cbx.rss_gb()[0]:.1f} GB; fit rows {ptr.num_row()} cols {ptr.num_col()}")
        for seed in a.seeds:
            tag = f"{arm}_n{nk}k_s{seed}{a.suffix}"
            if (out_dir / f"{tag}.eval.json").exists() and not a.dry and not a.force:
                log(f"[{tag}] exists, skipped"); continue
            p = make_params(cfg, seed, a.iters or None, a.lr or None, a.od_wait)
            for sp in ("val", "hold"):
                (fn.CACHE / "nats" / f"{tag}_{sp}.npz").unlink(missing_ok=True)
            r, m = fn.run_arm2(tag, p, ptr, pes, pp, out_dir=out_dir, ev=ev, budget_min=a.budget_min or None, test_pool=pt,
                               extra=dict(arm=arm, seed=seed, nfit=int(fit.sum()), nsel=int(sel.sum()), pool_seconds=pool_s, cfg={k: str(v) for k, v in cfg.items()}))
            s = fn.nats_line(tag, np.load(out_dir / f"{tag}_val.npy"), np.load(out_dir / f"{tag}_hold.npy"))
            x = r["extra"]
            line = dict(tag=tag, arm=arm, seed=seed, nfit=int(fit.sum()), nsel=int(sel.sum()), dry=bool(a.dry), val=r["val"], hold=r["hold"], rho_val=r["rho_val"],
                        rho_hold=r["rho_hold"], nats_val=s["val"]["nats"], nats_val_se=s["val"]["nats_se"], nats_hold=s["hold"]["nats"],
                        nats_hold_se=s["hold"]["nats_se"], promising=s["promising"], pooled_z=s["pooled_z"], member_pass=s["member_pass"],
                        mix01_val=s["val"]["mix01"], mix01_val_se=s["val"]["mix01_se"], mix01_hold=s["hold"]["mix01"], mix01_hold_se=s["hold"]["mix01_se"],
                        best_iter=x["best_iter"], trees=x["best_iter"] + 1, trees_built=x["trees_built"], fit_min=x["fit_seconds"] / 60,
                        s_per_iter=x["s_per_iter"], predict_s=x["predict_seconds"], model_mb=x["model_mb"], pool_s=pool_s, peak_rss_gb=x["peak_rss_gb"],
                        new_feature_importance=x["new_feature_importance"], top=x["top"][:6], time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            with open(fn.LOGS / ("results_dry.jsonl" if a.dry else "results.jsonl"), "a") as fh:
                fh.write(json.dumps(line) + "\n")
            del m
        del ptr, pes, pp, pt
    log("ALL DONE")


if __name__ == "__main__":
    main()
