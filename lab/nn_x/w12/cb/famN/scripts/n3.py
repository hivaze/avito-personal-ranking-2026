"""N3 information reads (no val / holdout label is used here: everything is computed on train-side feeds).
usage: CUDA_VISIBLE_DEVICES="" python -u n3.py lfc     -> LossFunctionChange importance of the reference Yref_n30k_s0/s1 on the train-side selection feeds
       CUDA_VISIBLE_DEVICES="" python -u n3.py select  -> one select_features run (RecursiveByShapValues, 177 -> 90 features in 3 steps), YetiRank
       CUDA_VISIBLE_DEVICES="" python -u n3.py qcheck  -> check that per_float_feature_quantization addresses the intended two columns
Writes logs/n3_lfc.json, logs/n3_select.json."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn  # noqa: E402
from fn import cbx, np, log  # noqa: E402
from catboost import CatBoost  # noqa: E402
import run as R  # noqa: E402

mode = sys.argv[1]
tr = cbx.load("xm"); names = tr.names
if mode == "qcheck":
    fit, sel = fn.subset(3000, 1000); bf = fn.block("xm", fit, "f")
    ptr = fn.build_pool(bf, label="y", shuffle=True)
    for tag, extra in (("default", {}), ("q1024", R.ARMS["Y_q1024"]["cb"])):
        p = R.make_params(R.ARMS["Yref"], 0, iters=3); p.update(extra)
        m = CatBoost(p); m.fit(ptr, verbose=0)
        fl = names  # get_borders() is keyed by the flat feature index
        b = m.get_borders()
        print(tag, {fl[i]: len(v) for i, v in b.items() if fl[i] in ("it_hist_et1_luo", "pre2_fe_z", "f1", "cos_u")}, "max borders over features", max(len(v) for v in b.values()),
              "one_hot_max_size", m.get_all_params().get("one_hot_max_size"), "ctr", str(m.get_all_params().get("simple_ctr"))[:80], flush=True)
    p = R.make_params(R.ARMS["Y_oh64"], 0, iters=3); m = CatBoost(p); m.fit(ptr, verbose=0)
    print("oh64: one_hot_max_size", m.get_all_params().get("one_hot_max_size"), "ctr features in model:", len(m.get_all_params().get("simple_ctr") or []),
          "category_id splits are one-hot:", any("category_id" in str(s) and "value" in str(s) for s in m._get_tree_splits(0, None)), m._get_tree_splits(0, None)[:8])
    sys.exit()
fit, sel = fn.subset(30000, 5000)
bs = fn.block("xm", sel, "sel")
pes = fn.build_pool(bs, label="y")
if mode == "lfc":
    out = {}
    for s in (0, 1):
        mp = fn.PRED / f"Yref_n30k_s{s}.cbm"
        m = CatBoost(); m.load_model(str(mp))
        k = json.load(open(fn.PRED / f"Yref_n30k_s{s}.eval.json"))["extra"]["best_iter"] + 1
        m.shrink(k)
        t0 = time.time()
        lfc = m.get_feature_importance(data=pes, type="LossFunctionChange", thread_count=cbx.THREADS)
        pvc = m.get_feature_importance(type="PredictionValuesChange", thread_count=cbx.THREADS)
        out[f"s{s}"] = dict(lfc=dict(zip(m.feature_names_, map(float, lfc))), pvc=dict(zip(m.feature_names_, map(float, pvc))), seconds=time.time() - t0, trees=k)
        log(f"seed {s}: LossFunctionChange on {pes.num_row()} selection rows in {time.time() - t0:.0f}s")
    json.dump(out, open(fn.LOGS / "n3_lfc.json", "w"), indent=0)
    a = np.array([[out["s0"]["lfc"][n], out["s1"]["lfc"][n]] for n in names]); pv = np.array([[out["s0"]["pvc"][n], out["s1"]["pvc"][n]] for n in names])
    o = np.argsort(-a.mean(1))
    print("corr of LossFunctionChange between the two seeds:", float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), "| Spearman-like corr LFC vs PVC (seed mean):",
          float(np.corrcoef(np.argsort(np.argsort(a.mean(1))), np.argsort(np.argsort(pv.mean(1))))[0, 1]))
    print("top 25 by LossFunctionChange (mean of 2 seeds; value s0 / s1; PredictionValuesChange rank):")
    pr = np.argsort(np.argsort(-pv.mean(1)))
    for i in o[:25]:
        print(f"  {names[i]:22s} {a[i].mean():.6f}  ({a[i, 0]:.6f} / {a[i, 1]:.6f})  PVC {pv[i].mean():.2f} rank {pr[i] + 1}")
    print("features with LossFunctionChange <= 0 in BOTH seeds:", int(((a <= 0).all(1)).sum()), "of", len(names), "| <= 0 on the seed mean:", int((a.mean(1) <= 0).sum()))
    print("share of total positive LFC in the top 20 / 50 / 90:", [round(float(np.sort(np.maximum(a.mean(1), 0))[::-1][:k].sum() / np.maximum(a.mean(1), 0).sum()), 3) for k in (20, 50, 90)])
if mode == "select":
    bf = fn.block("xm", fit, "fit")
    ptr = fn.build_pool(bf, label="y", shuffle=True)
    p = R.make_params(R.ARMS["Yref"], 0, iters=int(sys.argv[2]) if len(sys.argv) > 2 else 250)
    p.pop("od_type"); p.pop("od_wait")
    m = CatBoost(p)
    t0 = time.time()
    r = m.select_features(ptr, eval_set=pes, features_for_select=list(range(len(names))), num_features_to_select=int(sys.argv[3]) if len(sys.argv) > 3 else 90,
                          steps=3, algorithm="RecursiveByShapValues", shap_calc_type="Approximate", train_final_model=False, logging_level="Verbose", plot=False)
    out = dict(seconds=time.time() - t0, selected=r["selected_features_names"], eliminated=r["eliminated_features_names"], loss_graph=r["loss_graph"], params={k: str(v) for k, v in p.items()})
    json.dump(out, open(fn.LOGS / "n3_select.json", "w"), indent=0)
    log(f"select_features done in {time.time() - t0:.0f}s: kept {len(out['selected'])}, eliminated {len(out['eliminated'])}")
    print("eliminated (in order):", out["eliminated"])
    print("loss graph:", json.dumps(r["loss_graph"])[:1500])
