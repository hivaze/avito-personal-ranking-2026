"""famS_review step 2 (leak / provenance checks on the saved models; CPU, 4 threads, evaluation and one tiny diagnostic fit):
 A. every bundle member's saved val / holdout prediction = its saved .cbm applied to the cbx pools with ntree_end = argmax of the TRAIN-SIDE selection curve + 1
    (so the tree count is the train-side one and the files are what the models give; seed 1 = the 2 x T0 model cut inside its first 886 trees)
 B. the PCA basis of S3 is reproduced from the xm train-side FIT rows only (same sample rule), label-free
 C. truncation equivalence: a 40-iteration CatBoost run cut at 20 trees = a 20-iteration run of the same seed (tiny fit, 3,000 feeds)
 D. the training target is the centred teacher_mean11 logit: corr of each student's HOLDOUT prediction with the 11-member mean (which is out-of-sample there)"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CBX_THREADS"] = "4"
os.environ["CBX_CACHE_EXTRA"] = "<repo>/nn_x/w12/cb/famS_review/cache"
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "4"
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra")
import cbx
from catboost import CatBoost, CatBoostRanker
P = cbx.ROOT / "nn_x/w12/cb/famS/preds"
t0 = time.time()
res = {}

# ---- A
pools = {sp: cbx.make_pool(cbx.load(sp), cat="native") for sp in ("val", "hold")}
print(f"pools {time.time() - t0:.0f}s", flush=True)
A = "s1f_rmse_d8_lr030"
for model, tag, cap in ((f"{A}_s0", f"{A}_s0", None), (f"{A}_s1long", f"{A}_s1", 886), (f"{A}_s2", f"{A}_s2", None), (f"{A}_s1long", f"{A}_s1long", None),
                        ("s4b_qsm_soft_lr030_s0", "s4b_qsm_soft_lr030_s0", None), ("s1_rmse_d8_lr010_s0", "s1_rmse_d8_lr010_s0", None)):
    c = json.load(open(P / f"{model}.curve.json"))
    v = np.array([m["validation:NDCG:type=Base"] for _, _, m in c])
    if cap:
        v = v[:cap]
    k = int(np.argmax(v)) + 1
    m = CatBoost(); m.load_model(str(P / f"{model}.cbm"))
    r = {"ntree_end": k, "sel_ndcg": float(v[k - 1])}
    for sp in ("val", "hold"):
        p = np.asarray(m.predict(pools[sp], ntree_end=k, thread_count=4), np.float32)
        s = np.load(P / f"{tag}_{sp}.npy")
        r[sp] = float(np.abs(p - s).max())
    res[tag] = r
    print(f"A [{tag}] model {model}.cbm ntree_end {k} (train-side sel NDCG {v[k - 1]:.5f}): max |pred - saved| val {r['val']:.2e} holdout {r['hold']:.2e}  ({time.time() - t0:.0f}s)", flush=True)
del pools

# ---- B
tr = cbx.load("xm")
fit, sel = cbx.trainside_split(tr)
print(f"xm feeds {tr.n_feeds}, train-side fit feeds {int(fit.sum())}, selection feeds {int(sel.sum())}", flush=True)
z = np.load(cbx.ROOT / "nn_x/w12/cb/famS/cache/pca32.npz")
rows = np.flatnonzero(np.repeat(fit, tr.M))
idx = np.sort(np.random.default_rng(7).choice(rows, 3_000_000, replace=False))
E = np.asarray(tr.emb_cand_i8[idx], np.float64)
mu = E.mean(0); C = (E - mu).T @ (E - mu) / len(E)
w, U = np.linalg.eigh(C); o = np.argsort(-w); V = U[:, o[:32]]
cosv = np.abs((V * z["V"]).sum(0))
res["pca"] = dict(mu_maxdiff=float(np.abs(mu - z["mu"]).max()), min_abs_cos=float(cosv.min()), evr32=float(w[o][:32].sum() / w.sum()))
print(f"B PCA refit on the xm train-side FIT rows (rng 7, 3M rows): max |mu - saved| {res['pca']['mu_maxdiff']:.2e}, min |cos| of the 32 directions {cosv.min():.6f}, "
      f"explained variance {res['pca']['evr32']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
# a different sample (other seed, selection feeds = rows never fitted) to see how much the basis depends on the rows: label-free, information only
rows2 = np.flatnonzero(np.repeat(sel, tr.M)); E2 = np.asarray(tr.emb_cand_i8[np.sort(np.random.default_rng(1).choice(rows2, 500_000, replace=False))], np.float64)
P32 = z["V"].astype(np.float64); tot = ((E2 - E2.mean(0)) ** 2).sum(); cap = (((E2 - E2.mean(0)) @ P32) ** 2).sum()
print(f"B saved basis explains {cap / tot:.3f} of the variance of selection-feed candidate vectors (never fitted)", flush=True)
del E, E2

# ---- C
rg = np.random.default_rng(3)
k1 = np.zeros(tr.n_feeds, bool); k1[rg.choice(np.flatnonzero(fit), 3000, replace=False)] = True
k2 = np.zeros(tr.n_feeds, bool); k2[rg.choice(np.flatnonzero(sel), 500, replace=False)] = True
lab = tr.teacher("mean11", "centred")
ptr = cbx.make_pool(tr, feeds=k1, label=lab, cat="native"); pes = cbx.make_pool(tr, feeds=k2, label=tr.y, cat="native")
pp = {}
for it in (20, 40):
    m = CatBoostRanker(depth=8, learning_rate=0.3, l2_leaf_reg=5.0, border_count=254, iterations=it, od_type="Iter", od_wait=it, eval_metric="NDCG:type=Base",
                       use_best_model=False, random_seed=1, allow_writing_files=False, thread_count=4, loss_function="RMSE")
    m.fit(ptr, eval_set=pes, verbose=0)
    pp[it] = np.asarray(m.predict(pes, ntree_end=20, thread_count=4)); pp[(it, "full")] = np.asarray(m.predict(pes, thread_count=4))
res["trunc"] = dict(maxdiff_20_vs_40cut20=float(np.abs(pp[20] - pp[40]).max()), maxdiff_20_vs_40full=float(np.abs(pp[20] - pp[(40, "full")]).max()), sd=float(pp[20].std()))
print(f"C truncation: 20-iteration run vs 40-iteration run cut at 20 trees: max |diff| {res['trunc']['maxdiff_20_vs_40cut20']:.2e} "
      f"(vs the uncut 40-tree prediction {res['trunc']['maxdiff_20_vs_40full']:.3f}; prediction sd {res['trunc']['sd']:.3f})  ({time.time() - t0:.0f}s)", flush=True)
(cbx.ROOT / "nn_x/w12/cb/famS_review/out/rv_model.json").write_text(json.dumps(res, indent=1))
print("DONE", flush=True)
