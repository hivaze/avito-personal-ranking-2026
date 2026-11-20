"""audit / ceiling probe: the distilled LightGBM student of record (long_bets/gbdt_kd.py arm K: L2 on the per-feed-centred
teacher_mean11 logit, cb26 177 features, exploration feeds minus the 30K holdout, rng(7) feed sample) run for MORE rounds /
other capacity. Same data selection code as gbdt_kd.py (copied, not imported: that script writes to its own out dir).
usage: gbdt_long.py <tag> <n_feeds> <rounds> <lr> <leaves> <min_data> <eval rounds comma list>
CPU only, 12 threads. Writes only to nn_x/w12/cb/audit/out/."""
import sys, time, json, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
import numpy as np, lightgbm as lgb
from cbdata import meta, hold_index, load_tab, ROOT
from cb_member import member_features
from metrics import ndcg_per_feed
tag, n_feeds, rounds, lr, leaves, mdl = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
evs = set(int(x) for x in sys.argv[7].split(","))
OUT = ROOT / "nn_x/w12/cb/audit/out"
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
names = member_features(); log(len(names), "features")
hold, ftr = hold_index()
M = ftr["M"].to_numpy().astype(np.int64); off = ftr["cand_off"].to_numpy().astype(np.int64)
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz"); p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[M < 50] = 0.0
cand = np.setdiff1d(np.where((p > 0.5) & (M >= 50))[0], hold)
sel = np.sort(np.random.default_rng(7).choice(cand, min(n_feeds, len(cand)), replace=False))
assert np.isin(sel, hold).sum() == 0
rows = np.repeat(off[sel] - np.r_[0, np.cumsum(M[sel])[:-1]], M[sel]) + np.arange(M[sel].sum())
log(f"{len(cand)} candidate feeds, using {len(sel)} feeds / {len(rows)} rows")
X = load_tab("train", names, rows); g = M[sel]
t = np.asarray(np.load(ROOT / "nn_x/w12/kd/out/teacher_mean11.npy", mmap_mode="r")[rows], np.float64)
if np.abs(t).mean() > 20: t = t / 11.0
o = np.r_[0, np.cumsum(g)[:-1]]; mu = np.add.reduceat(t, o) / g
lab = (t - np.repeat(mu, g)).astype(np.float32)
log("X", X.shape, "teacher centred logit sd", float(lab.std()))
P = dict(objective="regression", learning_rate=lr, num_leaves=leaves, min_data_in_leaf=mdl, feature_fraction=0.5, bagging_fraction=0.5,
         bagging_freq=1, lambda_l2=10, max_bin=255, num_threads=12, verbose=-1, seed=0)
d = lgb.Dataset(X, lab, feature_name=names, categorical_feature=["category_id"], free_raw_data=True)
fv, Mv, ov, yv = meta("val"); Xv = load_tab("val", names)
fh, Mh, oh, yh = meta("trH"); Xh = load_tab("trH", names)
log("eval matrices", Xv.shape, Xh.shape)
res = {}
def cb(env):
    it = env.iteration + 1
    if it in evs or it == rounds:
        pv = env.model.predict(Xv, num_iteration=it, num_threads=12); ph = env.model.predict(Xh, num_iteration=it, num_threads=12)
        nv = ndcg_per_feed(pv.astype(np.float32), yv, ov, Mv).mean(); nh = ndcg_per_feed(ph.astype(np.float32), yh, oh, Mh).mean()
        res[it] = (float(nv), float(nh)); log(f"round {it}: val {nv:.5f} holdout {nh:.5f}")
        np.save(OUT / f"{tag}_r{it}_val.npy", pv.astype(np.float32)); np.save(OUT / f"{tag}_r{it}_trH.npy", ph.astype(np.float32))
        json.dump(dict(tag=tag, n_feeds=len(sel), rows=int(len(rows)), params=P, res=res, minutes=(time.time() - t0) / 60), open(OUT / f"{tag}.json", "w"))
b = lgb.train(P, d, rounds, callbacks=[cb])
b.save_model(str(OUT / f"{tag}.lgb.txt"))
log("DONE")
