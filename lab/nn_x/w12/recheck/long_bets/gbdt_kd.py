"""long_bets probe P3: is a GBDT a useful STUDENT? Same feeds, same features (cb26 member set), same LightGBM params, fixed rounds.
 arms: H = lambdarank on the hard click; K = L2 on the per-feed-centred in-sample teacher logit (teacher_mean11);
       X = L2 on the centred cross-fitted teacher logit (teacher_oof); B = xentropy on 0.5*y + 0.5*M*softmax(teacher)/... (pointwise mix)
 usage: gbdt_kd.py <arm> <n_feeds> <pop: xm|all> [rounds]   -> out/gbdt_<arm>_<pop>_<n>_{val,trH}.npy (+ per-200-round NDCG log)
CPU only, 12 threads."""
import sys, time, json, os
os.environ.setdefault("OMP_NUM_THREADS", "10")
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
import numpy as np, lightgbm as lgb
from cbdata import meta, hold_index, load_tab, ROOT
from cb_member import member_features
from metrics import ndcg_per_feed
arm, n_feeds, pop = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 600
OUT = ROOT / "nn_x/w12/recheck/long_bets/out"; OUT.mkdir(exist_ok=True)
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
names = member_features(); log(len(names), "features")
hold, ftr = hold_index()
M = ftr["M"].to_numpy().astype(np.int64); off = ftr["cand_off"].to_numpy().astype(np.int64)
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz"); p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[M < 50] = 0.0
if pop == "xm":
    cand = np.setdiff1d(np.where((p > 0.5) & (M >= 50))[0], hold)
else:  # all train feeds (production + exploration), any M >= 5, minus the holdout
    cand = np.setdiff1d(np.arange(len(M)), hold)
sel = np.sort(np.random.default_rng(7).choice(cand, min(n_feeds, len(cand)), replace=False))
rows = np.repeat(off[sel] - np.r_[0, np.cumsum(M[sel])[:-1]], M[sel]) + np.arange(M[sel].sum())
log(f"pop {pop}: {len(cand)} candidate feeds, using {len(sel)} feeds / {len(rows)} rows")
X = load_tab("train", names, rows); g = M[sel]
y = np.load(ROOT / "data/flat/train/cand_target.npy", mmap_mode="r")[rows].astype(np.float32)
log("X", X.shape)
def centred(file):
    t = np.asarray(np.load(ROOT / f"nn_x/w12/kd/out/{file}", mmap_mode="r")[rows], np.float64)
    if file == "teacher_mean11.npy" and np.abs(t).mean() > 20: t = t / 11.0
    o = np.r_[0, np.cumsum(g)[:-1]]; mu = np.add.reduceat(t, o) / g
    return (t - np.repeat(mu, g)).astype(np.float32)
P = dict(learning_rate=0.08, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.5, bagging_freq=1,
         lambda_l2=10, max_bin=255, num_threads=10, verbose=-1, seed=0)
cat = ["category_id"]
if arm == "H":
    P.update(objective="lambdarank", lambdarank_truncation_level=1000, learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8); lab = y; grp = g  # src/train_lgb.py settings
elif arm in ("K", "X"):
    lab = centred("teacher_mean11.npy" if arm == "K" else "teacher_oof.npy"); P.update(objective="regression"); grp = None
    log("teacher centred logit sd", float(lab.std()))
elif arm in ("C", "CY", "Y"):  # Y = teacher logit + 1.0*y (click boosted); C / CY add co-click label-conditional target: teacher logit + 0.33*z(maxcos to OTHER clicked items) + 0.15*samecat (P2 coefficients)
    import json as _j
    dd = ROOT / "data/competition/train/item_embeddings"; mt = _j.loads((dd / "data.meta").read_text())
    table = np.memmap(dd / "data.memmap", dtype=np.int8, mode="r", shape=tuple(mt["shape"]))
    ei = np.load(ROOT / "data/flat/train/cand_emb_idx.npy", mmap_mode="r"); ct = np.load(ROOT / "data/flat/train/cand_category_id.npy", mmap_mode="r")
    t = centred("teacher_mean11.npy").astype(np.float64); add = np.zeros(len(t)); o = np.r_[0, np.cumsum(g)[:-1]]
    for fi, i in enumerate(sel if arm != "Y" else []):
        sl = slice(off[i], off[i] + M[i]); e = np.asarray(ei[sl]); E = np.asarray(table[np.maximum(e, 0)], np.float32)
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6); E[e < 0] = 0
        c = np.asarray(ct[sl]); yy = y[o[fi]:o[fi] + g[fi]] > 0; Pp = np.where(yy)[0]; S = E @ E[Pp].T
        sim = S.max(1).astype(np.float64); sc = np.isin(c, c[Pp]).astype(np.float64)
        for k, ip in enumerate(Pp):
            Rk = [kk for kk in range(len(Pp)) if kk != k]
            if Rk: sim[ip] = S[ip, Rk].max(); sc[ip] = float(c[ip] in set(c[Pp[Rk]].tolist()))
            else: sim[ip] = np.nan; sc[ip] = 0.0
        ok = ~np.isnan(sim); z = np.zeros(len(sim)); z[ok] = (sim[ok] - sim[ok].mean()) / (sim[ok].std() + 1e-9)
        a_ = 0.33 * z + 0.15 * (sc - sc[ok].mean()); add[o[fi]:o[fi] + g[fi]] = a_ - a_.mean()
    if arm in ("Y", "CY"):
        yc = y.astype(np.float64) - np.repeat(np.add.reduceat(y.astype(np.float64), o) / g, g); add = add + 1.0 * yc
    lab = (t + add).astype(np.float32); P.update(objective="regression"); grp = None
    log("co-click target: sd of the added term", float(add.std()), "teacher sd", float(t.std()))
elif arm == "B":  # pointwise soft-target mix: 0.5*y + 0.5*softmax(teacher) (per feed), cross-entropy on [0,1] labels
    t = centred("teacher_mean11.npy").astype(np.float64); o = np.r_[0, np.cumsum(g)[:-1]]
    e = np.exp(t - np.repeat(np.maximum.reduceat(t, o), g)); q = e / np.repeat(np.add.reduceat(e, o), g)
    ysum = np.repeat(np.add.reduceat(y.astype(np.float64), o), g)
    lab = np.clip(0.5 * y + 0.5 * q * ysum, 0, 1).astype(np.float32); P.update(objective="cross_entropy"); grp = None
else:
    raise SystemExit("arm")
d = lgb.Dataset(X, lab, group=grp, feature_name=names, categorical_feature=cat, free_raw_data=True)
fv, Mv, ov, yv = meta("val"); Xv = load_tab("val", names)
fh, Mh, oh, yh = meta("trH"); Xh = load_tab("trH", names)
log("eval matrices", Xv.shape, Xh.shape)
res = {}
def cb(env):
    it = env.iteration + 1
    if it in (50, 100, 150) or it % 200 == 0 or it == rounds:
        pv = env.model.predict(Xv, num_iteration=it); ph = env.model.predict(Xh, num_iteration=it)
        nv = ndcg_per_feed(pv.astype(np.float32), yv, ov, Mv).mean(); nh = ndcg_per_feed(ph.astype(np.float32), yh, oh, Mh).mean()
        res[it] = (float(nv), float(nh)); log(f"round {it}: val {nv:.5f} holdout {nh:.5f}")
        if it == rounds:
            tag = f"gbdt_{arm}_{pop}_{n_feeds}"
            np.save(OUT / f"{tag}_val.npy", pv.astype(np.float32)); np.save(OUT / f"{tag}_trH.npy", ph.astype(np.float32))
b = lgb.train(P, d, rounds, callbacks=[cb])
json.dump(dict(arm=arm, n_feeds=len(sel), pop=pop, rows=int(len(rows)), res=res, minutes=(time.time() - t0) / 60),
          open(OUT / f"gbdt_{arm}_{pop}_{n_feeds}.json", "w"))
log("DONE")
