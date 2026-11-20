"""Information-only reads declared in AMENDMENT 3: MIN arms (control + [z_w, cs_w]) and the convergence check (maxiter 5000).
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 NUMBA_NUM_THREADS=6 .venv/bin/python nn_x/w12/fresh/pregate/s3c_min.py
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "<repo>/nn_x/w12/fresh/pregate")
from lib import OUT, WIN, forms
from nested import ARMS, FOLD_SEEDS, NFOLD, Fitter, Val, gain, nested_scores, scale, wstat

sys.path.insert(0, "<repo>/src")
import pmix
from metrics import ndcg_per_feed

t0 = time.time()
V = Val(); W = pmix.weights()
cm = np.load(f"{OUT}/cnt_main.npy")[:V.N]
Xc, _ = V.control()
mats = {"control": scale(Xc)[0]}
for w, wn in enumerate(WIN):
    n = cm[:, w].astype(np.float64)
    r, z = forms(V.off, V.M, n)
    mats[f"min:{wn}"] = scale(np.hstack([Xc, np.stack([z, V.cold * (n >= 1)], 1)]))[0]
    # MINB (declared 15:2xZ, information): the same plus the interactions with the pool score b
    mats[f"minb:{wn}"] = scale(np.hstack([Xc, np.stack([(n >= 1) * V.b, V.cold * (n >= 1) * V.b, z, V.cold * (n >= 1)], 1)]))[0]
out, th = nested_scores(V, mats, verbose=True, workers=12, use_nb=True)
ref = np.load(f"{OUT}/s3_perfeed.npz")
print("control vs s3: max per-feed abs diff", max(float(np.abs(out["control"][i] - ref[f"control_seed{s}"]).max()) for i, s in enumerate(FOLD_SEEDS)),
      "| mean diff", [float((out["control"][i] - ref[f"control_seed{s}"]).mean()) for i, s in enumerate(FOLD_SEEDS)])
res = {}
print("\n=== MIN arms (information): control + [z_w, cs_w]; gain vs control, paired per-feed SE")
for k in mats:
    if k == "control":
        continue
    d, g, se, z = gain(out[k], out["control"])
    pub, priv = wstat(d, W["public"]), wstat(d, W["private"])
    tm = np.mean(th[k], 0)
    res[k] = dict(gain=g, se=se, z=z, pmix_public=pub, pmix_private=priv, coef_z=float(tm[-2]), coef_cs=float(tm[-1]))
    print(f"{k:8s} GAIN {g:+.5f} +- {se:.5f} (z {z:+.2f}) | pmix public {pub[0]:+.5f}+-{pub[1]:.5f} private {priv[0]:+.5f}+-{priv[1]:.5f} | mean coef z_w {tm[-2]:+.4f} cs_w {tm[-1]:+.4f}")
# convergence check on fold seed 0: maxiter 300 (declared) vs 5000
print("\n=== CONV check, fold seed 0: declared maxiter 300 vs 5000 (same objective)")
fold = V.folds(0)
full = {"control": mats["control"]}
for arm in ("A", "B", "ALL", "L"):
    full[arm] = scale(V.arm_matrix(cm, arm)[0])[0]
sc = {(k, mi): np.zeros(V.N) for k in full for mi in (300, 5000)}
info = {}
from concurrent.futures import ThreadPoolExecutor
fits = [Fitter(V, fold != kf) for kf in range(NFOLD)]


def cjob(a):
    kf, k, mi = a
    t, r = fits[kf].fit(full[k], maxiter=mi, use_nb=True)
    return kf, k, mi, t, r.nit, float(r.fun), float(np.abs(r.jac).max())


with ThreadPoolExecutor(12) as ex:
    for kf, k, mi, t, nit, fun, gmax in ex.map(cjob, [(kf, k, mi) for kf in range(NFOLD) for k in full for mi in (300, 5000)]):
        te_rows = (fold == kf)[V.fid]
        sc[(k, mi)][te_rows] = t[0] * V.b[te_rows] + full[k][te_rows] @ t[1:]
        info.setdefault((k, mi), []).append((nit, fun, gmax))
nd = {key: ndcg_per_feed(s, V.y, V.off, V.M) for key, s in sc.items()}
res["conv"] = {}
for k in full:
    line = f"{k:8s}"
    for mi in (300, 5000):
        nit = [x[0] for x in info[(k, mi)]]; fun = np.mean([x[1] for x in info[(k, mi)]]); gmax = max(x[2] for x in info[(k, mi)])
        line += f" | maxiter {mi}: NDCG {nd[(k, mi)].mean():.5f} nit {min(nit)}-{max(nit)} mean loss {fun:.3f} max|grad| {gmax:.2e}"
    if k != "control":
        g3 = (nd[(k, 300)] - nd[("control", 300)]).mean(); g5 = (nd[(k, 5000)] - nd[("control", 5000)]).mean()
        line += f" | gain vs control: {g3:+.5f} (300) {g5:+.5f} (5000)"
        res["conv"][k] = [float(g3), float(g5)]
    print(line, flush=True)
# in-sample ceiling (information; optimistic by construction): fit on ALL val feeds, score the same feeds
print("\n=== IN-SAMPLE ceiling (fit and score on all 14,423 val feeds; optimistic upper bound, NOT a result)")
fit_all = Fitter(V, np.ones(V.nf, bool))
ins = {}
allm = {**full, **{k2: v for k2, v in mats.items() if k2 != "control"}}


def ijob(k):
    t, r = fit_all.fit(allm[k], maxiter=5000, use_nb=True)
    return k, ndcg_per_feed(t[0] * V.b + allm[k] @ t[1:], V.y, V.off, V.M)


with ThreadPoolExecutor(12) as ex:
    for k, v in ex.map(ijob, list(allm)):
        ins[k] = v
res["in_sample"] = {}
for k in ins:
    if k == "control":
        print(f"control in-sample NDCG {ins[k].mean():.5f} (raw pool {V.base_ndcg.mean():.5f})")
        continue
    d = ins[k] - ins["control"]
    res["in_sample"][k] = [float(d.mean()), float(d.std() / np.sqrt(len(d)))]
    print(f"{k:8s} in-sample gain vs in-sample control {d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f}")
json.dump(res, open(f"{OUT}/s3c_min.json", "w"), indent=1)
print(f"DONE {time.time() - t0:.0f}s")
