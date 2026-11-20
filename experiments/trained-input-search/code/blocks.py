# Automatic feature search, stage 2: blocks of the best features. Selection, model fit and the choice of
# hyper-parameters (with 'do nothing' as an option) all happen inside one half of the val users; the other half
# is scored. A block of noise features and a planted interaction of known size go through the same code.
# Run as (docstring): blocks.py <tag> [block names...]; FORCE=1 removes the do-nothing option.
# It wrote blocks_<tag>.json. Best block: +0.00006 +- 0.00016 on val (lab/reports/AUTOFE_1114.md).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/autofe/blocks.py
"""Stage 2: nested block tests. Feature selection (by the stage-1 inner-CV gain of the FITTING half only), model
fit and hyper-parameter choice (rounds / ridge lambda, incl. the 'do nothing' option) all happen inside the fitting
user half; the other half is scored. Holdout: selection + fit on full val, applied to the 30K in-period holdout.
usage: blocks.py <tag> [block names...]"""
import sys, time, json, glob
import numpy as np, lightgbm as lgb
from scipy.optimize import minimize
from lib import *
from lib import _rk, _z

import os
FORCE = bool(int(os.environ.get("FORCE", "0")))  # 1 = no do-nothing option: GBM 100 rounds, ridge lambda 0.3
tag = sys.argv[1]; only = set(sys.argv[2:])
load_stats()
V = DS("val"); Hd = DS("hold")
S = [json.loads(l) for f in sorted(glob.glob(str(OUT / f"{SPREF}_*.jsonl"))) if "screen_test" not in f for l in open(f)]
S = [r for r in S if not r.get("const")]
print("screened", len(S), flush=True)
T = 1.0
hA = V.half == 0
base_v = V.ndcg(V.zb); base_h = Hd.ndcg(Hd.zb)
FITS = {"A": hA, "B": ~hA, "F": np.ones(V.nf, bool)}


def top(fit, k, fams=None, ctrl=False):
    c = [r for r in S if (r["fam"] == "CTRL") == ctrl and (fams is None or r["fam"] in fams)]
    c.sort(key=lambda r: -max(r[f"cv1_{fit}"], r[f"cv2_{fit}"]))
    return [r["name"] for r in c[:k]]


def toplin(fit, k):
    """alternative selection rule: in-half NDCG gain of the best linear tilt (fields linA / linB / lin_ins)"""
    key = {"A": "linA", "B": "linB", "F": "lin_ins"}[fit]
    c = [r for r in S if r["fam"] != "CTRL"]
    c.sort(key=lambda r: -r.get(key, 0.0))
    return [r["name"] for r in c[:k]]


BLOCKS = {"RAW": lambda fit: list(BASE), "CTRL40": lambda fit: top(fit, 40, ctrl=True),
          "LIN40": lambda fit: toplin(fit, 40), "LIN10": lambda fit: toplin(fit, 10),
          "ALL40": lambda fit: top(fit, 40), "ALL100": lambda fit: top(fit, 100), "ALL10": lambda fit: top(fit, 10)}
for fam in ("A_feedrel", "B_mono", "C_pair", "C_ctx", "D_catgrp", "E_hist", "F_emb", "C2_pair", "G_efx", "C3_pair", "D2_grp"):
    BLOCKS[fam + "20"] = (lambda fam: lambda fit: top(fit, 20, {fam}))(fam)
BLOCKS["EF_all"] = lambda fit: [r["name"] for r in S if r["fam"] in ("E_hist", "F_emb", "E2_hist", "H_het")]
if only:
    BLOCKS = {k: v for k, v in BLOCKS.items() if k in only}
sel = {b: {fit: fn(fit) for fit in FITS} for b, fn in BLOCKS.items()}
need = set(n for b in sel.values() for l in b.values() for n in l)
gen_names = need - set(BASE)
print("blocks", {b: [len(v) for v in d.values()] for b, d in sel.items()}, "distinct generated", len(gen_names), flush=True)


def materialise(D):
    X = {n: D.get(n) for n in need if n in BASE}
    rng = np.random.default_rng(1)
    for fam, n, x in gen(D, want=gen_names):
        X[n] = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return X


t0 = time.time()
XV = materialise(V); XH = materialise(Hd)
# controls are regenerated with the SAME generator state as in screen.py (ctrl seed 1)
ctrl_names = [n for n in need if n.startswith(("noise", "perm"))]
if ctrl_names:
    rng = np.random.default_rng(1)
    for i in range(400):
        if i % 2 == 0:
            a, b_ = rng.standard_normal(V.n).astype(np.float32), rng.standard_normal(Hd.n).astype(np.float32)
            nm = f"noisectrl{i}"
        else:
            b = BASE[(i // 2) % len(BASE)]
            a = V.get(b)[np.argsort(V.fid + rng.random(V.n), kind="stable")]
            b_ = Hd.get(b)[np.argsort(Hd.fid + rng.random(Hd.n), kind="stable")]
            nm = f"permctrl{i}({b})"
        if nm in need:
            XV[nm] = a; XH[nm] = b_
missing = need - set(XV)
assert not missing, list(missing)[:5]
print("materialised", len(XV), f"{time.time() - t0:.0f}s", flush=True)
zrkV = _rk(V.off, V.M, V.zb); zrkH = _rk(Hd.off, Hd.M, Hd.zb)
def softmax_obj(preds, ds):
    """exact listwise softmax CE (multi-positive) on init_score + trees; LightGBM's rank_xendcg is NOT usable as a
    residual objective here: its randomised targets pull a sharp init score back to a log(3) logit gap (-0.02 NDCG)"""
    Mg = ds.get_group().astype(np.int64); og = np.concatenate([[0], np.cumsum(Mg)[:-1]]).astype(np.int64)
    g, h, _ = softmax_gh(og, Mg, preds.astype(np.float32), ds.get_label().astype(np.float32))
    return -g.astype(np.float64), np.maximum(h, 1e-6).astype(np.float64)


P = dict(objective=softmax_obj, learning_rate=0.03, num_leaves=7, min_data_in_leaf=3000, lambda_l2=100.0,
         feature_fraction=0.5, max_bin=31, num_threads=12, verbose=-1, seed=1, min_sum_hessian_in_leaf=20.0)
ROUNDS = [25, 50, 100, 200]


def rows_of(D, fmask):
    return np.repeat(fmask, D.M)


def mat(X, D, names, zrk):
    return np.column_stack([D.zb, zrk] + [X[n] for n in names])


def gbm_fit(Xm, fmask, n):
    r = rows_of(V, fmask)
    ds = lgb.Dataset(Xm[r], label=V.y[r], group=V.M[fmask], init_score=(T * V.zb[r]).astype(np.float64))
    return lgb.train(P, ds, num_boost_round=n)


def gbm_choose(Xm, fit_mask):
    """inner split of the fitting feeds: train on inner==0, pick rounds on inner==1 (0 = do nothing)"""
    if FORCE:
        return 100
    tr = fit_mask & (V.inner == 0); ev = fit_mask & (V.inner == 1)
    m = gbm_fit(Xm, tr, max(ROUNDS))
    best = (0.0, 0)
    for n in ROUNDS:
        s = T * V.zb + m.predict(Xm, num_iteration=n, num_threads=12)
        d = (V.ndcg(s) - base_v)[ev].mean()
        if d > best[0]:
            best = (d, n)
    return best[1]


def ce(w, Z, D, fmask_rows, lam, nf):
    s = (T * D.zb + Z @ w.astype(np.float32)).astype(np.float32)
    g, h, loss = softmax_gh(D.off, D.M, s, D.y)
    g = g * fmask_rows
    L = loss[fmask_rows_f[0]].sum() / nf + 0.5 * lam * (w ** 2).sum()
    return L, -(Z.T @ g).astype(np.float64) / nf + lam * w


fmask_rows_f = [None]


def ridge_fit(Z, fmask, lam):
    fmask_rows_f[0] = fmask
    r = rows_of(V, fmask).astype(np.float32)
    nf = float((fmask & V.pos_any).sum())
    res = minimize(ce, np.zeros(Z.shape[1]), args=(Z, V, r, lam, nf), jac=True, method="L-BFGS-B", options=dict(maxiter=60))
    return res.x


def ridge_choose(Z, fit_mask):
    if FORCE:
        return 0.3
    tr = fit_mask & (V.inner == 0); ev = fit_mask & (V.inner == 1)
    best = (0.0, None)
    for lam in (3.0, 0.3, 0.03):
        w = ridge_fit(Z, tr, lam)
        d = (V.ndcg(T * V.zb + Z @ w.astype(np.float32)) - base_v)[ev].mean()
        if d > best[0]:
            best = (d, lam)
    return best[1]


def zmat(X, D, names):
    return np.column_stack([_z(D.off, D.M, X[n]) for n in names]).astype(np.float32)


def rep(d):
    se = d.std() / np.sqrt(len(d))
    return float(d.mean()), float(se), float(d.mean() / se) if se > 0 else 0.0


res = {}
for b in BLOCKS:
    t0 = time.time()
    out = {"n": {f: len(sel[b][f]) for f in FITS}}
    for kind in ("gbm", "ridge"):
        if kind == "gbm" and b.startswith("B_mono"):
            continue  # monotone transforms are invisible to trees
        d = np.zeros(V.nf); ch = {}
        for fit, other in (("A", "B"), ("B", "A")):
            names = sel[b][fit]
            if kind == "gbm":
                Xm = mat(XV, V, names, zrkV)
                n = gbm_choose(Xm, FITS[fit]); ch[fit] = n
                if n:
                    s = T * V.zb + gbm_fit(Xm, FITS[fit], n).predict(Xm, num_threads=12)
                    d[FITS[other]] = (V.ndcg(s) - base_v)[FITS[other]]
            else:
                Z = zmat(XV, V, names)
                lam = ridge_choose(Z, FITS[fit]); ch[fit] = lam
                if lam:
                    w = ridge_fit(Z, FITS[fit], lam)
                    d[FITS[other]] = (V.ndcg(T * V.zb + Z @ w.astype(np.float32)) - base_v)[FITS[other]]
        out[kind + "_val"] = rep(d)
        # holdout: full-val fit; hyper-parameter = the more conservative of the two half choices
        names = sel[b]["F"]
        dh = np.zeros(Hd.nf)
        if kind == "gbm":
            n = min(ch["A"], ch["B"]) if min(ch["A"], ch["B"]) > 0 else max(ch["A"], ch["B"])
            if n:
                Xm = mat(XV, V, names, zrkV)
                m = gbm_fit(Xm, FITS["F"], n)
                dh = Hd.ndcg(T * Hd.zb + m.predict(mat(XH, Hd, names, zrkH), num_threads=12)) - base_h
                imp = m.feature_importance("gain"); o = np.argsort(-imp)[:8]
                out["gbm_top"] = [(["zb", "zb_rk"] + names)[i] for i in o]
        else:
            ls = [l for l in ch.values() if l]
            lam = max(ls) if ls else None
            if lam:
                w = ridge_fit(zmat(XV, V, names), FITS["F"], lam)
                dh = Hd.ndcg(T * Hd.zb + zmat(XH, Hd, names) @ w.astype(np.float32)) - base_h
                o = np.argsort(-np.abs(w))[:8]
                out["ridge_top"] = [(names[i], round(float(w[i]), 4)) for i in o]
        out[kind + "_hold"] = rep(dh); out[kind + "_choice"] = ch
    res[b] = out
    print(b, json.dumps(out), f"{time.time() - t0:.0f}s", flush=True)
    json.dump({"res": res, "sel": {k: sel[k] for k in res}}, open(OUT / f"blocks_{SPREF}_{tag}.json", "w"), indent=1)
