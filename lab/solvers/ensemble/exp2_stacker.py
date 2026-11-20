"""(2) Small stackers over family scores with nested user-fold CV (outer folds = nn/finetune.py GroupKFold(5) by user).

Stackers (all pre-declared, nothing tuned on the outer folds):
  LGB-narrow   LightGBM lambdarank over [z F5_ft, z F5_to, rank01 of both, M, H, user0, cold, cold_share]
  LGB-wide     + z of the other families (F4, F5cd, V13, F3, pre_final, pre2_final, xgb_x300_luo, cb_tr300k_luo) and rank01 of v11
  LGB-res      LGB-wide as a residual model: init_score = 2 * v11 z-mix (the plain mix is the starting point)
  LIN-narrow   listwise softmax-CE linear model on [F5_ft, F5_to] x [1, M>=50, cold_share, user0]
  LIN-wide     listwise linear model on all family z's (main effects only), L2 1e-3
LightGBM: num_leaves 7, max_depth 3, min_data_in_leaf 1000, lambda_l2 10, lr 0.05, ff 0.8, bagging 0.8, <= 600 rounds;
inner protocol per outer fold k: 4 inner models, each trains on 3 of the 4 non-k folds and early-stops on the 4th
(lambdarank NDCG over the full list); outer-fold prediction = mean of the 4 inner models' per-feed z.
Optimism = inner early-stopping NDCG - outer NDCG (both vs v11 on the same feeds).
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import ROOT, Split, fmt_slices, slices  # noqa: E402
from exp3_diversity import fam_scores  # noqa: E402

SCR = "/tmp/scratchpad/solvers/ensemble"
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[300], lambdarank_truncation_level=300,
              label_gain=[0.0, 1.0], num_leaves=7, max_depth=3, min_data_in_leaf=1000, lambda_l2=10.0,
              learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=16,
              verbose=-1, seed=7, deterministic=True, force_row_wise=True)
OTHER = ["F4_ft", "F4_to", "F5cd_ft", "F5cd_to", "V13_ft", "V13_to", "F3_ft", "F3_to", "pre_final", "pre2_final",
         "xgb_x300_luo", "cb_tr300k_luo"]


def build(S, fv):
    v11 = 0.5 * fv["F5_ft"] + 0.5 * fv["F5_to"]
    meta = np.stack([S.M[S.fid], S.H[S.fid], (S.user == 0)[S.fid], S.cold, S.cold_share[S.fid]], 1).astype(np.float32)
    narrow = np.concatenate([np.stack([fv["F5_ft"], fv["F5_to"], S.rank01(fv["F5_ft"]), S.rank01(fv["F5_to"])], 1), meta], 1)
    wide = np.concatenate([narrow, np.stack([fv[k] for k in OTHER] + [S.rank01(v11)], 1)], 1)
    return v11, narrow.astype(np.float32), wide.astype(np.float32)


def lgb_nested(S, X, init=None, tag=""):
    oof = np.zeros(S.N)
    inner_gain = []
    rows_of = lambda m: np.isin(S.fid, np.where(m)[0])  # noqa: E731
    for k in range(5):
        te_r = rows_of(S.fold == k)
        zsum = np.zeros(te_r.sum())
        others = [j for j in range(5) if j != k]
        for j in others:
            tr_f = np.isin(S.fold, [o for o in others if o != j])
            va_f = S.fold == j
            tr_r, va_r = rows_of(tr_f), rows_of(va_f)
            dtr = lgb.Dataset(X[tr_r], S.y[tr_r], group=S.M[tr_f], init_score=None if init is None else init[tr_r],
                              free_raw_data=False)
            dva = lgb.Dataset(X[va_r], S.y[va_r], group=S.M[va_f], init_score=None if init is None else init[va_r],
                              reference=dtr)
            bst = lgb.train(PARAMS, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            pv = bst.predict(X[va_r], num_iteration=bst.best_iteration) + (0 if init is None else init[va_r])
            # inner-fold NDCG vs v11 on the early-stopping fold (optimistic: the fold chose the iteration)
            ndcg_in = _sub_ndcg(S, va_f, pv)
            inner_gain.append((ndcg_in, bst.best_iteration))
            pt = bst.predict(X[te_r], num_iteration=bst.best_iteration) + (0 if init is None else init[te_r])
            zsum += _sub_z(S, S.fold == k, pt)
        oof[te_r] = zsum / 4
        print(f"   {tag} outer fold {k}: inner (ndcg, best_iter) {[(round(a, 5), b) for a, b in inner_gain[-4:]]}", flush=True)
    return oof, inner_gain


def _sub_ndcg(S, fmask, s):
    from metrics import ndcg_per_feed
    M = S.M[fmask]
    off = np.concatenate([[0], np.cumsum(M)[:-1]])
    rows = np.isin(S.fid, np.where(fmask)[0])
    return float(ndcg_per_feed(s, S.y[rows], off, M).mean())


def _sub_z(S, fmask, s):
    M = S.M[fmask]
    fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


def lin_fit(S, X, rows_mask, l2=1e-3):
    """listwise softmax CE with target y/sum(y); feeds without positives are skipped."""
    fm = rows_mask
    fids = np.where(fm)[0]
    r = np.isin(S.fid, fids)
    Xr, yr, fr = X[r], S.y[r], S.fid[r]
    _, fr = np.unique(fr, return_inverse=True)
    nf = fr.max() + 1
    npos = np.bincount(fr, yr, nf)
    keep = npos[fr] > 0
    Xr, yr, fr = Xr[keep], yr[keep], fr[keep]
    _, fr = np.unique(fr, return_inverse=True)
    nf = fr.max() + 1
    t = yr / np.bincount(fr, yr, nf)[fr]

    def f(w):
        s = Xr @ w
        mx = np.full(nf, -np.inf)
        np.maximum.at(mx, fr, s)
        e = np.exp(s - mx[fr])
        Z = np.bincount(fr, e, nf)
        lse = np.log(Z) + mx
        loss = -(t * (s - lse[fr])).sum() / nf + l2 * (w @ w)
        pr = e / Z[fr]
        g = Xr.T @ (pr - t) / nf + 2 * l2 * w
        return loss, g

    w0 = np.zeros(X.shape[1])
    w0[:2] = 1.0
    res = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x


def lin_nested(S, X, tag):
    oof = np.zeros(S.N)
    W = []
    for k in range(5):
        w = lin_fit(S, X, S.fold != k)
        W.append(w)
        r = np.isin(S.fid, np.where(S.fold == k)[0])
        oof[r] = X[r] @ w
    w_all = lin_fit(S, X, np.ones(S.n, bool))
    print(f"   {tag} weights per fold:\n" + "\n".join("     " + " ".join(f"{x:+.3f}" for x in w) for w in W))
    return oof, W, w_all


def main():
    t0 = time.time()
    S = Split("val")
    fv, _ = fam_scores(S, "val")
    v11, Xn, Xw = build(S, fv)
    base = S.per_feed(v11)
    print(f"v11 {base.mean():.5f}  features narrow {Xn.shape} wide {Xw.shape}  {time.time() - t0:.0f}s", flush=True)
    rep = {}

    def report(name, s, extra=None):
        p = S.per_feed(s)
        sl = slices(S, p - base)
        rep[name] = {"ndcg": float(p.mean()), "delta": {a: [float(x) for x in b] for a, b in sl.items()}}
        if extra:
            rep[name].update(extra)
        print(f"{name:14s} {p.mean():.5f}  {fmt_slices(sl)}", flush=True)
        return p

    # pre-declared plain mixes for reference
    nnfam = {k: 0.5 * fv[f"{k}_ft"] + 0.5 * fv[f"{k}_to"] for k in ["F5", "F4", "F5cd", "V13", "F3"]}
    report("eq_F5_F4", S.z(nnfam["F5"]) + S.z(nnfam["F4"]))
    report("eq_F5_F5cd", S.z(nnfam["F5"]) + S.z(nnfam["F5cd"]))
    report("eq_F5_F5cd_V13", S.z(nnfam["F5"]) + S.z(nnfam["F5cd"]) + S.z(nnfam["V13"]))
    report("eq_F5_F4_F5cd_V13", sum(S.z(nnfam[k]) for k in ["F5", "F4", "F5cd", "V13"]))
    # linear listwise stackers
    seg = np.stack([np.ones(S.N), (S.M >= 50)[S.fid].astype(float), S.cold_share[S.fid] - S.cold_share.mean(),
                    (S.user == 0)[S.fid].astype(float)], 1)
    Xln = np.concatenate([fv["F5_ft"][:, None] * seg, fv["F5_to"][:, None] * seg], 1)
    Xln = Xln[:, [0, 4, 1, 2, 3, 5, 6, 7]]  # F5_ft, F5_to first (w0 starts at 1,1)
    oof, W, wall = lin_nested(S, Xln, "LIN-narrow")
    report("LIN-narrow", oof, {"w_folds": [w.tolist() for w in W], "w_all": wall.tolist()})
    report("LIN-narrow-ins", Xln @ wall)
    Xlw = np.stack([fv["F5_ft"], fv["F5_to"]] + [fv[k] for k in OTHER], 1)
    oof, W, wall = lin_nested(S, Xlw, "LIN-wide")
    report("LIN-wide", oof, {"w_folds": [w.tolist() for w in W], "w_all": wall.tolist(), "cols": ["F5_ft", "F5_to"] + OTHER})
    report("LIN-wide-ins", Xlw @ wall)
    # LightGBM stackers
    for name, X, init in [("LGB-narrow", Xn, None), ("LGB-wide", Xw, None), ("LGB-res", Xw, 2.0 * v11)]:
        oof, inner = lgb_nested(S, X, init, name)
        # inner optimism: inner ES-fold NDCG vs v11 on the same fold
        inner_d = []
        it = 0
        for k in range(5):
            for j in [o for o in range(5) if o != k]:
                inner_d.append(inner[it][0] - base[S.fold == j].mean())
                it += 1
        report(name, oof, {"inner_es_delta_mean": float(np.mean(inner_d)), "best_iters": [b for _, b in inner]})
        print(f"   {name}: inner early-stopping-fold delta vs v11 (optimistic) {np.mean(inner_d):+.5f}", flush=True)
        np.save(f"{SCR}/{name}_oof.npy", oof.astype(np.float32))
    json.dump(rep, open(f"{ROOT}/solvers/out/ensemble/exp2.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
