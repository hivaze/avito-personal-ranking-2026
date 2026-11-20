"""docs_rank: verification of CatBoost learning-to-rank documentation claims on the INSTALLED version (1.2.10), CPU only.

Synthetic data only (feeds of 10..300 candidates, binary clicks, about 1.7 positives per feed, a strong 'teacher' score).
No project data is read. Every check is recorded as PASS / FAIL (+ the exact error text) / NOEFFECT (ran, predictions are
bit-identical to the reference run of the same loss = the parameter was ignored).

usage: CUDA_VISIBLE_DEVICES="" python verify_synth.py [section ...]      -> out/verify_<section>.json + stdout table
"""
import json, os, sys, time, traceback, warnings
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only: run with CUDA_VISIBLE_DEVICES=\"\""
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoost, CatBoostRanker, Pool, sum_models
from catboost.utils import eval_metric

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
TH = 4
IT = 40
RES = []

# ---- thread cap: every code path that defaults to thread_count=-1 (= all cores) is pinned to TH threads
_op = CatBoost._predict
_osp = CatBoost._staged_predict
_oem = CatBoost.eval_metrics
_oev = eval_metric
_OPool = Pool


def _predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, parent_method_name, task_type="CPU"):
    return _op(self, data, prediction_type, ntree_start, ntree_end, TH if thread_count == -1 else thread_count, verbose,
               parent_method_name, task_type)


def _staged(self, data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose, parent_method_name):
    return _osp(self, data, prediction_type, ntree_start, ntree_end, eval_period, TH if thread_count == -1 else thread_count,
                verbose, parent_method_name)


def _em(self, data, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, **kw):
    return _oem(self, data, metrics, ntree_start, ntree_end, eval_period, TH if thread_count == -1 else thread_count, **kw)


CatBoost._predict = _predict
CatBoost._staged_predict = _staged
CatBoost.eval_metrics = _em


def eval_metric(label, approx, metric, **kw):  # noqa: F811
    kw.setdefault("thread_count", TH)
    return _oev(label, approx, metric, **kw)


def Pool(*a, **kw):  # noqa: F811
    kw.setdefault("thread_count", TH)
    return _OPool(*a, **kw)


# ----------------------------------------------------------------------------------------------------------- data
def make(n_groups, seed, d=12, emb_d=16, min_m=10, max_m=300):
    r = np.random.default_rng(seed)
    M = np.clip(np.round(np.exp(r.normal(4.1, 0.75, n_groups))), min_m, max_m).astype(np.int64)
    n = int(M.sum()); gid = np.repeat(np.arange(n_groups), M); off = np.r_[0, np.cumsum(M)[:-1]]
    X = r.normal(size=(n, d)).astype(np.float32)
    rt = np.random.default_rng(123)                                     # the TRUE function is the same for every split
    w = rt.normal(size=d) * (np.arange(d) < 8); ceff = rt.normal(size=50) * 0.5; edir = rt.normal(size=emb_d)
    cat = r.integers(0, 50, n)
    u = X @ w * 0.35 + 0.5 * X[:, 0] * X[:, 1] + 0.4 * np.abs(X[:, 2]) + ceff[cat]
    u = u / u.std()
    E = (r.normal(size=(n, emb_d)) + 0.6 * u[:, None] * edir[None, :]).astype(np.float32)
    teacher = (u + 0.3 * r.normal(size=n)).astype(np.float32)          # teacher logit (noisy view of the truth)
    y = np.zeros(n, np.float32)
    for g in range(n_groups):                                           # about 1.7 positives per feed, at least 1
        s = slice(off[g], off[g] + M[g]); p = np.exp(3.0 * (u[s] - u[s].max())); p /= p.sum()
        k = min(M[g] - 1, 1 + r.poisson(0.7))
        y[off[g] + r.choice(M[g], size=k, replace=False, p=p)] = 1.0
    return dict(X=X, cat=cat, E=E, y=y, t=teacher, gid=gid, M=M, off=off, n=n, u=u)


def ndcg_pess(score, y, off, M):
    """project metric: full-list NDCG, binary gain, log2 discount, ties scored pessimistically. Per feed."""
    out = np.zeros(len(M))
    for g in range(len(M)):
        s = score[off[g]:off[g] + M[g]]; t = y[off[g]:off[g] + M[g]]
        o = np.lexsort((t, -s))                                         # by score desc, ties: lower label first
        disc = 1.0 / np.log2(np.arange(M[g]) + 2.0)
        dcg = (t[o] * disc).sum(); idcg = (np.sort(t)[::-1] * disc).sum()
        out[g] = dcg / idcg if idcg > 0 else 0.0
    return out


def soft_q(t, off, M, T=1.0):
    q = np.zeros_like(t, dtype=np.float64)
    for g in range(len(M)):
        s = slice(off[g], off[g] + M[g]); e = np.exp((t[s] - t[s].max()) / T); q[s] = e / e.sum()
    return q.astype(np.float32)


def centre(t, off, M):
    return (t - np.repeat(np.add.reduceat(t.astype(np.float64), off) / M, M)).astype(np.float32)


TR = make(600, 1); EV = make(200, 2)
NAMES = [f"x{i}" for i in range(TR["X"].shape[1])]


def frame(D, cat=False, emb=False):
    df = pd.DataFrame(D["X"], columns=NAMES)
    if cat:
        df["c0"] = D["cat"].astype(np.int32)
    if emb:
        df["e0"] = list(D["E"])
    return df


def pool(D, label="y", cat=False, emb=False, **kw):
    lab = D[label] if isinstance(label, str) else label
    return Pool(frame(D, cat, emb), label=lab, group_id=D["gid"], cat_features=["c0"] if cat else None,
                embedding_features=["e0"] if emb else None, **kw)


BASE = dict(iterations=IT, learning_rate=0.2, depth=6, thread_count=TH, random_seed=0, verbose=False,
            allow_writing_files=False, use_best_model=False)
_REF = {}


def fit(loss, label="y", cat=False, emb=False, pool_kw=None, ev_kw=None, cls=CatBoostRanker, fit_kw=None, **params):
    p = dict(BASE); p.update(params)
    m = CatBoost(dict(p, loss_function=loss)) if cls is CatBoost else cls(loss_function=loss, **p)
    ptr = pool(TR, label, cat, emb, **(pool_kw or {}))
    pev = pool(EV, "y", cat, emb, **(ev_kw or {}))
    m.fit(ptr, eval_set=pev, **(fit_kw or {}))
    return m, np.asarray(m.predict(pool(EV, "y", cat, emb)), np.float64)


def quality(pred):
    return float(ndcg_pess(pred, EV["y"], EV["off"], EV["M"]).mean())


def ref(loss, **kw):
    key = (loss, json.dumps(kw, sort_keys=True, default=str))
    if key not in _REF:
        _REF[key] = fit(loss, **kw)[1]
    return _REF[key]


def rec(section, name, status, detail="", sec=0.0):
    RES.append(dict(section=section, name=name, status=status, detail=detail, sec=round(sec, 2)))
    print(f"[{section}] {status:9s} {name}  | {detail}  ({sec:.1f}s)", flush=True)


def check(section, name, fn):
    t0 = time.time()
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            d = fn()
        status = "PASS"
        if isinstance(d, tuple):
            status, d = d
        ws = "; ".join(sorted({str(w.message)[:160] for w in wl}))
        rec(section, name, status, (d or "") + (f" [warning: {ws}]" if ws else ""), time.time() - t0)
    except Exception as e:  # noqa: BLE001
        msg = " ".join(str(e).split())
        rec(section, name, "FAIL", f"{type(e).__name__}: {msg[:600]}", time.time() - t0)


def run_loss(loss, effect_vs=None, ref_kw=None, **kw):
    """fit; report NDCG; if effect_vs is given, compare predictions to the reference loss string (NOEFFECT if identical)."""
    def f():
        m, p = fit(loss, **kw)
        q = quality(p); sd = float(np.std(p))
        d = f"eval NDCG {q:.4f}, pred sd {sd:.3g}, trees {m.tree_count_}"
        if sd < 1e-12:
            return "DEGENER", d + " (constant predictions)"
        if effect_vs is not None:
            r = ref(effect_vs, **(ref_kw or {}))
            if np.array_equal(r, p):
                return "NOEFFECT", d + f" (identical to '{effect_vs}')"
            d += f"; differs from '{effect_vs}' (corr {np.corrcoef(r, p)[0, 1]:.3f})"
        return d
    return f


RANDOM_Q = quality(np.random.default_rng(0).normal(size=EV["n"]))
TEACH_Q = quality(EV["t"].astype(np.float64))


# ------------------------------------------------------------------------------------------------------- sections
def sec_objectives():
    S = "a.objectives"
    rec(S, "reference levels", "INFO", f"random scores NDCG {RANDOM_Q:.4f}; teacher score NDCG {TEACH_Q:.4f}; "
        f"train {len(TR['M'])} feeds / {TR['n']} rows, eval {len(EV['M'])} feeds / {EV['n']} rows, "
        f"positives per feed {TR['y'].sum() / len(TR['M']):.2f}, mean M {TR['M'].mean():.1f}, max M {TR['M'].max()}")
    for loss in ["YetiRank", "YetiRankPairwise", "QuerySoftMax", "QueryRMSE", "PairLogit", "PairLogitPairwise",
                 "LambdaMart", "StochasticRank:metric=NDCG", "StochasticFilter", "GroupQuantile",
                 "QueryCrossEntropy", "StochasticRank"]:
        check(S, f"{loss} (defaults, CPU, binary labels)", run_loss(loss))
    check(S, "unknown loss parameter YetiRank:bogus=1 (rejected, or silently ignored = NOEFFECT?)", run_loss("YetiRank:bogus=1", effect_vs="YetiRank"))
    check(S, "unknown loss parameter QuerySoftMax:temperature=2 (rejected, or silently ignored?)", run_loss("QuerySoftMax:temperature=2", effect_vs="QuerySoftMax"))
    check(S, "unknown TRAINING parameter bogus_param=1", run_loss("YetiRank", bogus_param=1))
    for loss in ["YetiRank", "YetiRank:mode=NDCG", "YetiRankPairwise", "QuerySoftMax", "QueryRMSE", "PairLogit", "LambdaMart", "StochasticRank:metric=NDCG"]:
        def f(loss=loss):
            a = fit(loss)[1]; b = fit(loss)[1]; c = fit(loss, random_seed=1)[1]; d = fit(loss, thread_count=2)[1]
            return ("PASS" if np.array_equal(a, b) else "NONDETERM", f"same seed twice identical: {np.array_equal(a, b)}; other seed corr {np.corrcoef(a, c)[0, 1]:.3f}; "
                    f"2 threads instead of 4 identical: {np.array_equal(a, d)}")
        check(S, f"determinism of {loss} (the NOEFFECT test relies on it)", f)
    # --- YetiRank parameters
    for s in ["YetiRank:mode=Classic", "YetiRank:permutations=3", "YetiRank:permutations=30", "YetiRank:decay=0.5",
              "YetiRank:decay=0.99", "YetiRank:top=10", "YetiRank:num_neighbors=3", "YetiRank:noise=Gauss",
              "YetiRank:noise=Gauss;noise_power=0.3", "YetiRank:noise=No", "YetiRank:dcg_type=Exp",
              "YetiRank:mode=NDCG", "YetiRank:mode=DCG", "YetiRank:mode=MRR", "YetiRank:mode=MAP", "YetiRank:mode=ERR",
              "YetiRank:mode=PFound", "YetiRank:mode=NDCG;top=10", "YetiRank:mode=NDCG;top=300",
              "YetiRank:mode=NDCG;num_neighbors=3", "YetiRank:mode=NDCG;num_neighbors=-1",
              "YetiRank:mode=NDCG;num_neighbors=1000", "YetiRank:mode=NDCG;decay=0.5",
              "YetiRank:mode=NDCG;permutations=30",
              "YetiRank:mode=NDCG;dcg_type=Exp", "YetiRank:mode=NDCG;dcg_denominator=Position",
              "YetiRank:mode=NDCG;dcg_denominator=LogPosition", "YetiRank:mode=NDCG;noise=Gauss;noise_power=0.5",
              "YetiRank:mode=NDCG;noise=No", "YetiRank:mode=MRR;num_neighbors=2", "YetiRank:mode=MAP;top=20"]:
        base = "YetiRank:mode=NDCG" if "mode=NDCG;" in s else "YetiRank"
        check(S, s, run_loss(s, effect_vs=None if s in ("YetiRank:mode=NDCG",) else base))
    for s in ["YetiRankPairwise:mode=NDCG", "YetiRankPairwise:permutations=3;decay=0.9", "YetiRankPairwise:mode=MRR"]:
        check(S, s, run_loss(s, effect_vs="YetiRankPairwise"))
    # --- LambdaMart parameters
    for s in ["LambdaMart:metric=NDCG", "LambdaMart:metric=DCG", "LambdaMart:metric=MRR", "LambdaMart:metric=MAP",
              "LambdaMart:metric=ERR", "LambdaMart:sigma=2", "LambdaMart:sigma=0.5", "LambdaMart:norm=false",
              "LambdaMart:norm=0", "LambdaMart:metric=NDCG;top=10", "LambdaMart:top=10",
              "LambdaMart:metric=NDCG;type=Exp", "LambdaMart:metric=NDCG;denominator=Position",
              "LambdaMart:metric=PFound"]:
        check(S, s, run_loss(s, effect_vs="LambdaMart"))
    # --- StochasticRank parameters
    for s in ["StochasticRank:metric=DCG", "StochasticRank:metric=PFound", "StochasticRank:metric=MRR",
              "StochasticRank:metric=ERR", "StochasticRank:metric=MAP", "StochasticRank:metric=FilteredDCG",
              "StochasticRank:metric=NDCG;sigma=0.3", "StochasticRank:metric=NDCG;mu=0.1",
              "StochasticRank:metric=NDCG;num_estimations=8", "StochasticRank:metric=NDCG;top=10",
              "StochasticRank:metric=NDCG;nu=0.1", "StochasticRank:metric=NDCG;lambda=0.5",
              "StochasticRank:metric=NDCG;type=Exp", "StochasticRank:metric=PFound;decay=0.7"]:
        check(S, s, run_loss(s, effect_vs="StochasticRank:metric=NDCG"))
    # --- QuerySoftMax / QueryRMSE / PairLogit / GroupQuantile
    for s, b in [("QuerySoftMax:beta=0.5", "QuerySoftMax"), ("QuerySoftMax:beta=3", "QuerySoftMax"),
                 ("QuerySoftMax:lambda=0.5", "QuerySoftMax"), ("QuerySoftMax:lambda=0", "QuerySoftMax"),
                 ("PairLogit:max_pairs=5", "PairLogit"), ("PairLogit:max_pairs=100000", "PairLogit"),
                 ("PairLogitPairwise:max_pairs=5", "PairLogitPairwise"), ("GroupQuantile:alpha=0.9", "GroupQuantile"),
                 ("QueryCrossEntropy:alpha=0.5", None), ("StochasticFilter:sigma=2;num_estimations=4", "StochasticFilter")]:
        check(S, s, run_loss(s, effect_vs=b))
    # --- pointwise baselines through the generic class (group_id is kept for the metric)
    for s in ["Logloss", "CrossEntropy", "RMSE"]:
        check(S, f"{s} via CatBoost() with group_id in the pool (pointwise baseline)",
              run_loss(s, cls=CatBoost, eval_metric="NDCG:type=Base"))
    check(S, "Logloss via CatBoostRanker (is it allowed?)", run_loss("Logloss"))
    check(S, "RMSE via CatBoostRanker (regression loss in the ranker class)", run_loss("RMSE"))


def sec_defaults():
    S = "x.resolved_defaults"
    keys = ["leaf_estimation_method", "leaf_estimation_iterations", "leaf_estimation_backtracking", "l2_leaf_reg",
            "bootstrap_type", "subsample", "bagging_temperature", "sampling_frequency", "sampling_unit", "boosting_type",
            "random_strength", "random_score_type", "border_count", "feature_border_type", "score_function",
            "grow_policy", "depth", "min_data_in_leaf", "max_leaves", "learning_rate", "rsm", "one_hot_max_size",
            "model_size_reg", "model_shrink_rate", "model_shrink_mode", "boost_from_average", "eval_metric",
            "penalties_coefficient", "nan_mode", "use_best_model", "best_model_min_trees", "posterior_sampling",
            "max_ctr_complexity", "fold_permutation_block", "approx_on_full_history", "force_unit_auto_pair_weights",
            "pool_metainfo_options", "loss_function"]
    tab = {}
    for loss in ["YetiRank", "YetiRank:mode=NDCG", "YetiRankPairwise", "QuerySoftMax", "QueryRMSE", "PairLogit",
                 "PairLogitPairwise", "LambdaMart", "StochasticRank:metric=NDCG", "GroupQuantile"]:
        try:
            m = CatBoostRanker(loss_function=loss, iterations=3, thread_count=TH, verbose=False, allow_writing_files=False)
            m.fit(pool(TR, cat=True))
            ap = m.get_all_params()
            tab[loss] = {k: ap.get(k, "<absent>") for k in keys}
            rec(S, loss, "PASS", json.dumps(tab[loss], default=str))
        except Exception as e:  # noqa: BLE001
            rec(S, loss, "FAIL", " ".join(str(e).split())[:400])
    for loss, cls_kw in [("Logloss", {}), ("CrossEntropy", {}), ("RMSE", {})]:
        try:
            m = CatBoost(dict(loss_function=loss, iterations=3, thread_count=TH, verbose=False, allow_writing_files=False))
            m.fit(pool(TR, cat=True))
            ap = m.get_all_params()
            tab[loss] = {k: ap.get(k, "<absent>") for k in keys}
            rec(S, loss, "PASS", json.dumps(tab[loss], default=str))
        except Exception as e:  # noqa: BLE001
            rec(S, loss, "FAIL", " ".join(str(e).split())[:400])
    for gp in ["Depthwise", "Lossguide"]:
        try:
            m = CatBoostRanker(loss_function="YetiRank", grow_policy=gp, iterations=3, thread_count=TH, verbose=False,
                               allow_writing_files=False)
            m.fit(pool(TR))
            ap = m.get_all_params()
            tab[f"YetiRank/{gp}"] = {k: ap.get(k, "<absent>") for k in keys}
            rec(S, f"YetiRank grow_policy={gp}", "PASS", json.dumps(tab[f"YetiRank/{gp}"], default=str))
        except Exception as e:  # noqa: BLE001
            rec(S, f"YetiRank grow_policy={gp}", "FAIL", " ".join(str(e).split())[:400])
    json.dump(tab, open(os.path.join(OUT, "resolved_defaults.json"), "w"), indent=1, default=str)


def sec_labels():
    S = "c.labels"
    q1 = soft_q(TR["t"], TR["off"], TR["M"], 1.0)                      # teacher probability, sums to 1 per feed
    sig = (1 / (1 + np.exp(-(TR["t"] - 2.0)))).astype(np.float32)       # pointwise teacher probability in (0,1)
    logit = TR["t"]                                                     # any real, negative values included
    cen = centre(TR["t"], TR["off"], TR["M"])
    mix = (0.5 * TR["y"] / np.repeat(np.add.reduceat(TR["y"], TR["off"]), TR["M"]) + 0.5 * q1).astype(np.float32)
    graded = (TR["y"] * 3 + (TR["t"] > np.quantile(TR["t"], 0.8))).astype(np.float32)   # 0,1,3,4
    forms = [("soft prob (softmax of teacher, sums to 1 per feed)", q1), ("pointwise prob in (0,1)", sig),
             ("teacher logit (any real, negatives)", logit), ("feed-centred teacher logit", cen),
             ("0.5*y/npos + 0.5*softmax(teacher)", mix), ("graded 0/1/3/4", graded)]
    losses = ["YetiRank", "YetiRank:mode=NDCG", "YetiRank:mode=MRR", "YetiRankPairwise", "QuerySoftMax", "QueryRMSE",
              "PairLogit", "LambdaMart", "LambdaMart:metric=MRR", "StochasticRank:metric=NDCG", "StochasticRank:metric=DCG"]
    for loss in losses:
        for nm, lab in forms:
            def f(loss=loss, lab=lab):
                m, p = fit(loss, label=lab)
                # agreement with the teacher on the eval feeds (mean within-feed Spearman) and hard-label NDCG
                from scipy.stats import spearmanr
                rs = [spearmanr(p[o:o + mm], EV["t"][o:o + mm])[0] for o, mm in zip(EV["off"][:60], EV["M"][:60])]
                d = f"eval NDCG(hard clicks) {quality(p):.4f}; within-feed Spearman to teacher {np.nanmean(rs):.3f}; pred sd {np.std(p):.3g}"
                return ("DEGENER", d) if np.std(p) < 1e-12 else d
            check(S, f"{loss} <- {nm}", f)
    # pointwise losses with soft labels
    for loss, lab, nm, kw in [("CrossEntropy", sig, "pointwise prob in (0,1)", {}),
                              ("CrossEntropy", q1, "soft prob (sums to 1 per feed)", {}),
                              ("CrossEntropy", logit, "teacher logit (outside [0,1])", {}),
                              ("Logloss", sig, "pointwise prob in (0,1), no target_border", {}),
                              ("Logloss", sig, "pointwise prob in (0,1), target_border=0.5", dict(target_border=0.5)),
                              ("RMSE", logit, "teacher logit", {}), ("RMSE", cen, "feed-centred teacher logit", {}),
                              ("QueryRMSE", logit, "teacher logit (QueryRMSE removes the feed mean itself)", {})]:
        def f(loss=loss, lab=lab, kw=kw):
            m, p = fit(loss, label=lab, cls=CatBoost, **kw)
            return f"eval NDCG(hard clicks) {quality(p):.4f}; pred sd {np.std(p):.3g}"
        check(S, f"{loss} <- {nm}", f)
    # target_border with ranking losses
    for loss in ["YetiRank", "QuerySoftMax"]:
        check(S, f"{loss} + target_border=0.5 on graded labels", run_loss(loss, label=graded, target_border=0.5,
                                                                         effect_vs=loss, ref_kw=dict(label=graded)))
    # is QueryRMSE on raw logits == QueryRMSE on centred logits? (the loss removes the per-feed mean)
    def f():
        a = fit("QueryRMSE", label=logit)[1]; b = fit("QueryRMSE", label=cen)[1]
        za = centre(a.astype(np.float32), EV["off"], EV["M"]); zb = centre(b.astype(np.float32), EV["off"], EV["M"])
        return f"max |centred pred difference| {np.abs(za - zb).max():.3g}, corr {np.corrcoef(za, zb)[0, 1]:.5f}"
    check(S, "QueryRMSE: raw teacher logit vs feed-centred teacher logit give the same ranking model", f)
    # QuerySoftMax semantics: label scale = feed weight; the fitted softmax approaches the soft target
    def f():
        m, _ = fit("QuerySoftMax", label=q1, iterations=150, learning_rate=0.2)
        p = np.asarray(m.predict(pool(TR, q1)), np.float64); qm = soft_q(p.astype(np.float32), TR["off"], TR["M"], 1.0)
        kl = float(np.mean([np.sum(q1[o:o + mm] * (np.log(q1[o:o + mm] + 1e-12) - np.log(qm[o:o + mm] + 1e-12)))
                            for o, mm in zip(TR["off"], TR["M"])]))
        u = np.repeat(1.0 / TR["M"], TR["M"]).astype(np.float32)
        kl0 = float(np.mean([np.sum(q1[o:o + mm] * (np.log(q1[o:o + mm] + 1e-12) - np.log(u[o:o + mm])))
                             for o, mm in zip(TR["off"], TR["M"])]))
        return f"train KL(teacher softmax || model softmax): uniform start {kl0:.4f} -> after 150 trees {kl:.4f}"
    check(S, "QuerySoftMax with soft labels minimises the listwise cross-entropy to the soft target (KL falls)", f)
    def f():
        a = fit("QuerySoftMax", label=q1)[1]; b = fit("QuerySoftMax", label=(3.0 * q1).astype(np.float32))[1]
        return ("NOEFFECT" if np.array_equal(a, b) else "PASS",
                f"labels x3: corr {np.corrcoef(a, b)[0, 1]:.4f}, sd ratio {np.std(b) / np.std(a):.3f} (label sum acts as a feed weight)")
    check(S, "QuerySoftMax: multiplying all labels by 3 changes the fit (sum of labels = feed weight)", f)
    def f():
        lab = TR["y"].copy(); g0 = slice(TR["off"][0], TR["off"][0] + TR["M"][0]); lab[g0] = 0
        m, p = fit("QuerySoftMax", label=lab)
        return f"one all-negative feed in train: trains, eval NDCG {quality(p):.4f}"
    check(S, "QuerySoftMax with an all-zero-label feed", f)
    def f():
        lab = cen.copy()
        m, p = fit("QuerySoftMax", label=lab)
        return f"labels with negatives: eval NDCG {quality(p):.4f}, pred sd {np.std(p):.3g}"
    check(S, "QuerySoftMax with negative labels (centred logits)", f)


def sec_groups():
    S = "b.groups"
    gw = np.repeat(np.random.default_rng(3).uniform(0.2, 3.0, len(TR["M"])), TR["M"]).astype(np.float32)
    ow = np.random.default_rng(4).uniform(0.2, 3.0, TR["n"]).astype(np.float32)
    for loss in ["YetiRank", "YetiRank:mode=NDCG", "QuerySoftMax", "QueryRMSE", "LambdaMart", "StochasticRank:metric=NDCG",
                 "PairLogit", "YetiRankPairwise"]:
        check(S, f"{loss} + group_weight (constant within feed)", run_loss(loss, pool_kw=dict(group_weight=gw), effect_vs=loss))
        check(S, f"{loss} + per-object weight", run_loss(loss, pool_kw=dict(weight=ow), effect_vs=loss))
    check(S, "group_weight that varies inside a feed", run_loss("YetiRank", pool_kw=dict(group_weight=ow)))
    check(S, "weight and group_weight together", run_loss("YetiRank", pool_kw=dict(group_weight=gw, weight=ow)))
    check(S, "YetiRank:use_weights=false + group_weight", run_loss("YetiRank:use_weights=false", pool_kw=dict(group_weight=gw),
                                                                 effect_vs="YetiRank"))
    sub = (np.arange(TR["n"]) % 3).astype(np.int64)
    for loss in ["YetiRank", "QuerySoftMax", "PairLogit"]:
        check(S, f"{loss} + subgroup_id", run_loss(loss, pool_kw=dict(subgroup_id=sub), effect_vs=loss))
    # explicit pairs
    r = np.random.default_rng(5); pairs = []
    for g in range(len(TR["M"])):
        o, mm = TR["off"][g], TR["M"][g]; pos = np.where(TR["y"][o:o + mm] > 0)[0]; neg = np.where(TR["y"][o:o + mm] == 0)[0]
        for i in pos:
            for j in r.choice(neg, size=min(5, len(neg)), replace=False):
                pairs.append((o + i, o + j))
    pairs = np.asarray(pairs, np.int64); pw = r.uniform(0.5, 2.0, len(pairs)).astype(np.float32)
    for loss in ["PairLogit", "PairLogitPairwise", "YetiRank", "QuerySoftMax"]:
        check(S, f"{loss} + explicit pairs ({len(pairs)} pairs)", run_loss(loss, pool_kw=dict(pairs=pairs), effect_vs=loss))
    check(S, "PairLogit + explicit pairs + pairs_weight", run_loss("PairLogit", pool_kw=dict(pairs=pairs, pairs_weight=pw),
                                                                 effect_vs="PairLogit", ref_kw=dict(pool_kw=dict(pairs=pairs))))
    def f():
        m = CatBoostRanker(loss_function="PairLogit", **BASE)
        m.fit(Pool(frame(TR), group_id=TR["gid"], pairs=pairs))
        return f"no label at all, pairs only: trains, eval NDCG {quality(np.asarray(m.predict(pool(EV)), np.float64)):.4f}"
    check(S, "PairLogit with pairs and NO label column", f)
    def f():
        bad = pairs.copy(); bad[0, 1] = TR["off"][5]  # loser from another group
        m = CatBoostRanker(loss_function="PairLogit", **BASE); m.fit(Pool(frame(TR), label=TR["y"], group_id=TR["gid"], pairs=bad))
        return "cross-group pair accepted"
    check(S, "pair whose members are in different groups", f)
    def f():
        gid = TR["gid"].copy(); gid[:5] = 99999; gid[-5:] = 99999      # same id at both ends = not contiguous
        m = CatBoostRanker(loss_function="YetiRank", **BASE); m.fit(Pool(frame(TR), label=TR["y"], group_id=gid))
        return "non-contiguous group ids accepted (silently treated as ?)"
    check(S, "group ids not contiguous", f)
    def f():
        m = CatBoostRanker(loss_function="YetiRank", **BASE); m.fit(Pool(frame(TR), label=TR["y"]))
        return "trained without group_id"
    check(S, "YetiRank without group_id", f)
    # all-negative feeds and constant-label feeds
    def f():
        lab = TR["y"].copy()
        for g in range(0, 150):
            lab[TR["off"][g]:TR["off"][g] + TR["M"][g]] = 0
        out = []
        for loss in ["YetiRank", "YetiRank:mode=NDCG", "LambdaMart", "StochasticRank:metric=NDCG", "QuerySoftMax", "PairLogit", "QueryRMSE"]:
            m, p = fit(loss, label=lab); out.append(f"{loss} {quality(p):.4f}")
        return "150 of 600 train feeds all-negative: " + ", ".join(out)
    check(S, "all-negative feeds in train (each loss trains?)", f)
    def f():
        lab = np.zeros(TR["n"], np.float32)
        m, p = fit("YetiRank", label=lab)
        return "constant label accepted"
    check(S, "YetiRank with a constant label over the whole dataset", f)
    # metric semantics: ties and all-negative groups
    def f():
        y = EV["y"]; s = np.zeros(EV["n"])
        cb = eval_metric(y, s, "NDCG:type=Base", group_id=EV["gid"])[0]
        mine = float(ndcg_pess(s, y, EV["off"], EV["M"]).mean())
        opt = []
        for g in range(len(EV["M"])):
            t = y[EV["off"][g]:EV["off"][g] + EV["M"][g]]; disc = 1 / np.log2(np.arange(len(t)) + 2.0)
            opt.append(1.0)
        return f"constant scores: CatBoost NDCG {cb:.6f} vs pessimistic-tie NDCG {mine:.6f} (|diff| {abs(cb - mine):.2e})"
    check(S, "built-in NDCG:type=Base ties = pessimistic (matches the project metric on all-tied scores)", f)
    def f():
        r = np.random.default_rng(0); out = []
        for k in range(3):
            s = np.round(r.normal(size=EV["n"]), 1 if k < 2 else 6)       # many ties / no ties
            cb = eval_metric(EV["y"], s, "NDCG:type=Base", group_id=EV["gid"])[0]
            mine = float(ndcg_pess(s, EV["y"], EV["off"], EV["M"]).mean()); out.append(f"{cb:.6f}/{mine:.6f}")
        s = r.normal(size=EV["n"])
        v = {k: eval_metric(EV["y"], s, k, group_id=EV["gid"])[0] for k in
             ["NDCG", "NDCG:type=Base", "NDCG:type=Exp", "NDCG:top=10", "NDCG:denominator=Position", "NDCG:top=-1;type=Base;denominator=LogPosition"]}
        return "CatBoost/project on rounded and continuous scores: " + ", ".join(out) + " | variants: " + json.dumps({k: round(x, 5) for k, x in v.items()})
    check(S, "built-in NDCG equals the project metric on scores with partial ties", f)
    def f():
        sys.path.insert(0, "<repo>/src")
        from metrics import ndcg_per_feed
        r = np.random.default_rng(1); s = np.round(r.normal(size=EV["n"]), 1).astype(np.float32)
        a = float(ndcg_per_feed(s, EV["y"], EV["off"], EV["M"]).mean()); b = float(ndcg_pess(s, EV["y"], EV["off"], EV["M"]).mean())
        c = eval_metric(EV["y"], s.astype(np.float64), "NDCG:type=Base", group_id=EV["gid"])[0]
        return f"src/metrics.ndcg_per_feed {a:.6f}, this script {b:.6f}, CatBoost {c:.6f}"
    check(S, "project evaluator (src/metrics.ndcg_per_feed, read-only import) vs CatBoost NDCG:type=Base", f)
    def f():
        y = EV["y"].copy(); g0 = slice(EV["off"][0], EV["off"][0] + EV["M"][0]); y[:] = 0
        one = eval_metric(y[g0], np.random.default_rng(0).normal(size=EV["M"][0]), "NDCG:type=Base", group_id=EV["gid"][g0])[0]
        return f"NDCG of a feed with no positive = {one} (CatBoost counts it as 1; src/metrics.ndcg_per_feed also returns 1 when IDCG = 0)"
    check(S, "built-in NDCG on an all-negative feed", f)
    # large groups
    def f():
        D = make(6, 7, min_m=3000, max_m=3000)
        m = CatBoostRanker(loss_function="YetiRank", **BASE); m.fit(Pool(pd.DataFrame(D["X"], columns=NAMES), label=D["y"], group_id=D["gid"]))
        m2 = CatBoostRanker(loss_function="YetiRank:mode=NDCG", **BASE); m2.fit(Pool(pd.DataFrame(D["X"], columns=NAMES), label=D["y"], group_id=D["gid"]))
        m3 = CatBoostRanker(loss_function="LambdaMart", **BASE); m3.fit(Pool(pd.DataFrame(D["X"], columns=NAMES), label=D["y"], group_id=D["gid"]))
        return "6 feeds of 3000 candidates: YetiRank, YetiRank:mode=NDCG, LambdaMart train on CPU"
    check(S, "group size 3000 on CPU (no CPU size limit hit)", f)


def sec_tree():
    S = "d.tree"
    L = ["YetiRank", "YetiRank:mode=NDCG", "QuerySoftMax", "QueryRMSE", "LambdaMart", "StochasticRank:metric=NDCG", "PairLogit",
         "YetiRankPairwise", "PairLogitPairwise"]
    for gp in ["Depthwise", "Lossguide"]:
        for loss in L:
            check(S, f"{loss} + grow_policy={gp}", run_loss(loss, grow_policy=gp, effect_vs=loss))
    check(S, "YetiRank Lossguide max_leaves=255 depth=16 min_data_in_leaf=50", run_loss("YetiRank", grow_policy="Lossguide", max_leaves=255, depth=16, min_data_in_leaf=50))
    check(S, "YetiRank Depthwise depth=10 min_data_in_leaf=50", run_loss("YetiRank", grow_policy="Depthwise", depth=10, min_data_in_leaf=50))
    check(S, "YetiRank SymmetricTree + min_data_in_leaf=50 (documented: only for Depthwise/Lossguide)", run_loss("YetiRank", min_data_in_leaf=50, effect_vs="YetiRank"))
    check(S, "YetiRank SymmetricTree + max_leaves=31", run_loss("YetiRank", max_leaves=31, effect_vs="YetiRank"))
    for d in [10, 12, 16, 17]:
        check(S, f"YetiRank depth={d} (CPU)", run_loss("YetiRank", depth=d, iterations=5))
    for d in [8, 9]:
        check(S, f"YetiRankPairwise depth={d} (CPU, 3 iterations; note the seconds)", run_loss("YetiRankPairwise", depth=d, iterations=3))
    for bt in ["Ordered", "Plain"]:
        for loss in L:
            check(S, f"{loss} + boosting_type={bt}", run_loss(loss, boosting_type=bt, effect_vs=loss))
    check(S, "YetiRank Ordered + grow_policy=Lossguide", run_loss("YetiRank", boosting_type="Ordered", grow_policy="Lossguide"))
    for loss in ["YetiRank", "YetiRank:mode=NDCG", "QuerySoftMax", "QueryRMSE", "LambdaMart", "StochasticRank:metric=NDCG", "PairLogit"]:
        for lem in ["Newton", "Gradient"]:
            check(S, f"{loss} + leaf_estimation_method={lem}", run_loss(loss, leaf_estimation_method=lem, effect_vs=loss))
        check(S, f"{loss} + leaf_estimation_iterations=5", run_loss(loss, leaf_estimation_iterations=5, effect_vs=loss))
    for loss in ["YetiRank", "QuerySoftMax", "LambdaMart"]:
        for bk in ["No", "AnyImprovement", "Armijo"]:
            check(S, f"{loss} + leaf_estimation_backtracking={bk} (leaf_estimation_iterations=5 where allowed)",
                  run_loss(loss, leaf_estimation_backtracking=bk, leaf_estimation_iterations=5, effect_vs=loss, ref_kw=dict(leaf_estimation_iterations=5)))
    for loss in ["YetiRank", "QuerySoftMax", "LambdaMart", "StochasticRank:metric=NDCG", "YetiRankPairwise"]:
        check(S, f"{loss} + langevin=True, diffusion_temperature=10000", run_loss(loss, langevin=True, diffusion_temperature=10000, effect_vs=loss))
        check(S, f"{loss} + model_shrink_rate=0.1 (Constant)", run_loss(loss, model_shrink_rate=0.1, model_shrink_mode="Constant", effect_vs=loss))
    check(S, "YetiRank + model_shrink_rate=0.5, model_shrink_mode=Decreasing", run_loss("YetiRank", model_shrink_rate=0.5, model_shrink_mode="Decreasing", effect_vs="YetiRank"))
    check(S, "StochasticRank + posterior_sampling=True", run_loss("StochasticRank:metric=NDCG", posterior_sampling=True, effect_vs="StochasticRank:metric=NDCG"))
    for sf in ["Cosine", "L2", "NewtonL2", "NewtonCosine"]:
        check(S, f"YetiRank + score_function={sf} (CPU, SymmetricTree)", run_loss("YetiRank", score_function=sf, effect_vs="YetiRank"))
        check(S, f"YetiRank + score_function={sf} (CPU, Lossguide)", run_loss("YetiRank", score_function=sf, grow_policy="Lossguide",
                                                                           effect_vs="YetiRank", ref_kw=dict(grow_policy="Lossguide")))
    for loss in ["YetiRank", "QuerySoftMax", "YetiRankPairwise", "PairLogitPairwise"]:
        check(S, f"{loss} + random_strength=10", run_loss(loss, random_strength=10, effect_vs=loss))
        check(S, f"{loss} + random_strength=0", run_loss(loss, random_strength=0, effect_vs=loss))
    check(S, "YetiRank + random_score_type=Gumbel", run_loss("YetiRank", random_score_type="Gumbel", effect_vs="YetiRank"))
    for l2 in [0, 1, 30, 300]:
        check(S, f"YetiRank + l2_leaf_reg={l2}", run_loss("YetiRank", l2_leaf_reg=l2, effect_vs="YetiRank"))
    check(S, "YetiRank + border_count=1024", run_loss("YetiRank", border_count=1024, effect_vs="YetiRank"))
    check(S, "YetiRank + border_count=32", run_loss("YetiRank", border_count=32, effect_vs="YetiRank"))
    for fbt in ["Median", "Uniform", "UniformAndQuantiles", "MaxLogSum", "MinEntropy", "GreedyLogSum"]:
        check(S, f"YetiRank + feature_border_type={fbt}", run_loss("YetiRank", feature_border_type=fbt, effect_vs="YetiRank"))
    check(S, "YetiRank + per_float_feature_quantization=['0:border_count=1024','1:border_count=1024']",
          run_loss("YetiRank", per_float_feature_quantization=["0:border_count=1024", "1:border_count=1024"], effect_vs="YetiRank"))
    check(S, "YetiRank + model_size_reg=10 with a categorical feature", run_loss("YetiRank", cat=True, model_size_reg=10,
                                                                                effect_vs="YetiRank", ref_kw=dict(cat=True)))
    check(S, "YetiRank + boost_from_average=True", run_loss("YetiRank", boost_from_average=True))
    check(S, "YetiRank + fold_len_multiplier=1.5 + approx_on_full_history=True (Ordered)",
          run_loss("YetiRank", boosting_type="Ordered", fold_len_multiplier=1.5, approx_on_full_history=True))
    check(S, "YetiRank + has_time=True", run_loss("YetiRank", has_time=True, effect_vs="YetiRank"))
    check(S, "YetiRank + nan_mode=Max with NaNs in a feature", lambda: _nan_test())


def _nan_test():
    D = dict(TR); X = TR["X"].copy(); X[::7, 0] = np.nan
    m = CatBoostRanker(loss_function="YetiRank", nan_mode="Max", **BASE)
    m.fit(Pool(pd.DataFrame(X, columns=NAMES), label=TR["y"], group_id=TR["gid"]))
    return "trains with NaN (nan_mode=Max)"


def sec_sampling():
    S = "e.sampling"
    L = ["YetiRank", "YetiRank:mode=NDCG", "QuerySoftMax", "LambdaMart", "StochasticRank:metric=NDCG", "YetiRankPairwise", "PairLogit"]
    for loss in L:
        for kw in [dict(bootstrap_type="Bayesian"), dict(bootstrap_type="Bayesian", bagging_temperature=3),
                   dict(bootstrap_type="Bernoulli", subsample=0.5), dict(bootstrap_type="MVS", subsample=0.5),
                   dict(bootstrap_type="No"), dict(bootstrap_type="Poisson", subsample=0.5),
                   dict(sampling_unit="Group"), dict(sampling_unit="Group", bootstrap_type="Bernoulli", subsample=0.5),
                   dict(sampling_unit="Group", bootstrap_type="Bayesian", bagging_temperature=1),
                   dict(sampling_unit="Group", bootstrap_type="MVS", subsample=0.5),
                   dict(sampling_frequency="PerTree"), dict(sampling_frequency="PerTreeLevel"), dict(rsm=0.3),
                   dict(subsample=0.5)]:
            check(S, f"{loss} + {json.dumps(kw)}", run_loss(loss, effect_vs=loss, **kw))
    check(S, "YetiRank Lossguide + sampling_frequency=PerTreeLevel", run_loss("YetiRank", grow_policy="Lossguide", sampling_frequency="PerTreeLevel"))
    check(S, "YetiRank Depthwise + sampling_unit=Group Bernoulli 0.5 + rsm=0.5", run_loss("YetiRank", grow_policy="Depthwise", sampling_unit="Group",
                                                                                      bootstrap_type="Bernoulli", subsample=0.5, rsm=0.5))
    check(S, "YetiRank + colsample_bylevel=0.3 (synonym of rsm)", run_loss("YetiRank", colsample_bylevel=0.3, effect_vs="YetiRank", ref_kw=dict(rsm=0.3)))
    check(S, "YetiRank + mvs_reg=0.1 (MVS)", run_loss("YetiRank", bootstrap_type="MVS", mvs_reg=0.1, effect_vs="YetiRank"))


def sec_control():
    S = "f.control"
    for em in ["NDCG", "NDCG:type=Base", "NDCG:top=10", "NDCG:top=10;type=Exp;denominator=Position",
               "NDCG:top=-1;type=Base;denominator=LogPosition;hints=skip_train~false", "NDCG:type=Base;use_weights=false", "DCG", "MRR",
               "MRR:top=10", "MAP", "MAP:top=10", "ERR", "PFound", "PFound:top=10;decay=0.9", "QueryAUC", "QueryAUC:type=Ranking",
               "AUC", "AverageGain:top=1", "QueryAverage:top=1", "PrecisionAt:top=5", "RecallAt:top=5", "PairAccuracy", "FilteredDCG",
               "PairLogit", "QuerySoftMax", "QueryRMSE", "YetiRank", "LambdaMart", "StochasticRank:metric=NDCG", "Logloss", "RMSE"]:
        def f(em=em):
            m, p = fit("YetiRank", eval_metric=em, iterations=10)
            ev = m.get_evals_result(); k = list(ev.get("validation", {}).keys()); lk = list(ev.get("learn", {}).keys())
            v = ev["validation"][k[0]][-1] if k else None
            return f"validation keys {k}, learn keys {lk}, last value {v}"
        check(S, f"eval_metric='{em}' with loss YetiRank", f)
    def f():
        m, p = fit("YetiRank", eval_metric="NDCG:type=Base", custom_metric=["NDCG:top=10", "MRR", "MAP:top=10", "PFound", "QueryAUC:type=Ranking"], iterations=10)
        ev = m.get_evals_result()
        return f"validation keys {list(ev['validation'].keys())}; learn keys {list(ev.get('learn', {}).keys())}"
    check(S, "custom_metric list with ranking metrics", f)
    def f():
        m, p = fit("YetiRank", eval_metric="NDCG:type=Base", iterations=40)
        ev = m.get_evals_result()["validation"]["NDCG:type=Base"]
        staged = list(m.staged_predict(pool(EV), eval_period=10))
        mine = [float(ndcg_pess(np.asarray(s, np.float64), EV["y"], EV["off"], EV["M"]).mean()) for s in staged]
        cbv = [ev[i] for i in (9, 19, 29, 39)]
        return f"built-in eval NDCG at 10/20/30/40 {np.round(cbv, 6).tolist()} vs project metric on staged_predict {np.round(mine, 6).tolist()}"
    check(S, "eval_metric NDCG:type=Base during training == project metric on staged predictions", f)
    # python custom eval metric with a closure on the eval feed layout
    def f():
        calls = []
        class FeedNDCG:
            def is_max_optimal(self): return True
            def get_final_error(self, error, weight): return error / (weight + 1e-38)
            def evaluate(self, approxes, target, weight):
                a = np.asarray(approxes[0], np.float64); t = np.asarray(target, np.float32); calls.append(len(a))
                if len(a) != EV["n"]:
                    return 0.0, 1.0
                assert np.array_equal(t, EV["y"]), "target order differs from the pool order"
                return float(ndcg_pess(a, EV["y"], EV["off"], EV["M"]).sum()), float(len(EV["M"]))
        m, p = fit("YetiRank", eval_metric=FeedNDCG(), custom_metric=["NDCG:type=Base"], iterations=12, od_type="Iter", od_wait=50)
        ev = m.get_evals_result()["validation"]; k = list(ev.keys())
        a = ev[[x for x in k if "FeedNDCG" in x][0]]; b = ev["NDCG:type=Base"]
        return f"keys {k}; custom vs built-in max |diff| {np.abs(np.array(a) - np.array(b)).max():.2e}; evaluate() call sizes {sorted(set(calls))}"
    check(S, "python custom eval_metric (closure on feed offsets) = built-in NDCG; approxes arrive in pool order", f)
    # python custom objective: how is calc_ders_range called?
    def f():
        sizes = []
        class Obj:
            def calc_ders_range(self, approxes, targets, weights):
                sizes.append(len(approxes))
                a = np.asarray(approxes, np.float64); t = np.asarray(targets, np.float64); p = 1 / (1 + np.exp(-a))
                return list(zip((t - p).tolist(), (-p * (1 - p)).tolist()))
        m = CatBoostRanker(loss_function=Obj(), eval_metric="NDCG:type=Base", **dict(BASE, iterations=5))
        m.fit(pool(TR), eval_set=pool(EV))
        return f"per-object python objective accepted by CatBoostRanker; calc_ders_range call sizes {sorted(set(sizes))} (train rows {TR['n']})"
    check(S, "python custom objective (calc_ders_range) under CatBoostRanker", f)
    def f():
        sizes = []
        class Obj:
            def calc_ders_range(self, approxes, targets, weights):
                sizes.append(len(approxes))
                a = np.asarray(approxes, np.float64); t = np.asarray(targets, np.float64); p = 1 / (1 + np.exp(-a))
                return list(zip((t - p).tolist(), (-p * (1 - p)).tolist()))
        m = CatBoost(dict(BASE, loss_function=Obj(), eval_metric="NDCG:type=Base", iterations=5))
        m.fit(pool(TR), eval_set=pool(EV))
        return f"calc_ders_range call sizes {sorted(set(sizes))} of {TR['n']} train rows (whole-dataset calls = a listwise objective by closure is possible)"
    check(S, "python custom objective under CatBoost(): size of each calc_ders_range call", f)
    # early stopping
    for kw, nm in [(dict(od_type="Iter", od_wait=5), "od_type=Iter od_wait=5"), (dict(early_stopping_rounds=5), "early_stopping_rounds=5"),
                   (dict(od_type="IncToDec", od_pval=1e-2, od_wait=5), "od_type=IncToDec od_pval=1e-2 od_wait=5"),
                   (dict(od_type="Iter", od_wait=5, od_pval=1e-2), "od_type=Iter with od_pval"),
                   (dict(od_type="Iter", od_wait=5, use_best_model=False), "od_type=Iter od_wait=5 use_best_model=False"),
                   (dict(od_type="Iter", od_wait=5, best_model_min_trees=60), "best_model_min_trees=60")]:
        def f(kw=kw):
            kw2 = dict(use_best_model=True); kw2.update(kw)
            m, p = fit("YetiRank", eval_metric="NDCG:type=Base", iterations=400, learning_rate=0.5, **kw2)
            return f"trees kept {m.tree_count_}, best_iteration {m.get_best_iteration()}, best score {m.get_best_score()['validation']}"
        check(S, f"early stopping: {nm}", f)
    def f():
        ptr = pool(TR); p1 = pool(EV); p2 = pool(make(60, 9))
        m = CatBoostRanker(loss_function="YetiRank", eval_metric="NDCG:type=Base", **dict(BASE, iterations=15, use_best_model=True))
        m.fit(ptr, eval_set=[p1, p2])
        ev = m.get_evals_result()
        return f"evals_result keys {list(ev.keys())}; best_iteration {m.get_best_iteration()} (the LAST eval set drives early stopping per the docs)"
    check(S, "two eval sets on CPU", f)
    def f():
        m, p = fit("YetiRank", iterations=10, metric_period=5)
        return f"metric_period=5: recorded values {len(m.get_evals_result()['validation'][list(m.get_evals_result()['validation'].keys())[0]])}"
    check(S, "metric_period", f)
    # snapshots
    def f():
        d = os.path.join(OUT, "snap_train_dir"); sf = "snap_test.cbsnapshot"
        import shutil; shutil.rmtree(d, ignore_errors=True)
        kw = dict(BASE, allow_writing_files=True, train_dir=d)
        m1 = CatBoostRanker(loss_function="YetiRank", **dict(kw, iterations=12)); m1.fit(pool(TR), save_snapshot=True, snapshot_file=sf, snapshot_interval=1)
        full = CatBoostRanker(loss_function="YetiRank", **dict(kw, iterations=12, train_dir=d + "_ref")); full.fit(pool(TR))
        ok = os.path.exists(os.path.join(d, sf))
        m2 = CatBoostRanker(loss_function="YetiRank", **dict(kw, iterations=12)); m2.fit(pool(TR), save_snapshot=True, snapshot_file=sf, snapshot_interval=1)
        same = np.array_equal(m2.predict(pool(EV)), full.predict(pool(EV)))
        return f"snapshot file written: {ok}; resume with identical params returns {m2.tree_count_} trees, equal to an uninterrupted run: {same}"
    check(S, "save_snapshot / snapshot_file with YetiRank (needs allow_writing_files=True)", f)
    def f():
        m = CatBoostRanker(loss_function="YetiRank", **dict(BASE, iterations=5)); m.fit(pool(TR), save_snapshot=True)
        return "save_snapshot accepted with allow_writing_files=False"
    check(S, "save_snapshot with allow_writing_files=False", f)
    # staged prediction, shrink, eval_metrics
    def f():
        m, p = fit("YetiRank", iterations=20)
        a = np.asarray(m.predict(pool(EV), ntree_end=10)); st = list(m.staged_predict(pool(EV), eval_period=10))
        em = m.eval_metrics(pool(EV), ["NDCG:type=Base", "MRR"], eval_period=10)
        m2 = m.copy(); m2.shrink(10)
        return (f"predict(ntree_end=10)==staged[0]: {np.allclose(a, st[0])}; shrink(10)==ntree_end=10: {np.allclose(a, m2.predict(pool(EV)))}; "
                f"eval_metrics keys {list(em.keys())} values {[round(x, 5) for x in em['NDCG:type=Base']]}")
    check(S, "staged_predict / predict(ntree_end) / shrink / eval_metrics", f)


def sec_baseline():
    S = "f.baseline_init_sum"
    btr = (0.7 * TR["t"]).astype(np.float64); bev = (0.7 * EV["t"]).astype(np.float64)
    for loss in ["YetiRank", "YetiRank:mode=NDCG", "YetiRankPairwise", "QuerySoftMax", "QueryRMSE", "LambdaMart", "StochasticRank:metric=NDCG", "PairLogit",
                 "PairLogitPairwise", "GroupQuantile"]:
        def f(loss=loss):
            m = CatBoostRanker(loss_function=loss, eval_metric="NDCG:type=Base", **BASE)
            m.fit(pool(TR, baseline=btr), eval_set=pool(EV, baseline=bev))
            ev = m.get_evals_result()["validation"]["NDCG:type=Base"]
            d_only = np.asarray(m.predict(pool(EV)), np.float64); with_b = np.asarray(m.predict(pool(EV, baseline=bev)), np.float64)
            incl = np.allclose(with_b, d_only + bev, atol=1e-5); excl = np.allclose(with_b, d_only, atol=1e-9)
            qb = quality(bev); qd = quality(d_only); qs = quality(d_only + bev)
            ref_nob = ref(loss)
            return (f"baseline alone {qb:.4f}; eval NDCG it0 {ev[0]:.4f} -> last {ev[-1]:.4f}; trees alone {qd:.4f}; baseline+trees {qs:.4f}; "
                    f"model without baseline {quality(ref_nob):.4f}; predict(pool WITH baseline) {'= trees + baseline' if incl else ('= trees only' if excl else '= other')}")
        check(S, f"{loss} + Pool(baseline=0.7*teacher) on train and eval", f)
    for loss in ["Logloss", "CrossEntropy", "RMSE"]:
        def f(loss=loss):
            m = CatBoost(dict(BASE, loss_function=loss, eval_metric="NDCG:type=Base"))
            m.fit(pool(TR, baseline=btr), eval_set=pool(EV, baseline=bev))
            ev = m.get_evals_result()["validation"]["NDCG:type=Base"]; d_only = np.asarray(m.predict(pool(EV)), np.float64)
            return f"eval NDCG it0 {ev[0]:.4f} -> last {ev[-1]:.4f}; trees alone {quality(d_only):.4f}; baseline+trees {quality(d_only + bev):.4f}"
        check(S, f"{loss} (pointwise) + baseline", f)
    def f():
        out = []
        for sc in [0.1, 0.7, 3.0, 10.0]:
            m = CatBoostRanker(loss_function="YetiRank", **BASE); m.fit(pool(TR, baseline=sc * TR["t"].astype(np.float64)))
            d = np.asarray(m.predict(pool(EV)), np.float64); out.append(f"scale {sc}: total {quality(d + sc * EV['t']):.4f} (trees sd {np.std(d):.3f})")
        return "; ".join(out)
    check(S, "the SCALE of the baseline matters for YetiRank (pairwise logistic on score differences)", f)
    def f():
        m = CatBoostRanker(loss_function="YetiRank", **BASE)
        m.fit(pool(TR, baseline=btr), eval_set=pool(EV))          # eval pool without baseline
        return "train with baseline, eval set without baseline: accepted"
    check(S, "baseline on the train pool only", f)
    def f():
        m = CatBoostRanker(loss_function="YetiRank", **BASE); m.fit(frame(TR), TR["y"], group_id=TR["gid"], baseline=btr)
        return "baseline passed through fit(X, y, baseline=...)"
    check(S, "baseline through fit() arguments", f)
    # init_model
    def cont(loss1, loss2, cat=False, emb=False, kw1=None, kw2=None):
        def f():
            m1, p1 = fit(loss1, cat=cat, emb=emb, iterations=10, **(kw1 or {}))
            m2 = CatBoostRanker(loss_function=loss2, **dict(BASE, iterations=10, **(kw2 or {})))
            m2.fit(pool(TR, cat=cat, emb=emb), init_model=m1)
            a = np.asarray(m2.predict(pool(EV, cat=cat, emb=emb), ntree_end=10), np.float64)
            return (f"continued: {m2.tree_count_} trees; first 10 trees reproduce the init model: {np.allclose(a, p1, atol=1e-6)}; "
                    f"NDCG init {quality(p1):.4f} -> continued {quality(np.asarray(m2.predict(pool(EV, cat=cat, emb=emb)), np.float64)):.4f}")
        return f
    check(S, "init_model: YetiRank -> YetiRank", cont("YetiRank", "YetiRank"))
    check(S, "init_model: QuerySoftMax -> YetiRank (objective switch)", cont("QuerySoftMax", "YetiRank"))
    check(S, "init_model: QueryRMSE -> YetiRank:mode=NDCG", cont("QueryRMSE", "YetiRank:mode=NDCG"))
    check(S, "init_model: YetiRank -> LambdaMart", cont("YetiRank", "LambdaMart"))
    check(S, "init_model: YetiRank lr 0.15 -> lr 0.03, depth 6 -> 8", cont("YetiRank", "YetiRank", kw2=dict(learning_rate=0.03, depth=8)))
    check(S, "init_model with a categorical feature (CTRs)", cont("YetiRank", "YetiRank", cat=True))
    check(S, "init_model with an embedding feature", cont("YetiRank", "YetiRank", emb=True))
    check(S, "init_model: Lossguide -> Lossguide", cont("YetiRank", "YetiRank", kw1=dict(grow_policy="Lossguide"), kw2=dict(grow_policy="Lossguide")))
    check(S, "init_model: SymmetricTree -> Lossguide", cont("YetiRank", "YetiRank", kw2=dict(grow_policy="Lossguide")))
    check(S, "init_model: Plain -> Ordered", cont("YetiRank", "YetiRank", kw2=dict(boosting_type="Ordered")))
    check(S, "init_model with different border_count (254 -> 64)", cont("YetiRank", "YetiRank", kw2=dict(border_count=64)))
    def f():
        m0 = CatBoost(dict(BASE, loss_function="RMSE", iterations=10)); m0.fit(pool(TR, label="t"))
        m2 = CatBoostRanker(loss_function="YetiRank", **dict(BASE, iterations=10)); m2.fit(pool(TR), init_model=m0)
        return f"continued from an RMSE regressor on teacher logits: {m2.tree_count_} trees, NDCG {quality(np.asarray(m2.predict(pool(EV)), np.float64)):.4f}"
    check(S, "init_model: RMSE regressor (teacher logit) -> YetiRank on clicks (distil then fine-tune)", f)
    def f():
        m1, p1 = fit("YetiRank", iterations=10)
        m2 = CatBoostRanker(loss_function="YetiRank", **dict(BASE, iterations=10)); m2.fit(pool(TR, baseline=btr), init_model=m1)
        return "init_model together with a baseline: accepted"
    check(S, "init_model + baseline together", f)
    # sum_models
    def summ(cat=False, emb=False, kw=None, w=(0.5, 0.5)):
        def f():
            m1, p1 = fit("YetiRank", cat=cat, emb=emb, **(kw or {})); m2, p2 = fit("QuerySoftMax", cat=cat, emb=emb, random_seed=1, **(kw or {}))
            s = sum_models([m1, m2], weights=list(w)); ps = np.asarray(s.predict(pool(EV, cat=cat, emb=emb)), np.float64)
            err = np.abs(ps - (w[0] * p1 + w[1] * p2)).max()
            return f"sum_models max |pred - weighted sum| {err:.2e}; trees {s.tree_count_}; type {type(s).__name__}"
        return f
    check(S, "sum_models(YetiRank, QuerySoftMax) numeric features", summ())
    check(S, "sum_models with weights (0.3, 1.7)", summ(w=(0.3, 1.7)))
    check(S, "sum_models with a categorical feature (CTRs)", summ(cat=True))
    check(S, "sum_models with an embedding feature", summ(emb=True))
    check(S, "sum_models of Lossguide models", summ(kw=dict(grow_policy="Lossguide")))
    def f():
        m1, p1 = fit("YetiRank"); m2, p2 = fit("YetiRank", random_seed=1, border_count=64)
        s = sum_models([m1, m2]); ps = np.asarray(s.predict(pool(EV)), np.float64)
        return f"different border sets: max err {np.abs(ps - (p1 + p2)).max():.2e}"
    check(S, "sum_models of models with different quantisation", f)


def sec_constraints():
    S = "g.constraints"
    mc = {"x0": 1, "x3": -1}
    for loss in ["YetiRank", "YetiRank:mode=NDCG", "QuerySoftMax", "QueryRMSE", "LambdaMart", "StochasticRank:metric=NDCG", "PairLogit", "YetiRankPairwise",
                 "PairLogitPairwise"]:
        check(S, f"{loss} + monotone_constraints={mc}", run_loss(loss, monotone_constraints=mc, effect_vs=loss))
    def f():
        m, p = fit("YetiRank", monotone_constraints={"x0": 1})
        X = EV["X"][:2000].copy(); g = np.zeros(2000, np.int64); lo = X.copy(); hi = X.copy(); lo[:, 0] = -2; hi[:, 0] = 2
        a = m.predict(Pool(pd.DataFrame(lo, columns=NAMES), group_id=g)); b = m.predict(Pool(pd.DataFrame(hi, columns=NAMES), group_id=g))
        return f"x0: -2 -> +2 never lowers the score: {bool((b >= a - 1e-9).all())} (min diff {float((b - a).min()):.4f})"
    check(S, "monotone constraint is actually enforced (YetiRank)", f)
    check(S, "YetiRank + monotone_constraints + grow_policy=Lossguide", run_loss("YetiRank", monotone_constraints=mc, grow_policy="Lossguide"))
    check(S, "YetiRank + monotone_constraints + boosting_type=Ordered", run_loss("YetiRank", monotone_constraints=mc, boosting_type="Ordered"))
    check(S, "YetiRank + feature_weights={x0:0.1,x1:0.1}", run_loss("YetiRank", feature_weights={"x0": 0.1, "x1": 0.1}, effect_vs="YetiRank"))
    check(S, "YetiRank + first_feature_use_penalties={x0:5}", run_loss("YetiRank", first_feature_use_penalties={"x0": 5.0}, effect_vs="YetiRank"))
    check(S, "YetiRank + per_object_feature_penalties={x0:0.01}", run_loss("YetiRank", per_object_feature_penalties={"x0": 0.01}, effect_vs="YetiRank"))
    check(S, "YetiRank + penalties_coefficient=10 + first_feature_use_penalties", run_loss("YetiRank", penalties_coefficient=10, first_feature_use_penalties={"x0": 5.0},
                                                                                         effect_vs="YetiRank", ref_kw=dict(first_feature_use_penalties={"x0": 5.0})))
    check(S, "YetiRank + ignored_features=['x0','x1']", run_loss("YetiRank", ignored_features=["x0", "x1"], effect_vs="YetiRank"))
    check(S, "YetiRank Lossguide + feature_weights", run_loss("YetiRank", grow_policy="Lossguide", feature_weights={"x0": 0.1}, effect_vs="YetiRank", ref_kw=dict(grow_policy="Lossguide")))
    check(S, "YetiRankPairwise + feature_weights", run_loss("YetiRankPairwise", feature_weights={"x0": 0.1}, effect_vs="YetiRankPairwise"))
    check(S, "interaction_constraints (not a CatBoost parameter)", run_loss("YetiRank", interaction_constraints=[[0, 1]]))
    def f():
        import inspect
        sig = inspect.signature(CatBoostRanker.__init__).parameters
        return "CatBoostRanker.__init__ has: " + ", ".join(k for k in ["monotone_constraints", "feature_weights", "first_feature_use_penalties", "per_object_feature_penalties",
                                                                       "penalties_coefficient", "ignored_features", "interaction_constraints", "fixed_binary_splits",
                                                                       "embedding_features", "feature_calcers", "text_processing", "embedding_processing"] if k in sig) + \
               " | absent: " + ", ".join(k for k in ["interaction_constraints", "embedding_processing"] if k not in sig)
    check(S, "constructor signature (installed core.py)", f)


def sec_embedding():
    S = "x.embedding_with_rank_losses"
    q1 = soft_q(TR["t"], TR["off"], TR["M"], 1.0)
    for loss in ["YetiRank", "YetiRank:mode=NDCG", "YetiRankPairwise", "QuerySoftMax", "QueryRMSE", "LambdaMart", "StochasticRank:metric=NDCG", "PairLogit"]:
        def f(loss=loss):
            m, p = fit(loss, emb=True)
            r = ref(loss)
            fi = dict(zip(m.feature_names_, m.get_feature_importance(type="PredictionValuesChange")))
            return f"eval NDCG {quality(p):.4f} (without the embedding {quality(r):.4f}); importance of e0 {fi.get('e0', float('nan')):.2f}"
        check(S, f"{loss} + embedding_features (binary labels)", f)
    for loss, lab, nm in [("QuerySoftMax", q1, "soft prob labels"), ("QueryRMSE", TR["t"], "teacher logit labels"), ("YetiRank", TR["t"], "teacher logit labels")]:
        def f(loss=loss, lab=lab):
            m, p = fit(loss, emb=True, label=lab)
            fi = dict(zip(m.feature_names_, m.get_feature_importance(type="PredictionValuesChange")))
            return f"eval NDCG {quality(p):.4f}; importance of e0 {fi.get('e0', float('nan')):.2f}"
        check(S, f"{loss} + embedding_features + {nm}", f)
    for loss in ["Logloss", "CrossEntropy", "RMSE"]:
        def f(loss=loss):
            m, p = fit(loss, emb=True, cls=CatBoost, label="t" if loss == "RMSE" else "y")
            fi = dict(zip(m.feature_names_, m.get_feature_importance(type="PredictionValuesChange")))
            return f"eval NDCG {quality(p):.4f}; importance of e0 {fi.get('e0', float('nan')):.2f}"
        check(S, f"{loss} + embedding_features", f)
    def f():
        m, p = fit("YetiRank", emb=True, iterations=5)
        ap = m.get_all_params()
        return "resolved embedding options: " + json.dumps({k: v for k, v in ap.items() if "embed" in k.lower() or "calcer" in k.lower()}, default=str)[:900]
    check(S, "resolved embedding processing defaults under YetiRank", f)


def sec_pointwise():
    """pointwise losses through the generic CatBoost(params) class (re-run of the checks that need it)."""
    S = "p.pointwise"
    q1 = soft_q(TR["t"], TR["off"], TR["M"], 1.0)
    sig = (1 / (1 + np.exp(-(TR["t"] - 2.0)))).astype(np.float32)
    logit = TR["t"]; cen = centre(TR["t"], TR["off"], TR["M"])
    for s_ in ["Logloss", "CrossEntropy", "RMSE"]:
        check(S, f"{s_} via CatBoost(params) with group_id in the pool, eval_metric NDCG:type=Base (binary labels)",
              run_loss(s_, cls=CatBoost, eval_metric="NDCG:type=Base"))
    for loss, lab, nm, kw in [("CrossEntropy", sig, "pointwise prob in (0,1)", {}),
                              ("CrossEntropy", q1, "soft prob (sums to 1 per feed)", {}),
                              ("CrossEntropy", logit, "teacher logit (outside [0,1])", {}),
                              ("Logloss", sig, "pointwise prob in (0,1), no target_border", {}),
                              ("Logloss", sig, "pointwise prob in (0,1), target_border=0.5", dict(target_border=0.5)),
                              ("RMSE", logit, "teacher logit", {}), ("RMSE", cen, "feed-centred teacher logit", {}),
                              ("RMSE", sig, "pointwise prob in (0,1)", {}),
                              ("QueryRMSE", logit, "teacher logit (QueryRMSE removes the feed mean itself)", {}),
                              ("Huber:delta=1.0", cen, "feed-centred teacher logit", {}),
                              ("MAE", cen, "feed-centred teacher logit", {}),
                              ("LogCosh", cen, "feed-centred teacher logit", {})]:
        def f(loss=loss, lab=lab, kw=kw):
            from scipy.stats import spearmanr
            m, p = fit(loss, label=lab, cls=CatBoost, **kw)
            rs = [spearmanr(p[o:o + mm], EV["t"][o:o + mm])[0] for o, mm in zip(EV["off"][:60], EV["M"][:60])]
            return f"eval NDCG(hard clicks) {quality(p):.4f}; within-feed Spearman to teacher {np.nanmean(rs):.3f}; pred sd {np.std(p):.3g}"
        check(S, f"{loss} <- {nm}", f)
    for loss in ["Logloss", "CrossEntropy", "RMSE"]:
        def f(loss=loss):
            m, p = fit(loss, emb=True, cls=CatBoost, label="t" if loss == "RMSE" else "y")
            r = fit(loss, cls=CatBoost, label="t" if loss == "RMSE" else "y")[1]
            fi = dict(zip(m.feature_names_, m.get_feature_importance(type="PredictionValuesChange")))
            return f"eval NDCG {quality(p):.4f} (without the embedding {quality(r):.4f}); importance of e0 {fi.get('e0', float('nan')):.2f}"
        check(S, f"{loss} + embedding_features", f)
    btr = (0.7 * TR["t"]).astype(np.float64); bev = (0.7 * EV["t"]).astype(np.float64)
    for loss in ["Logloss", "CrossEntropy", "RMSE"]:
        def f(loss=loss):
            m = CatBoost(dict(BASE, loss_function=loss, eval_metric="NDCG:type=Base"))
            m.fit(pool(TR, baseline=btr), eval_set=pool(EV, baseline=bev))
            ev = m.get_evals_result()["validation"]["NDCG:type=Base"]; d_only = np.asarray(m.predict(pool(EV)), np.float64)
            return f"eval NDCG it0 {ev[0]:.4f} -> last {ev[-1]:.4f}; trees alone {quality(d_only):.4f}; baseline+trees {quality(d_only + bev):.4f}"
        check(S, f"{loss} (pointwise) + baseline", f)
    def f():
        m0 = CatBoost(dict(BASE, loss_function="RMSE", iterations=10)); m0.fit(pool(TR, label="t"))
        m2 = CatBoostRanker(loss_function="YetiRank", **dict(BASE, iterations=10)); m2.fit(pool(TR), init_model=m0)
        return f"continued from an RMSE regressor on teacher logits: {m2.tree_count_} trees, NDCG {quality(np.asarray(m2.predict(pool(EV)), np.float64)):.4f}"
    check(S, "init_model: RMSE regressor (teacher logit) -> YetiRank on clicks (distil then fine-tune)", f)
    def f():
        sizes = []
        class Obj:
            def calc_ders_range(self, approxes, targets, weights):
                sizes.append(len(approxes))
                a = np.asarray(approxes, np.float64); t = np.asarray(targets, np.float64); p = 1 / (1 + np.exp(-a))
                return list(zip((t - p).tolist(), (-p * (1 - p)).tolist()))
        m = CatBoost(dict(BASE, loss_function=Obj(), eval_metric="NDCG:type=Base", iterations=5))
        m.fit(pool(TR), eval_set=pool(EV))
        return f"calc_ders_range call sizes {sorted(set(sizes))} of {TR['n']} train rows; n calls {len(sizes)}"
    check(S, "python custom objective under CatBoost(params): size of each calc_ders_range call", f)


def sec_followup():
    """second-round checks raised by the first-round results."""
    S = "z.followup"
    # early stopping (get_best_score() is empty for YetiRank on 1.2.10, so read tree_count_ / get_best_iteration())
    for kw, nm in [(dict(od_type="Iter", od_wait=5), "od_type=Iter od_wait=5"), (dict(early_stopping_rounds=5), "early_stopping_rounds=5"),
                   (dict(od_type="IncToDec", od_pval=1e-2, od_wait=5), "od_type=IncToDec od_pval=1e-2 od_wait=5"),
                   (dict(od_type="Iter", od_wait=5, use_best_model=False), "od_type=Iter od_wait=5 use_best_model=False"),
                   (dict(od_type="Iter", od_wait=5, best_model_min_trees=60), "od_type=Iter od_wait=5 best_model_min_trees=60")]:
        def f(kw=kw):
            kw2 = dict(use_best_model=True); kw2.update(kw)
            m, p = fit("YetiRank", eval_metric="NDCG:type=Base", iterations=400, learning_rate=0.5, **kw2)
            n_rec = len(m.get_evals_result()["validation"]["NDCG:type=Base"])
            return f"iterations run {n_rec} of 400; trees kept {m.tree_count_}; best_iteration {m.get_best_iteration()}; get_best_score() = {m.get_best_score()}"
        check(S, f"early stopping: {nm}", f)
    def f():
        m, p = fit("YetiRank", iterations=400, learning_rate=0.5, od_type="Iter", od_wait=5, use_best_model=True)
        return f"no eval_metric: stops on {list(m.get_evals_result()['validation'].keys())}; iterations run {len(list(m.get_evals_result()['validation'].values())[0])}"
    check(S, "early stopping without eval_metric uses the objective's default metric", f)
    # leaf estimation stability of YetiRank (default l2 resolves to 1e-20)
    for kw in [dict(leaf_estimation_iterations=2), dict(leaf_estimation_iterations=5), dict(leaf_estimation_iterations=5, l2_leaf_reg=1),
               dict(leaf_estimation_iterations=5, l2_leaf_reg=5), dict(leaf_estimation_iterations=5, l2_leaf_reg=30)]:
        def f(kw=kw):
            m, p = fit("YetiRank", **kw)
            st = "PASS" if (m.tree_count_ == IT and np.isfinite(p).all() and np.std(p) < 1e6) else "UNSTABLE"
            return st, f"trees {m.tree_count_} of {IT}; pred sd {np.std(p):.3g}; eval NDCG {quality(p):.4f}"
        check(S, f"YetiRank + {json.dumps(kw)}", f)
    # QuerySoftMax: Newton leaves, lambda, beta, learning rate
    for loss, kw in [("QuerySoftMax", dict(leaf_estimation_method="Newton")), ("QuerySoftMax:lambda=0.5", dict(leaf_estimation_method="Newton")),
                     ("QuerySoftMax:lambda=0.0001", dict(leaf_estimation_method="Newton")), ("QuerySoftMax", dict(learning_rate=1.0)),
                     ("QuerySoftMax", dict(leaf_estimation_iterations=10)), ("QuerySoftMax:beta=3", dict(leaf_estimation_method="Newton"))]:
        ref_kw = dict(leaf_estimation_method="Newton") if "lambda" in loss else None
        check(S, f"{loss} + {json.dumps(kw)}", run_loss(loss, effect_vs="QuerySoftMax" if ref_kw else None, ref_kw=ref_kw, **kw))
    # LambdaMart with Gradient leaves: sign?
    def f():
        a = fit("LambdaMart")[1]; b = fit("LambdaMart", leaf_estimation_method="Gradient")[1]
        c = fit("LambdaMart:norm=false", leaf_estimation_method="Gradient")[1]
        return ("BUG?" if quality(b) < RANDOM_Q else "PASS",
                f"Newton NDCG {quality(a):.4f}; Gradient NDCG {quality(b):.4f} (random {RANDOM_Q:.4f}), corr(Newton, Gradient) {np.corrcoef(a, b)[0, 1]:.3f}; "
                f"NDCG of MINUS the Gradient model {quality(-b):.4f}; norm=false + Gradient {quality(c):.4f}")
    check(S, "LambdaMart + leaf_estimation_method=Gradient ranks in the WRONG direction?", f)
    # identities
    def f():
        a = fit("YetiRank:mode=MRR")[1]; b = fit("YetiRank:mode=ERR")[1]; c = fit("LambdaMart:metric=DCG")[1]; d = fit("LambdaMart:metric=NDCG")[1]
        e = fit("StochasticRank:metric=MRR")[1]; g = fit("StochasticRank:metric=ERR")[1]
        return (f"YetiRank MRR == ERR: {np.array_equal(a, b)}; LambdaMart DCG == NDCG: {np.array_equal(c, d)}; StochasticRank MRR == ERR: {np.array_equal(e, g)} "
                f"(binary labels)")
    check(S, "modes that coincide with binary labels", f)
    # label scale for YetiRank / LambdaMart / PairLogit (matters for soft labels)
    for loss in ["YetiRank", "YetiRank:mode=NDCG", "LambdaMart", "LambdaMart:norm=false", "PairLogit", "QueryRMSE", "StochasticRank:metric=NDCG;top=10"]:
        def f(loss=loss):
            a = fit(loss)[1]; b = fit(loss, label=(10.0 * TR["y"]).astype(np.float32))[1]
            same = np.array_equal(a, b)
            return ("NOEFFECT" if same else "PASS", f"labels x10: identical {same}; corr {np.corrcoef(a, b)[0, 1]:.4f}; sd ratio {np.std(b) / np.std(a):.3g}; "
                    f"NDCG {quality(a):.4f} -> {quality(b):.4f}")
        check(S, f"{loss}: does the label SCALE matter (y vs 10*y)?", f)
    # YetiRank with soft labels at different temperatures: order-only or magnitude-aware?
    def f():
        t = TR["t"]; a = fit("YetiRank", label=t)[1]; b = fit("YetiRank", label=np.sign(t) * np.abs(t) ** 3)[1]
        rk = np.zeros(TR["n"], np.float32)
        for o, mm in zip(TR["off"], TR["M"]):
            rk[o:o + mm] = np.argsort(np.argsort(t[o:o + mm])) / mm
        c = fit("YetiRank", label=rk)[1]; d = fit("PairLogit:max_pairs=200", label=t)[1]; e = fit("PairLogit:max_pairs=200", label=rk)[1]
        return (f"YetiRank on logit vs logit^3 (same order): corr {np.corrcoef(a, b)[0, 1]:.3f}, NDCG {quality(a):.4f} / {quality(b):.4f}; on within-feed rank: {quality(c):.4f}; "
                f"PairLogit logit vs rank identical: {np.array_equal(d, e)} (PairLogit uses the ORDER only; YetiRank weights pairs by the label gap)")
    check(S, "soft labels: YetiRank is magnitude-aware, PairLogit is order-only", f)
    # eval pool with group weights changes the built-in NDCG (weighted mean over feeds)
    def f():
        gw = np.repeat(np.random.default_rng(3).uniform(0.2, 3.0, len(EV["M"])), EV["M"]).astype(np.float32)
        s = np.random.default_rng(0).normal(size=EV["n"])
        a = eval_metric(EV["y"], s, "NDCG:type=Base", group_id=EV["gid"])[0]; b = eval_metric(EV["y"], s, "NDCG:type=Base", group_id=EV["gid"], group_weight=gw)[0]
        c = eval_metric(EV["y"], s, "NDCG:type=Base;use_weights=false", group_id=EV["gid"], group_weight=gw)[0]
        return f"unweighted {a:.5f}; with group_weight {b:.5f}; with group_weight and use_weights=false {c:.5f}"
    check(S, "built-in NDCG with group weights on the eval pool", f)
    # YetiRank + first-order sanity of baseline: beta scaling via QuerySoftMax beta on a baseline model
    def f():
        btr = TR["t"].astype(np.float64); bev = EV["t"].astype(np.float64); out = []
        for loss in ["QuerySoftMax", "QuerySoftMax:beta=0.3", "QuerySoftMax:beta=3"]:
            m = CatBoostRanker(loss_function=loss, **dict(BASE, leaf_estimation_method="Newton")); m.fit(pool(TR, baseline=btr), eval_set=pool(EV, baseline=bev))
            d = np.asarray(m.predict(pool(EV)), np.float64); out.append(f"{loss}: baseline+trees {quality(d + bev):.4f} (trees sd {np.std(d):.3f})")
        return f"baseline alone {quality(bev):.4f}; " + "; ".join(out) + " (beta multiplies baseline AND trees inside the softmax)"
    check(S, "QuerySoftMax beta with a baseline", f)
    # group_weight semantics for QuerySoftMax = label scaling?
    def f():
        gw = np.repeat(np.random.default_rng(3).uniform(0.2, 3.0, len(TR["M"])), TR["M"]).astype(np.float32)
        a = fit("QuerySoftMax", pool_kw=dict(group_weight=gw))[1]; b = fit("QuerySoftMax", label=(TR["y"] * gw).astype(np.float32))[1]
        return f"QuerySoftMax with group_weight w vs labels multiplied by w: identical {np.array_equal(a, b)}, corr {np.corrcoef(a, b)[0, 1]:.4f}"
    check(S, "QuerySoftMax: group_weight vs label scaling", f)


SECTIONS = dict(followup=sec_followup, pointwise=sec_pointwise, objectives=sec_objectives, defaults=sec_defaults, labels=sec_labels, groups=sec_groups, tree=sec_tree, sampling=sec_sampling,
                control=sec_control, baseline=sec_baseline, constraints=sec_constraints, embedding=sec_embedding)

if __name__ == "__main__":
    todo = sys.argv[1:] or list(SECTIONS)
    print("catboost", catboost.__version__, "| threads", TH, "| sections", todo, flush=True)
    for s in todo:
        n0 = len(RES); t0 = time.time()
        try:
            SECTIONS[s]()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            rec(s, "SECTION CRASHED", "FAIL", traceback.format_exc()[-600:])
        json.dump(RES[n0:], open(os.path.join(OUT, f"verify_{s}.json"), "w"), indent=1)
        print(f"== section {s}: {len(RES) - n0} checks in {time.time() - t0:.0f}s ==", flush=True)
    from collections import Counter
    print("TOTAL", dict(Counter(r["status"] for r in RES)))
