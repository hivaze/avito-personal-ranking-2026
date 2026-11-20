"""Shared helpers for the within-category analysis (agent within_cat)."""
import os
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "12"
import sys
sys.path.insert(0, "<repo>/lab/nn_x/errfix_1117/code")
import numpy as np
import polars as pl
import kit

_DISC = 1.0 / np.log2(np.arange(2, 302))


def load():
    df = kit.load()
    # group ids
    df = df.with_columns(
        (pl.col("fidx") * 100 + pl.col("cat")).alias("fc"),
        pl.col("score").rank("ordinal", descending=True).over(["fidx", "cat"]).alias("crank"),
    )
    return df


def ndcg_fast(scores, labels, off, sz):
    return kit.ndcg_per_feed(scores, labels, off, sz)


def cross_fit(df, make_scores, grid, folds=None, verbose=False):
    """make_scores(df, param) -> new scores for all rows. Choose param on 4 folds, evaluate on 5th.

    Returns dict with oof delta, se, z and chosen params per fold, plus the per-param full-sample deltas.
    """
    off, sz = kit.feed_layout(df)
    y = df["label"].to_numpy()
    base = df["score"].to_numpy().astype(np.float64)
    nb = kit.ndcg_per_feed(base, y, off, sz)
    f = kit.folds(df) if folds is None else folds
    ff = f[off]
    per = {}
    for p in grid:
        s = np.asarray(make_scores(df, p), dtype=np.float64)
        per[p] = kit.ndcg_per_feed(s, y, off, sz) - nb
    oof = np.zeros(len(sz))
    chosen = []
    for k in range(5):
        tr = ff != k
        best = max(grid, key=lambda p: per[p][tr].mean())
        chosen.append(best)
        oof[~tr] = per[best][~tr]
    se = oof.std(ddof=1) / np.sqrt(len(oof))
    res = dict(delta=float(oof.mean()), se=float(se), z=float(oof.mean() / se) if se > 0 else 0.0, chosen=chosen,
               full={p: (round(float(per[p].mean()), 5), round(float(per[p].std(ddof=1) / np.sqrt(len(sz))), 5)) for p in grid})
    if verbose:
        print(res)
    return res


def clogit(X, score, y, off, sz, l2=1e-3, fit_scale=True, w=None):
    """Listwise softmax (ListNet with label distribution) on z = a*score + X@b. Returns (a, b, se_b).

    X: (n, k) float64, already NaN-filled. Feeds with no positives contribute nothing.
    """
    from scipy.optimize import minimize
    n, k = X.shape
    seg = np.repeat(np.arange(len(sz)), sz)
    npos = np.add.reduceat(y.astype(np.float64), off)
    fw = np.ones(len(sz)) if w is None else w

    def f(theta):
        a = theta[0] if fit_scale else 1.0
        b = theta[1:]
        z = a * score + X @ b
        zmax = np.maximum.reduceat(z, off)
        e = np.exp(z - zmax[seg])
        S = np.add.reduceat(e, off)
        lse = np.log(S) + zmax
        L = (fw * (npos * lse - np.add.reduceat(y * z, off))).sum()
        p = e / S[seg]
        g_z = (fw * npos)[seg] * p - fw[seg] * y
        ga = g_z @ score if fit_scale else 0.0
        gb = X.T @ g_z
        L += 0.5 * l2 * (b ** 2).sum() * len(sz)
        gb = gb + l2 * b * len(sz)
        return L, np.r_[ga, gb]

    th0 = np.r_[1.0, np.zeros(k)]
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    th = r.x
    # approximate SE from the Fisher information (feed-level sandwich skipped: plain Hessian)
    a = th[0] if fit_scale else 1.0
    z = a * score + X @ th[1:]
    zmax = np.maximum.reduceat(z, off)
    e = np.exp(z - zmax[seg])
    S = np.add.reduceat(e, off)
    p = e / S[seg]
    Xf = np.c_[score, X]
    wp = (fw * npos)[seg] * p
    H = (Xf * wp[:, None]).T @ Xf
    m = np.zeros((len(sz), Xf.shape[1]))
    np.add.at(m, seg, Xf * p[:, None])
    H -= (m * (fw * npos)[:, None]).T @ m
    try:
        cov = np.linalg.inv(H + 1e-9 * np.eye(H.shape[0]))
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        se = np.full(Xf.shape[1], np.nan)
    return th[0], th[1:], se[1:], r.fun


BASEF = ["f1", "f2", "f3", "f4", "f5", "ucos", "hmax", "h1", "h5", "h10", "h32", "hmean5", "lhidx", "n_hdup95",
         "c_contact", "c_order", "c_fav", "c_lead", "c_vfeed", "c_vsearch", "cand_maxcos", "ldup", "lpop",
         "lpop_et2", "lpop_et4", "xp_imp_c50", "lcv", "pre2_fe_z", "pre2_new_z", "sdiff", "lcatn", "lcrank",
         "ccrank", "cat_hist_share", "lcatnew"]


def features(df):
    """Add derived numeric columns (NaN-filled) and within-(feed,cat) relative versions."""
    d = df.with_columns(
        (pl.col("hmax_idx").clip(0, None) + 1).log().alias("lhidx"),
        (pl.col("n_cand_dup95") + 1).log().alias("ldup"),
        (pl.col("it_hist_all_luo") + 1).log().alias("lpop"),
        (pl.col("it_hist_et2_luo") + 1).log().alias("lpop_et2"),
        (pl.col("it_hist_et4_luo") + 1).log().alias("lpop_et4"),
        (pl.col("seq_cv_invd20") + 1).log().alias("lcv"),
        (pl.col("s_ft") - pl.col("s_to")).alias("sdiff"),
        pl.col("cat_n_in_feed").cast(pl.Float64).log().alias("lcatn"),
        pl.col("cat_rank_in_feed").cast(pl.Float64).log().alias("lcrank"),
        ((pl.col("cat_rank_in_feed") - 1) / pl.col("cat_n_in_feed").clip(2, None).cast(pl.Float64) - 1).alias("ccrank"),
        (pl.when(pl.col("cat_newest_idx") < 0).then(256).otherwise(pl.col("cat_newest_idx")) + 1).cast(pl.Float64).log().alias("lcatnew"),
    )
    for c in ["ucos", "hmax", "h1", "h5", "h10", "h32", "hmean5", "cand_maxcos"]:
        d = d.with_columns(pl.col(c).fill_nan(pl.col(c).filter(pl.col(c).is_not_nan()).median()))
    for c in ["c_contact", "c_order", "c_fav", "c_lead", "c_vfeed", "c_vsearch"]:
        d = d.with_columns(pl.col(c).is_nan().alias(c + "_na"), pl.col(c).fill_nan(0.0))
    return d


def rel(d, cols, by=("fidx", "cat")):
    """Within-group deviations: x - mean, x - max (as <=0), and within-group percentile rank (0..1)."""
    by = list(by)
    ex = []
    for c in cols:
        ex += [(pl.col(c) - pl.col(c).mean().over(by)).alias(c + "_dm"),
               (pl.col(c) - pl.col(c).max().over(by)).alias(c + "_dmax"),
               ((pl.col(c).rank("average").over(by) - 1) / (pl.len().over(by) - 1).clip(1, None)).alias(c + "_pr")]
    return d.with_columns(ex)


def zcols(d, cols):
    X = d.select(cols).to_numpy().astype(np.float64)
    X = np.nan_to_num(X)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    return (X - mu) / sd


def add_expect(d, a=0.9825):
    """E = expected label under the per-feed softmax of a*score, scaled to the feed's number of positives."""
    return d.with_columns(
        ((a * pl.col("score") - (a * pl.col("score")).max().over("fidx")).exp()).alias("_e")
    ).with_columns(
        (pl.col("_e") / pl.col("_e").sum().over("fidx") * pl.col("label").sum().over("fidx")).alias("E")
    ).drop("_e")


def oe(d, by, min_n=0):
    t = d.group_by(by).agg(pl.len().alias("n"), pl.col("label").sum().alias("O"), pl.col("E").sum().alias("E"))
    t = t.with_columns((pl.col("O") / pl.col("E")).alias("O/E"),
                       ((pl.col("O") - pl.col("E")) / pl.col("E").sqrt()).alias("zp"))
    return t.filter(pl.col("n") >= min_n).sort(by)
