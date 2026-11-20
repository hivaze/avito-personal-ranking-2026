"""Cross-fitted residual LightGBM, key=value arguments.

usage: python resid_cv2.py obj=softmax base=score nl=3 lr=0.02 mdl=20000 l2=100 mh=50 ff=1.0 trees=400 fset=nobranch
base=s_to adds br_diff/s_ft as features (power check: the learner should recover the s_ft branch).
Appends one line per tree count to cv_log.txt.
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np

import prep
from prep import kit

a = dict(obj="softmax", base="score", nl="3", lr="0.02", mdl="20000", l2="100", mh="50", ff="1.0", trees="400",
         fset="nobranch", ef="0")
a.update(dict(x.split("=", 1) for x in sys.argv[1:]))
nl, lr, mdl, l2, mh, ff, T = int(a["nl"]), float(a["lr"]), int(a["mdl"]), float(a["l2"]), float(a["mh"]), \
    float(a["ff"]), int(a["trees"])
CHECK = [t for t in (10, 25, 50, 100, 200, 400, 800, 1600) if t <= T]
df = prep.derive(kit.load())
names = prep.feature_names(df)
if a["fset"] == "nobranch":
    names = [c for c in names if c not in ("s_ft", "br_diff")]
elif a["fset"] == "rel":   # relational / within-feed only: no absolute item-level or feed-level columns
    names = [c for c in names if c.startswith(("z_", "p_")) or c in (
        "rank", "rank_pct", "gap_top", "gap_next", "cat_rank_in_feed", "cat_rank_pct", "cat_feed_share",
        "cat_hist_minus_feed", "cat_user_rank", "cat_hist_share", "cat_newest_idx", "same_cat_newest",
        "hmax_idx", "hmax_type", "hmax_src", "n_hdup95", "n_cand_dup95", "pre2_fe_z", "pre2_new_z", "M", "H")]
GROUPS = dict(
    sim=["ucos", "hmax", "h1", "h5", "h10", "h32", "hmean5", "hmax_idx", "hmax_type", "hmax_src", "n_hdup95",
         "c_contact", "c_order", "c_fav", "c_lead", "c_vfeed", "c_vsearch", "cand_maxcos", "n_cand_dup95"],
    pop=["it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo",
         "xp_imp_c50", "f5", "f4", "seq_cv_invd20"],
    cat=["cat", "cat_views", "cat_actions", "cat_newest_idx", "cat_user_rank", "same_cat_newest", "cat_hist_share",
         "cat_n_in_feed", "cat_rank_in_feed", "cat_feed_share", "cat_hist_minus_feed", "cat_rank_pct", "f1", "f2",
         "f2_home", "f3"],
    ctx=["M", "H", "hour", "weekday", "newest_type", "newest_src", "newest_cat", "guest", "n_hist_cats32",
         "n_feed_cats", "gap_top", "gap_next", "pre2_fe_z", "pre2_new_z"])
if a["fset"] in GROUPS:
    g = GROUPS[a["fset"]]
    names = [c for c in prep.feature_names(df) if c in g or c[2:] in g] + ["rank", "rank_pct", "M"]
    names = list(dict.fromkeys(names))
if a["base"] != "score":
    names = [c for c in names if c != "score"] + ["br_diff"]
X = prep.matrix(df, names)
y = df["label"].to_numpy().astype(np.float64)
s = df[a["base"]].to_numpy().astype(np.float64)
off, sz = kit.feed_layout(df)
fold = kit.folds(df)
ffold = fold[off]
oof = {t: np.zeros(len(y)) for t in CHECK}
imp = np.zeros(len(names))
for k in range(5):
    tr, trf, te = fold != k, ffold != k, fold == k
    toff = np.r_[0, np.cumsum(sz[trf])[:-1]]
    fid_tr = np.repeat(np.arange(trf.sum()), sz[trf])
    npos = np.add.reduceat(y[tr], toff)
    yy = y[tr]
    c = prep.softmax_scale(s[tr], yy, toff, sz[trf])
    init = c * s

    def fobj(pred, data):
        m = np.maximum.reduceat(pred, toff)[fid_tr]
        e = np.exp(pred - m)
        pz = e / np.add.reduceat(e, toff)[fid_tr]
        w = 1.0 / np.maximum(npos, 1)[fid_tr]
        return (pz * npos[fid_tr] - yy) * w, np.maximum(pz * (1 - pz) * npos[fid_tr] * w, 1e-6)
    params = dict(objective=fobj, num_leaves=nl, learning_rate=lr, min_data_in_leaf=mdl, feature_fraction=ff,
                  lambda_l2=l2, min_sum_hessian_in_leaf=mh, num_threads=12, verbose=-1, seed=k,
                  extra_trees=a["ef"] == "1", bagging_fraction=0.8, bagging_freq=1)
    bst = lgb.train(params, lgb.Dataset(X[tr], yy, init_score=init[tr]), num_boost_round=T)
    imp += bst.feature_importance("gain")
    for t in CHECK:
        oof[t][te] = init[te] + bst.predict(X[te], num_iteration=t, raw_score=True)
tag = " ".join(f"{k}={v}" for k, v in a.items())
with open("cv_log.txt", "a") as fh:
    for t in CHECK:
        r = kit.evaluate(df, oof[t])   # always against the model score
        line = f"cv2 {tag} trees={t}: delta_vs_score={r['delta']:+.5f} se={r['se']:.5f} z={r['z']:.2f}"
        print(line)
        fh.write(line + "\n")
o = np.argsort(-imp)[:15]
print("gain:", ", ".join(f"{names[i]}={imp[i] / imp.sum():.3f}" for i in o))
