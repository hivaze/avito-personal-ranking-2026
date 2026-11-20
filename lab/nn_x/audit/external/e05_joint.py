"""e05: joint nested test of ALL label-free calendar x candidate priors at once (greedy coordinate ascent on 4 user-grouped folds, read on the
5th), to catch complementary small effects that single-feature tests cannot see. Priors (all fitted on random-like train feeds, holdout excluded):
 cat x weekday, cat x hour24, cat x (weekend x hour bucket), cat x day-trend slope, cold-day category deviation (Sep 6-8 vs 9-18),
 listing-age-bin x weekend, posting-hour and posting-weekday of the listing (< 2 weeks old), week-2-minus-week-1 category deviation.
Also the same on the holdout (train-period, has weekends). Then a LightGBM lambdarank residual (boosting from G3 as init_score) on the
same 9 columns + logM + category as a non-additive check, paired against the base residual model."""
from common_ext import *

v = meta("val"); tr = meta("train")
yv = np.asarray(cand("val", "target"), dtype=np.float64); sv = g3("val")
fidv = np.asarray(cand("val", "fidx")).astype(np.int64); catv = np.asarray(cand("val", "category_id")).astype(np.int64)
foldv = make_folds(v["groups"], 5, 0); zv = feed_z(sv, fidv, v["n"])
knots = np.load(FEATS / "life_f4_age_knots.npy")


def age_hours(x):
    x = np.asarray(x, dtype=np.float64); u = x * 64; k = np.clip(np.floor(u).astype(int), 1, 62); w = u - k
    return np.where(x < 63 / 64 - 1e-6, knots[k] * (1 - w) + knots[k + 1] * w, np.nan)


edges = np.array([0, 3, 12, 24, 72, 24 * 14, 24 * 90, 24 * 365, 1e9])
def age_bin(a):
    return np.clip(np.digitize(np.nan_to_num(a, nan=1e8), edges) - 1, 0, 7)
def posting(ts_row, a):
    t = ts_row - np.nan_to_num(a, nan=0.0) * 3600.0 + MSK
    return ((t // 3600) % 24).astype(np.int64), (((t // 86400) - DAY0 + 6) % 7).astype(np.int64)

hold, p = holdout_index()
is_hold = np.zeros(tr["n"], bool); is_hold[hold] = True
HOLD_SUB = np.sort(np.random.default_rng(7).choice(len(hold), 15000, replace=False))  # 15K-feed subsample of the holdout for speed
fit_feeds = np.where((p > 0.9) & (tr["M"] >= 50) & ~is_hold)[0]
rows = rows_of(tr["off"], tr["M"], fit_feeds)
ctr = np.asarray(cand("train", "category_id")[rows]).astype(np.int64); ytr = np.asarray(cand("train", "target")[rows]).astype(np.float64)
atr = age_hours(np.asarray(cand("train", "f4")[rows])); btr = age_bin(atr)
ffe = np.repeat(fit_feeds, tr["M"][fit_feeds])
day_tr = tr["day"][ffe]; dow_tr = tr["dow"][ffe]; hour_tr = tr["hour"][ffe]; wk_tr = (dow_tr >= 5).astype(np.int64); ts_tr = tr["ts"][ffe]
lr_dow, _, cnt_c, rate_c, _, _ = smoothed_logratio(ctr, dow_tr, 7, ytr)
lr_h24, _, _, _, _, _ = smoothed_logratio(ctr, hour_tr, 24, ytr)
lr_wh, _, _, _, _, _ = smoothed_logratio(ctr, wk_tr * 4 + hour_bucket(hour_tr), 8, ytr)
lr_day, _, _, _, pos_day, cnt_day = smoothed_logratio(ctr, day_tr, 24, ytr)
slope = np.zeros(64)
for c in range(64):
    dd = np.where(cnt_day[c] > 500)[0]
    if len(dd) < 6:
        continue
    r = (pos_day[c, dd] + 300 * rate_c[c]) / (cnt_day[c, dd] + 300); w = cnt_day[c, dd] / 1e4; x = dd - dd.mean(); yy = np.log(np.maximum(r, 1e-6))
    slope[c] = (w * x * yy).sum() / (w * x * x).sum()
mean_day = float(np.average(np.arange(24), weights=cnt_day.sum(0)))
lr_cold, _, _, _, _, _ = smoothed_logratio(ctr, (day_tr <= 2).astype(np.int64), 2, ytr)
lr_aw, _, _, _, _, _ = smoothed_logratio(btr, wk_tr, 2, ytr, NC=8, K=300.0)
ph_tr, pdw_tr = posting(ts_tr, atr); fresh_tr = np.nan_to_num(atr, nan=1e8) < 24 * 14
lr_ph, _, _, _, _, _ = smoothed_logratio(np.where(fresh_tr, 0, 1), ph_tr, 24, ytr, NC=2, K=300.0)
lr_pd, _, _, _, _, _ = smoothed_logratio(np.where(fresh_tr, 0, 1), pdw_tr, 7, ytr, NC=2, K=300.0)
lr_w2, _, _, _, _, _ = smoothed_logratio(ctr, (day_tr >= 6).astype(np.int64), 2, ytr)


def build(cat, dow, hour, day, ts, f4):
    a = age_hours(f4); b = age_bin(a); ph, pdw = posting(ts, a); fresh = np.nan_to_num(a, nan=1e8) < 24 * 14; wk = (dow >= 5).astype(np.int64)
    cols = {
        "cat_x_dow": lr_dow[cat, dow], "cat_x_hour24": lr_h24[cat, hour], "cat_x_wkhour": lr_wh[cat, wk * 4 + hour_bucket(hour)],
        "cat_trend": slope[cat] * (day - mean_day), "cold_dev": lr_cold[cat, 1] - lr_cold[cat, 0], "age_x_wk": lr_aw[b, wk],
        "post_hour": np.where(fresh, lr_ph[0, ph], 0.0), "post_dow": np.where(fresh, lr_pd[0, pdw], 0.0), "week2_dev": lr_w2[cat, 1] - lr_w2[cat, 0]}
    return cols


def greedy(base, cols, y, off, M, fold, steps=6):
    names = list(cols); X = np.stack([cols[n] for n in names], 1)
    grid = np.array([-0.3, -0.15, -0.05, 0.05, 0.15, 0.3])
    n = len(M); oof = np.zeros(n); base_nd = ndcg_per_feed(base, y, off, M)
    for k in np.unique(fold):
        trm = fold != k; w = np.zeros(len(names)); cur = base.copy(); best = ndcg_per_feed(cur, y, off, M)[trm].mean()
        for _ in range(steps):
            improved = False
            for j in range(len(names)):
                for g in grid:
                    s = ndcg_per_feed(cur + g * X[:, j], y, off, M)[trm].mean()
                    if s > best + 1e-6:
                        best, bj, bg, improved = s, j, g, True
            if not improved:
                break
            w[bj] += bg; cur = cur + bg * X[:, bj]
        oof[fold == k] = ndcg_per_feed(cur, y, off, M)[fold == k]
        log(f"  fold {k}: weights " + ", ".join(f"{nm} {ww:+.2f}" for nm, ww in zip(names, w) if ww != 0) + f"; in-fold gain {best - base_nd[trm].mean():+.5f}")
    return oof, base_nd


log("== val: greedy joint nested over 9 calendar priors ==")
cols_v = build(catv, v["dow"][fidv], v["hour"][fidv], v["day"][fidv], v["ts"][fidv], np.asarray(cand("val", "f4")))
oof, bnd = greedy(zv, cols_v, yv, v["off"], v["M"], foldv)
sl = {"Tue": v["dow"] == 1, "Thu": v["dow"] == 3, "M<50": v["M"] < 50, "M>=50": v["M"] >= 50}
log(f"VAL joint OOF {fmt(oof, bnd)}; " + "; ".join(f"{k} {fmt(oof[m], bnd[m])}" for k, m in sl.items()))

log("== holdout: greedy joint nested ==")
gh, used, Mh = g3_holdout_pool()
yh = np.load(PREDS / "holdout/holdout_y.npy").astype(np.float64)
offh = np.concatenate([[0], np.cumsum(Mh)[:-1]]); fidh = np.repeat(np.arange(len(hold)), Mh)
rows_h = rows_of(tr["off"], tr["M"], hold)
cath = np.asarray(cand("train", "category_id")[rows_h]).astype(np.int64)
foldh = make_folds(np.where(tr["user"][hold] == 0, -1 - np.arange(len(hold)), tr["user"][hold]), 5, 0)
zbh = feed_z(gh, fidh, len(hold))
cols_h = build(cath, tr["dow"][hold][fidh], tr["hour"][hold][fidh], tr["day"][hold][fidh], tr["ts"][hold][fidh], np.asarray(cand("train", "f4")[rows_h]))
sub_rows = np.isin(fidh, HOLD_SUB); Ms = Mh[HOLD_SUB]; offs = np.concatenate([[0], np.cumsum(Ms)[:-1]])
cols_s = {k: vv[sub_rows] for k, vv in cols_h.items()}
oofh, bndh = greedy(zbh[sub_rows], cols_s, yh[sub_rows], offs, Ms, foldh[HOLD_SUB])
wkm = (tr["dow"][hold] >= 5)[HOLD_SUB]
log(f"HOLDOUT (15K subsample) joint OOF {fmt(oofh, bndh)}; weekend {fmt(oofh[wkm], bndh[wkm])}; weekday {fmt(oofh[~wkm], bndh[~wkm])}")

log("== val: LightGBM residual from G3 (init_score) with the 9 priors + logM + category vs base [logM, category] ==")
import lightgbm as lgb
def resid(Xcols, seed):
    X = np.stack(Xcols, 1).astype(np.float32); n = v["n"]; row_fold = foldv[fidv]; oof = np.zeros(len(yv))
    params = dict(objective="lambdarank", metric="None", learning_rate=0.03, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=60, num_threads=12, verbose=-1, seed=seed, max_bin=63)
    for k in range(5):
        trm = row_fold != k; tem = ~trm
        ds = lgb.Dataset(X[trm], label=yv[trm], group=v["M"][foldv != k], init_score=4 * zv[trm], free_raw_data=True)
        b = lgb.train(params, ds, num_boost_round=150)
        oof[tem] = 4 * zv[tem] + b.predict(X[tem], num_threads=12)
    return ndcg_per_feed(oof, yv, v["off"], v["M"])
logM = np.log(v["M"])[fidv].astype(np.float64)
base_cols = [logM, catv.astype(np.float64)]
full_cols = base_cols + [cols_v[n] for n in cols_v]
for seed in [0, 1]:
    nb = resid(base_cols, seed); nf = resid(full_cols, seed)
    log(f"seed {seed}: base residual vs G3 {fmt(nb, bnd)}; +9 priors vs G3 {fmt(nf, bnd)}; +9 priors vs base residual {fmt(nf, nb)}")
log("done")
