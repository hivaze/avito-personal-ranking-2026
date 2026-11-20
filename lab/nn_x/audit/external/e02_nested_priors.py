"""e02: does any calendar x category (or calendar x listing-age) demand prior add to G3? Nested additive tests, user-grouped 5-fold.
Priors are fitted on random-like train feeds (M>=50, p_random>0.9, holdout excluded; 9.2M rows) unless stated:
 P1 category x weekday (7 levels)            -> val (Tue/Thu)
 P2 category x hour (24 levels)              -> val, holdout
 P3 category x day linear trend (slope per category over the 12 train days), extrapolated to Sep 22/24 (val) and Sep 26-29 (test)
 P4 category x (weekend x hour bucket)       -> holdout (train has weekends)
 P5 forward 'yesterday's demand': val Tue -> Thu per-category rate log-ratio (and Thu -> Tue as the mirror); G3 per-category act/exp on Tue -> Thu
 P6 ORACLE ceiling: val's own same-day category rate, (a) user-grouped OOF, (b) in-sample -> the most any external per-day category demand series could add
 P7 listing-age-bin x weekend lift (train random-like) -> holdout weekend feeds
 P8 within-train weekend->weekday replication: priors fitted on week 1 (Sep 6-12) random-like feeds, nested on the week-2 holdout feeds
 P9 production-feed priors (all production rows, 230M) for category x weekday and x hour, as a second (bigger, biased) source -> val
Also writes out/e02_test_priors.npz: per-row test values of the P1/P2/P3 priors for Sep 26-29, in case anyone wants them."""
from common_ext import *

v = meta("val"); te = meta("test"); tr = meta("train")
yv = np.asarray(cand("val", "target"), dtype=np.float64)
sv = g3("val")
fidv = np.asarray(cand("val", "fidx")).astype(np.int64)
catv = np.asarray(cand("val", "category_id")).astype(np.int64)
catt = np.asarray(cand("test", "category_id")).astype(np.int64)
fidt = np.asarray(cand("test", "fidx")).astype(np.int64)
foldv = make_folds(v["groups"], 5, 0)
zv = feed_z(sv, fidv, v["n"])
sl_v = {"Tue": v["dow"] == 1, "Thu": v["dow"] == 3, "M<50": v["M"] < 50, "M>=50": v["M"] >= 50}
log(f"val G3 {ndcg_per_feed(sv, yv, v['off'], v['M']).mean():.5f}")

log("== fit tables on random-like train feeds ==")
hold, p = holdout_index()
is_hold = np.zeros(tr["n"], bool); is_hold[hold] = True
fit_feeds = np.where((p > 0.9) & (tr["M"] >= 50) & ~is_hold)[0]
rows = rows_of(tr["off"], tr["M"], fit_feeds)
ctr = np.asarray(cand("train", "category_id")[rows]).astype(np.int64)
ytr = np.asarray(cand("train", "target")[rows]).astype(np.float64)
f4tr = np.asarray(cand("train", "f4")[rows]).astype(np.float64)
ffe = np.repeat(fit_feeds, tr["M"][fit_feeds])
day_tr = tr["day"][ffe]; dow_tr = tr["dow"][ffe]; hour_tr = tr["hour"][ffe]; wk_tr = (dow_tr >= 5).astype(np.int64)
ND = 24
log(f"fit rows {len(rows)} from {len(fit_feeds)} feeds; days present: {sorted(set(day_tr.tolist()))}")
lr_dow, chi_dow, cnt_c, rate_c, _, cnt_dow = smoothed_logratio(ctr, dow_tr, 7, ytr)
lr_h24, chi_h24, _, _, _, _ = smoothed_logratio(ctr, hour_tr, 24, ytr)
lr_hb, _, _, _, _, _ = smoothed_logratio(ctr, hour_bucket(hour_tr), 4, ytr)
lr_wh, _, _, _, _, _ = smoothed_logratio(ctr, wk_tr * 4 + hour_bucket(hour_tr), 8, ytr)
lr_day, chi_day, _, _, pos_day, cnt_day = smoothed_logratio(ctr, day_tr, ND, ytr)
log(f"category x weekday: categories with chi2 p<0.001 (df 6, 22.5): {(chi_dow > 22.5).sum()} of {(cnt_c > 0).sum()}; "
    f"x hour24: {(chi_h24 > 49.7).sum()}; x day: {(chi_day > 31.3).sum()}")

# P3: per-category linear trend of log rate over train days (weighted LS), slope per day
slope = np.zeros(64); slope_se = np.zeros(64)
for c in range(64):
    dd = np.where(cnt_day[c] > 500)[0]
    if len(dd) < 6:
        continue
    r = (pos_day[c, dd] + 300 * rate_c[c]) / (cnt_day[c, dd] + 300)
    w = cnt_day[c, dd] / 1e4
    x = dd - dd.mean()
    yy = np.log(np.maximum(r, 1e-6))
    b = (w * x * yy).sum() / (w * x * x).sum()
    resid = yy - b * x - (w * yy).sum() / w.sum()
    slope[c] = b
    slope_se[c] = np.sqrt((w * resid ** 2).sum() / w.sum() / (w * x * x).sum() * w.sum() / max(len(dd) - 2, 1))
big = np.argsort(-cnt_c)[:15]
log("P3 per-category trend slope (log-rate per day) top-15 by rows: " + ", ".join(f"cat {c} {slope[c]:+.4f}±{slope_se[c]:.4f}" for c in big))
log(f"P3 slopes with |t|>3: {(np.abs(slope) > 3 * np.maximum(slope_se, 1e-9)).sum()}; exposure-weighted |slope| {float((cnt_c / cnt_c.sum() * np.abs(slope)).sum()):.4f}/day -> over 10 days {10 * float((cnt_c / cnt_c.sum() * np.abs(slope)).sum()):.3f} nats")
mean_day = float(np.average(np.arange(ND), weights=cnt_day.sum(0)))

log("== nested tests on val (G3 within-feed z + w * prior) ==")
dow_row = v["dow"][fidv]; hour_row = v["hour"][fidv]; day_row = v["day"][fidv]
P = {}
P["P1 cat x weekday (train random-like)"] = lr_dow[catv, dow_row]
P["P2 cat x hour24 (train random-like)"] = lr_h24[catv, hour_row]
P["P2b cat x hour bucket (train random-like)"] = lr_hb[catv, hour_bucket(hour_row)]
P["P3 cat x day trend extrapolated (slope*(day-mean))"] = slope[catv] * (day_row - mean_day)
P["P3b cat x day trend, t>3 slopes only"] = np.where(np.abs(slope) > 3 * np.maximum(slope_se, 1e-9), slope, 0)[catv] * (day_row - mean_day)
P["ctl: category rate prior (log rate_c)"] = np.log(np.maximum(rate_c, 1e-6))[catv]
rng = np.random.default_rng(1)
P["ctl: random cat x weekday table (same sd)"] = (rng.normal(0, lr_dow[lr_dow != 0].std(), lr_dow.shape))[catv, dow_row]
for name, x in P.items():
    nested_report(name, zv, x, yv, v["off"], v["M"], foldv, slices=sl_v)

log("== P5 forward-in-time: val Tue -> Thu (and the mirror) ==")
tue = v["dow"] == 1; thu = v["dow"] == 3
tue_r = tue[fidv]; thu_r = thu[fidv]
# per-category log-ratio of the positive rate on Tue vs the train random-like category rate
def cat_lr(mask_rows, K=100.0):
    pos = np.bincount(catv[mask_rows], weights=yv[mask_rows], minlength=64)
    cnt = np.bincount(catv[mask_rows], minlength=64).astype(np.float64)
    sm = (pos + K * rate_c) / (cnt + K)
    return np.log(np.maximum(sm, 1e-6)) - np.log(np.maximum(rate_c, 1e-6))
lr_tue = cat_lr(tue_r); lr_thu = cat_lr(thu_r)
log(f"corr(lr_tue, lr_thu) over categories with >1000 val rows: {np.corrcoef(lr_tue[cnt_c > 1000], lr_thu[cnt_c > 1000])[0, 1]:+.3f}; sd lr_tue {lr_tue[cnt_c > 1000].std():.3f}")
def sub(mask_feed):
    rows_ = mask_feed[fidv]
    M_ = v["M"][mask_feed]
    off_ = np.concatenate([[0], np.cumsum(M_)[:-1]])
    return rows_, M_, off_, foldv[mask_feed]
r_, M_, off_, f_ = sub(thu)
nested_report("P5 Tue-fitted cat rate -> Thu feeds", zv[r_], lr_tue[catv[r_]], yv[r_], off_, M_, f_)
nested_report("P5 Tue-fitted cat rate -> Thu feeds, 'same-day' (Thu-fitted, IN-SAMPLE ceiling)", zv[r_], lr_thu[catv[r_]], yv[r_], off_, M_, f_)
r2, M2, off2, f2 = sub(tue)
nested_report("P5 mirror: Thu-fitted cat rate -> Tue feeds", zv[r2], lr_thu[catv[r2]], yv[r2], off2, M2, f2)
nested_report("P5 mirror: Tue-fitted -> Tue (IN-SAMPLE ceiling)", zv[r2], lr_tue[catv[r2]], yv[r2], off2, M2, f2)
# G3 calibration by category: act/exp with exp from the val-wide P(y | G3 rank) curve
order = np.lexsort((-sv, fidv))
rank = np.empty(len(sv), np.int64); rank[order] = np.arange(len(sv)) - np.repeat(v["off"], v["M"])
rk = np.minimum(rank, 60)
p_rank = np.bincount(rk, weights=yv, minlength=61) / np.maximum(np.bincount(rk, minlength=61), 1)
expv = p_rank[rk]
def cat_actexp(mask_rows, K=50.0):
    act = np.bincount(catv[mask_rows], weights=yv[mask_rows], minlength=64)
    ex = np.bincount(catv[mask_rows], weights=expv[mask_rows], minlength=64)
    return np.log((act + K * 1.0) / (ex + K * 1.0))
ae_tue = cat_actexp(tue_r); ae_thu = cat_actexp(thu_r); ae_all = cat_actexp(np.ones(len(yv), bool))
log(f"G3 per-category log(act/exp): all-val sd {ae_all[cnt_c > 1000].std():.3f}; corr(Tue, Thu) {np.corrcoef(ae_tue[cnt_c > 1000], ae_thu[cnt_c > 1000])[0, 1]:+.3f}")
nested_report("P5c G3 category act/exp on Tue -> Thu feeds", zv[r_], ae_tue[catv[r_]], yv[r_], off_, M_, f_)
nested_report("P5c mirror: Thu act/exp -> Tue feeds", zv[r2], ae_thu[catv[r2]], yv[r2], off2, M2, f2)

log("== P6 oracle ceiling: val same-day category rate (user-grouped OOF and in-sample) ==")
x_oof = np.zeros(len(yv)); x_oof_day = np.zeros(len(yv))
for k in range(5):
    trm = foldv[fidv] != k; tem = ~trm
    lr_k = cat_lr(trm)
    x_oof[tem] = lr_k[catv[tem]]
    for dd, dm in [(16, tue_r), (18, thu_r)]:
        lr_kd = cat_lr(trm & dm)
        x_oof_day[tem & dm] = lr_kd[catv[tem & dm]]
nested_report("P6a val category rate, user-OOF (no day)", zv, x_oof, yv, v["off"], v["M"], foldv, slices=sl_v)
nested_report("P6b val category x day rate, user-OOF", zv, x_oof_day, yv, v["off"], v["M"], foldv, slices=sl_v)
x_in = np.where(tue_r, lr_tue[catv], lr_thu[catv])
nested_report("P6c val category x day rate, IN-SAMPLE (ceiling, leaky)", zv, x_in, yv, v["off"], v["M"], foldv, slices=sl_v)
# finer: category x day x hour-bucket, in-sample ceiling
key = catv * 8 + np.where(tue_r, 0, 4) + hour_bucket(hour_row)
pos_k = np.bincount(key, weights=yv, minlength=64 * 8); cnt_k = np.bincount(key, minlength=64 * 8).astype(np.float64)
sm_k = (pos_k + 100 * np.repeat(rate_c, 8)) / (cnt_k + 100)
x_in2 = np.log(np.maximum(sm_k, 1e-6))[key] - np.log(np.maximum(rate_c, 1e-6))[catv]
nested_report("P6d val category x day x hour-bucket, IN-SAMPLE (ceiling, leaky)", zv, x_in2, yv, v["off"], v["M"], foldv, slices=sl_v)

log("== holdout tests (train-period random-like feeds, 32% weekend) ==")
gh, used, Mh = g3_holdout_pool()
yh = np.load(PREDS / "holdout/holdout_y.npy").astype(np.float64)
offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
fidh = np.repeat(np.arange(len(hold)), Mh)
rows_h = rows_of(tr["off"], tr["M"], hold)
cath = np.asarray(cand("train", "category_id")[rows_h]).astype(np.int64)
f4h = np.asarray(cand("train", "f4")[rows_h]).astype(np.float64)
assert np.allclose(np.asarray(cand("train", "target")[rows_h]).astype(np.float64), yh)
foldh = make_folds(np.where(tr["user"][hold] == 0, -1 - np.arange(len(hold)), tr["user"][hold]), 5, 0)
zbh = feed_z(gh, fidh, len(hold))
dow_h = tr["dow"][hold][fidh]; hour_h = tr["hour"][hold][fidh]; day_h = tr["day"][hold][fidh]; wk_h = (dow_h >= 5).astype(np.int64)
sl_h = {"weekend": (tr["dow"][hold] >= 5), "weekday": (tr["dow"][hold] < 5), "week2": tr["day"][hold] >= 7}
log(f"holdout G3 pool ({len(used)} members) {ndcg_per_feed(gh, yh, offh, Mh).mean():.5f}; weekend feed share {sl_h['weekend'].mean():.3f}")
nested_report("H-P1 cat x weekday", zbh, lr_dow[cath, dow_h], yh, offh, Mh, foldh, slices=sl_h)
nested_report("H-P2 cat x hour24", zbh, lr_h24[cath, hour_h], yh, offh, Mh, foldh, slices=sl_h)
nested_report("H-P4 cat x (weekend x hour bucket)", zbh, lr_wh[cath, wk_h * 4 + hour_bucket(hour_h)], yh, offh, Mh, foldh, slices=sl_h)
nested_report("H-P3 cat x day (same-period, in-sample table; ceiling)", zbh, lr_day[cath, day_h], yh, offh, Mh, foldh, slices=sl_h)
# P7 listing-age bin x weekend
knots = np.load(FEATS / "life_f4_age_knots.npy", allow_pickle=True)
def age_hours(f4):
    # life_f4_age_knots: monotone map f4 -> age; fall back to rank bins if the knot file is not a simple table
    try:
        k = np.asarray(knots, dtype=np.float64)
        if k.ndim == 2 and k.shape[0] == 2:
            return np.interp(f4, k[0], k[1])
        if k.ndim == 2 and k.shape[1] == 2:
            return np.interp(f4, k[:, 0], k[:, 1])
    except Exception:
        pass
    return None
ah = age_hours(f4h); atr = age_hours(f4tr)
if ah is not None:
    edges = np.array([0, 3, 12, 24, 72, 24 * 14, 24 * 90, 24 * 365, 1e9])
    bh = np.digitize(ah, edges) - 1; btr = np.digitize(atr, edges) - 1
    lr_aw, chi_aw, _, _, pos_aw, cnt_aw = smoothed_logratio(btr, wk_tr, 2, ytr, NC=9, K=300.0)
    log("P7 train random-like: age-bin x weekend log-ratio (weekday, weekend) per bin [<3h,3-12h,12-24h,1-3d,3-14d,2w-3m,3m-1y,>1y]: " +
        " ".join(f"[{lr_aw[b, 0]:+.3f},{lr_aw[b, 1]:+.3f}]" for b in range(8)) + f"; chi2 {chi_aw[:8].sum():.1f} on 8 df")
    nested_report("H-P7 age-bin x weekend", zbh, lr_aw[bh, wk_h], yh, offh, Mh, foldh, slices=sl_h)
else:
    log("P7 skipped: age knots file not a simple table; shape " + str(np.asarray(knots).shape))

log("== P8 within-train replication: fit on week 1 (Sep 6-12), test on week-2 holdout feeds (Sep 13-19) ==")
w1 = day_tr <= 6
lr_dow1, _, _, rate_c1, _, _ = smoothed_logratio(ctr[w1], dow_tr[w1], 7, ytr[w1])
lr_wk1, _, _, _, _, _ = smoothed_logratio(ctr[w1], wk_tr[w1], 2, ytr[w1])
lr_h1, _, _, _, _, _ = smoothed_logratio(ctr[w1], hour_tr[w1], 24, ytr[w1])
lr_day1, _, _, _, pos_d1, cnt_d1 = smoothed_logratio(ctr[w1], day_tr[w1], ND, ytr[w1])
slope1 = np.zeros(64)
for c in range(64):
    dd = np.where(cnt_d1[c] > 500)[0]
    if len(dd) < 5:
        continue
    r = (pos_d1[c, dd] + 300 * rate_c1[c]) / (cnt_d1[c, dd] + 300)
    w = cnt_d1[c, dd] / 1e4; x = dd - dd.mean(); yy = np.log(np.maximum(r, 1e-6))
    slope1[c] = (w * x * yy).sum() / (w * x * x).sum()
w2 = tr["day"][hold] >= 7
r2h = w2[fidh]; M2h = Mh[w2]; off2h = np.concatenate([[0], np.cumsum(M2h)[:-1]]); f2h = foldh[w2]
sl2 = {"weekend": tr["dow"][hold][w2] >= 5, "weekday": tr["dow"][hold][w2] < 5}
log(f"week-2 holdout feeds {w2.sum()} (weekend share {sl2['weekend'].mean():.3f}); week-1 fit rows {w1.sum()}")
nested_report("P8 week1 cat x weekend -> week2 holdout", zbh[r2h], lr_wk1[cath[r2h], wk_h[r2h]], yh[r2h], off2h, M2h, f2h, slices=sl2)
nested_report("P8 week1 cat x weekday -> week2 holdout", zbh[r2h], lr_dow1[cath[r2h], dow_h[r2h]], yh[r2h], off2h, M2h, f2h, slices=sl2)
nested_report("P8 week1 cat x hour24 -> week2 holdout", zbh[r2h], lr_h1[cath[r2h], hour_h[r2h]], yh[r2h], off2h, M2h, f2h, slices=sl2)
nested_report("P8 week1 cat x day trend -> week2 holdout (extrapolated)", zbh[r2h], slope1[cath[r2h]] * (day_h[r2h] - 3.0), yh[r2h], off2h, M2h, f2h, slices=sl2)
nested_report("P8 week1 category rate -> week2 holdout (control)", zbh[r2h], np.log(np.maximum(rate_c1, 1e-6))[cath[r2h]], yh[r2h], off2h, M2h, f2h, slices=sl2)
# same-period (week-2 in-sample table) ceiling on the same feeds
w2r = ~w1
lr_day2, _, _, _, _, _ = smoothed_logratio(ctr[w2r], day_tr[w2r], ND, ytr[w2r])
nested_report("P8 ceiling: week-2 cat x day table (same period, other feeds) -> week2 holdout", zbh[r2h], lr_day2[cath[r2h], day_h[r2h]], yh[r2h], off2h, M2h, f2h, slices=sl2)

log("== P9 priors from PRODUCTION train rows (biased, 20x more data) -> val ==")
prod_feeds = np.where(~((p > 0.9) & (tr["M"] >= 50)))[0]
rng = np.random.default_rng(0)
prod_feeds = np.sort(rng.choice(prod_feeds, 600000, replace=False))
rows_p = rows_of(tr["off"], tr["M"], prod_feeds)
cp = np.asarray(cand("train", "category_id")[rows_p]).astype(np.int64)
yp = np.asarray(cand("train", "target")[rows_p]).astype(np.float64)
fp = np.repeat(prod_feeds, tr["M"][prod_feeds])
lr_dow_p, chi_dow_p, _, _, _, _ = smoothed_logratio(cp, tr["dow"][fp], 7, yp)
lr_h_p, chi_h_p, _, _, _, _ = smoothed_logratio(cp, tr["hour"][fp], 24, yp)
log(f"production sample rows {len(cp)}: cat x weekday chi2 p<0.001: {(chi_dow_p > 22.5).sum()}; cat x hour: {(chi_h_p > 49.7).sum()}; corr with random-like tables: dow {np.corrcoef(lr_dow_p[cnt_c > 1000].ravel(), lr_dow[cnt_c > 1000].ravel())[0, 1]:+.3f}, hour {np.corrcoef(lr_h_p[cnt_c > 1000].ravel(), lr_h24[cnt_c > 1000].ravel())[0, 1]:+.3f}")
nested_report("P9 production cat x weekday -> val", zv, lr_dow_p[catv, dow_row], yv, v["off"], v["M"], foldv, slices=sl_v)
nested_report("P9 production cat x hour24 -> val", zv, lr_h_p[catv, hour_row], yv, v["off"], v["M"], foldv, slices=sl_v)

log("== write test-row priors for Sep 26-29 ==")
dow_t = te["dow"][fidt]; hour_t = te["hour"][fidt]; day_t = te["day"][fidt]
np.savez(OUT / "e02_test_priors.npz", p1_cat_dow=lr_dow[catt, dow_t].astype(np.float32), p2_cat_hour=lr_h24[catt, hour_t].astype(np.float32),
         p3_cat_trend=(slope[catt] * (day_t - mean_day)).astype(np.float32), slope=slope, slope_se=slope_se, lr_dow=lr_dow, lr_h24=lr_h24, mean_day=mean_day)
log("test rows by day: " + ", ".join(f"{day_label(d)}: {int((day_t == d).sum())}" for d in np.unique(day_t)))
log("done")
