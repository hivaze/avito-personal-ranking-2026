"""e03: calendar x listing-age interactions, listing posting time, and the in-train weather contrast.
 A. listing-age composition by day (share of <1 d / 1-7 d / >30 d candidates per day; val/test/train) and the age lift by weekday vs weekend
    (random-like train) -> nested age-bin x weekend prior on the holdout (train has weekends).
 B. POSTING time of the listing = feed timestamp - age: posting hour-of-day and posting weekday of the candidate (a candidate attribute, not a
    feed constant; private sellers post evenings/weekends, businesses post on weekday mornings) -> nested on val and holdout.
 C. Weather contrast inside train: Sep 6-8 were cold (coldest night since May on Sep 9), Sep 9-18 was the warm 'Indian summer', val/test are
    cold again (autumn from Sep 21, Arctic air Sep 27-29). Per-category 'cold-day' deviation fitted on train (random-like and production)
    -> nested on val (cold period). If weather-sensitive categories exist, the cold-day deviation should transfer to val.
 D. Second-week regime: the candidate category mix shifts from Sep 12 (JS x1e4 5-7 vs 0.4-1.8 before). Category deviation of week 2 vs week 1
    (production, all rows) -> nested on val, next to the week-1 and all-train versions (does the most recent week transfer better?)."""
from common_ext import *

knots = np.load(FEATS / "life_f4_age_knots.npy")
K = 64; CAP = 63 / 64 - 1e-6


def age_hours(x):
    x = np.asarray(x, dtype=np.float64)
    u = x * K
    k = np.clip(np.floor(u).astype(int), 1, K - 2)
    w = u - k
    a = knots[k] * (1 - w) + knots[k + 1] * w
    return np.where(x < CAP, a, np.nan)


edges = np.array([0, 1, 3, 6, 12, 24, 48, 72, 168, 336, 720, 1440, 2160, 4320, 8760, 1e9])
labels = ["<1h", "1-3h", "3-6h", "6-12h", "12-24h", "1-2d", "2-3d", "3-7d", "1-2w", "2-4w", "1-2m", "2-3m", "3-6m", "6-12m", ">1y"]
NB = len(labels)


def age_bin(a):
    b = np.digitize(np.nan_to_num(a, nan=1e8), edges) - 1
    return np.clip(b, 0, NB - 1)


v = meta("val"); te = meta("test"); tr = meta("train")
yv = np.asarray(cand("val", "target"), dtype=np.float64)
sv = g3("val")
fidv = np.asarray(cand("val", "fidx")).astype(np.int64)
catv = np.asarray(cand("val", "category_id")).astype(np.int64)
av = age_hours(np.asarray(cand("val", "f4"))); bv = age_bin(av)
at = age_hours(np.asarray(cand("test", "f4"))); bt = age_bin(at)
fidt = np.asarray(cand("test", "fidx")).astype(np.int64)
foldv = make_folds(v["groups"], 5, 0)
zv = feed_z(sv, fidv, v["n"])
sl_v = {"Tue": v["dow"] == 1, "Thu": v["dow"] == 3, "M<50": v["M"] < 50, "M>=50": v["M"] >= 50}

log("== A. listing-age composition by day ==")
def age_comp(b, days, name):
    for dd in np.unique(days):
        m = days == dd
        if m.sum() < 10000:
            continue
        sh = np.bincount(b[m], minlength=NB) / m.sum()
        log(f"{name:8s} {day_label(dd)} rows {m.sum():8d} share <1d {sh[:5].sum():.3f} 1-7d {sh[5:8].sum():.3f} 1-4w {sh[8:10].sum():.3f} >1m {sh[10:].sum():.3f} (>1y {sh[14]:.3f})")
age_comp(bv, v["day"][fidv], "val")
age_comp(bt, te["day"][fidt], "test")
hold, p = holdout_index()
is_hold = np.zeros(tr["n"], bool); is_hold[hold] = True
fit_feeds = np.where((p > 0.9) & (tr["M"] >= 50) & ~is_hold)[0]
rows = rows_of(tr["off"], tr["M"], fit_feeds)
ctr = np.asarray(cand("train", "category_id")[rows]).astype(np.int64)
ytr = np.asarray(cand("train", "target")[rows]).astype(np.float64)
atr = age_hours(np.asarray(cand("train", "f4")[rows])); btr = age_bin(atr)
ffe = np.repeat(fit_feeds, tr["M"][fit_feeds])
day_tr = tr["day"][ffe]; dow_tr = tr["dow"][ffe]; hour_tr = tr["hour"][ffe]; wk_tr = (dow_tr >= 5).astype(np.int64); ts_tr = tr["ts"][ffe]
age_comp(btr, day_tr, "trainRL")
# age lift by weekday/weekend in random-like train
lr_aw, chi_aw, _, _, pos_aw, cnt_aw = smoothed_logratio(btr, wk_tr, 2, ytr, NC=NB, K=300.0)
rate_b = pos_aw.sum(1) / np.maximum(cnt_aw.sum(1), 1)
log("random-like train: positive-rate lift by age bin, overall / weekday / weekend:")
for b in range(NB):
    log(f"  {labels[b]:6s} rows {int(cnt_aw[b].sum()):8d} lift {rate_b[b] / ytr.mean():.2f}  log-ratio weekday {lr_aw[b, 0]:+.3f} weekend {lr_aw[b, 1]:+.3f}")
log(f"age x weekend chi2 {chi_aw.sum():.1f} on {NB} df")
lr_adow, chi_adow, _, _, _, _ = smoothed_logratio(btr, dow_tr, 7, ytr, NC=NB, K=300.0)
log(f"age x weekday(7) chi2 {chi_adow.sum():.1f} on {6 * NB} df")

log("== holdout: age x weekend / age x weekday priors ==")
gh, used, Mh = g3_holdout_pool()
yh = np.load(PREDS / "holdout/holdout_y.npy").astype(np.float64)
offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
fidh = np.repeat(np.arange(len(hold)), Mh)
rows_h = rows_of(tr["off"], tr["M"], hold)
cath = np.asarray(cand("train", "category_id")[rows_h]).astype(np.int64)
ah = age_hours(np.asarray(cand("train", "f4")[rows_h])); bh = age_bin(ah)
foldh = make_folds(np.where(tr["user"][hold] == 0, -1 - np.arange(len(hold)), tr["user"][hold]), 5, 0)
zbh = feed_z(gh, fidh, len(hold))
dow_h = tr["dow"][hold][fidh]; wk_h = (dow_h >= 5).astype(np.int64); ts_h = tr["ts"][hold][fidh]; hour_h = tr["hour"][hold][fidh]
sl_h = {"weekend": (tr["dow"][hold] >= 5), "weekday": (tr["dow"][hold] < 5)}
nested_report("H age-bin x weekend", zbh, lr_aw[bh, wk_h], yh, offh, Mh, foldh, slices=sl_h)
nested_report("H age-bin x weekday(7)", zbh, lr_adow[bh, dow_h], yh, offh, Mh, foldh, slices=sl_h)
lr_a, _, _, _, _, _ = smoothed_logratio(btr, np.zeros(len(btr), np.int64), 1, ytr, NC=NB, K=300.0)
lr_ab = np.log(np.maximum(rate_b, 1e-6)) - np.log(ytr.mean())
nested_report("H ctl: age-bin prior alone (no calendar)", zbh, lr_ab[bh], yh, offh, Mh, foldh, slices=sl_h)
nested_report("V ctl: age-bin prior alone (no calendar) -> val", zv, lr_ab[bv], yv, v["off"], v["M"], foldv, slices=sl_v)

log("== B. listing POSTING time (feed ts - age) ==")
def posting(ts_row, a_hours):
    pt = ts_row - np.nan_to_num(a_hours, nan=0.0) * 3600.0
    t = pt + MSK
    ph = ((t // 3600) % 24).astype(np.int64)
    pdw = (((t // 86400) - DAY0 + 6) % 7).astype(np.int64)
    return ph, pdw
# resolution check: knots spacing in hours by bin
log("f4 code resolution (hours between consecutive knots) at ages 1h/1d/1w/1m: " +
    ", ".join(f"{labels[b]}: {np.diff(knots)[np.searchsorted(knots, edges[b + 1]) - 1 if np.searchsorted(knots, edges[b + 1]) > 0 else 0]:.1f}h" for b in [1, 5, 8, 10]))
ph_tr, pdw_tr = posting(ts_tr, atr)
fresh_tr = np.nan_to_num(atr, nan=1e8) < 24 * 14  # posting time is only meaningful (< ~6 h resolution) within ~2 weeks
lr_ph, chi_ph, _, _, pos_ph, cnt_ph = smoothed_logratio(np.where(fresh_tr, 0, 1), ph_tr, 24, ytr, NC=2, K=300.0)
lr_pd, chi_pd, _, _, pos_pd, cnt_pd = smoothed_logratio(np.where(fresh_tr, 0, 1), pdw_tr, 7, ytr, NC=2, K=300.0)
log("random-like train, listings < 2 weeks old: positive-rate log-ratio by POSTING hour: " + " ".join(f"{h}:{lr_ph[0, h]:+.2f}" for h in range(24)) + f"; chi2 {chi_ph[0]:.1f} on 23 df")
log("  by POSTING weekday (Mon..Sun): " + " ".join(f"{lr_pd[0, d]:+.3f}" for d in range(7)) + f"; chi2 {chi_pd[0]:.1f} on 6 df; rows {int(cnt_pd[0].sum())}")
# also category-conditional posting-hour (business vs private sellers differ by category)
# nested on val and holdout
ph_v, pdw_v = posting(v["ts"][fidv], av); fresh_v = np.nan_to_num(av, nan=1e8) < 24 * 14
ph_h, pdw_h = posting(ts_h, ah); fresh_h = np.nan_to_num(ah, nan=1e8) < 24 * 14
xv = np.where(fresh_v, lr_ph[0, ph_v], 0.0); xh = np.where(fresh_h, lr_ph[0, ph_h], 0.0)
nested_report("V posting-hour prior (<2w listings)", zv, xv, yv, v["off"], v["M"], foldv, slices=sl_v)
nested_report("H posting-hour prior (<2w listings)", zbh, xh, yh, offh, Mh, foldh, slices=sl_h)
xv = np.where(fresh_v, lr_pd[0, pdw_v], 0.0); xh = np.where(fresh_h, lr_pd[0, pdw_h], 0.0)
nested_report("V posting-weekday prior (<2w listings)", zv, xv, yv, v["off"], v["M"], foldv, slices=sl_v)
nested_report("H posting-weekday prior (<2w listings)", zbh, xh, yh, offh, Mh, foldh, slices=sl_h)
# posting hour relative to the feed hour (same-day-posted items): 'posted during this session' effect is listing age itself; skip.
# 'posted on a weekend' flag x category
key_tr = ctr * 2 + (pdw_tr >= 5)
lr_cpw, chi_cpw, _, _, _, _ = smoothed_logratio(np.where(fresh_tr, key_tr, 127), np.zeros(len(key_tr), np.int64), 1, ytr, NC=128, K=300.0)
base_c = np.log(np.maximum(np.bincount(ctr[fresh_tr], weights=ytr[fresh_tr], minlength=64) / np.maximum(np.bincount(ctr[fresh_tr], minlength=64), 1), 1e-6))
tab = lr_cpw[:, 0].reshape(64, 2) - 0  # log rate(cat, posted-weekend) - log overall rate
tab = tab - np.log(ytr.mean()) * 0
tab = (tab - base_c[:, None]) + np.log(ytr.mean())
tab = np.where(np.isfinite(tab), tab, 0.0)
xv = np.where(fresh_v, tab[catv, (pdw_v >= 5).astype(int)], 0.0); xh = np.where(fresh_h, tab[cath, (pdw_h >= 5).astype(int)], 0.0)
nested_report("V cat x posted-on-weekend (<2w listings)", zv, xv, yv, v["off"], v["M"], foldv, slices=sl_v)
nested_report("H cat x posted-on-weekend (<2w listings)", zbh, xh, yh, offh, Mh, foldh, slices=sl_h)

log("== C. weather contrast inside train: cold Sep 6-8 vs warm Sep 9-18 -> val (cold again) ==")
cold_tr = (day_tr <= 2).astype(np.int64)
lr_cw, chi_cw, cnt_c, rate_c, pos_cw, cnt_cw = smoothed_logratio(ctr, cold_tr, 2, ytr)
big = np.argsort(-cnt_c)[:20]
log("random-like: per-category log-ratio on COLD days (Sep 6-8) vs category mean, top-20: " + ", ".join(f"cat {c} {lr_cw[c, 1]:+.3f}" for c in big) + f"; chi2 {chi_cw[big].sum():.1f} on 20 df; categories p<0.001 (10.8): {(chi_cw[big] > 10.8).sum()}")
x_cold = lr_cw[catv, 1] - lr_cw[catv, 0]  # cold-minus-warm deviation, applied to every val row (val is cold)
nested_report("V cold-day category deviation (random-like train) -> val", zv, x_cold, yv, v["off"], v["M"], foldv, slices=sl_v)
# production version (all train rows, sampled 600K feeds) for power
prod_feeds = np.where(~((p > 0.9) & (tr["M"] >= 50)))[0]
prod_feeds = np.sort(np.random.default_rng(0).choice(prod_feeds, 600000, replace=False))
rows_p = rows_of(tr["off"], tr["M"], prod_feeds)
cp = np.asarray(cand("train", "category_id")[rows_p]).astype(np.int64)
yp = np.asarray(cand("train", "target")[rows_p]).astype(np.float64)
fp = np.repeat(prod_feeds, tr["M"][prod_feeds])
day_p = tr["day"][fp]
lr_cwp, chi_cwp, cnt_cp, rate_cp, _, _ = smoothed_logratio(cp, (day_p <= 2).astype(np.int64), 2, yp)
log("production: per-category cold-day log-ratio, top-20: " + ", ".join(f"cat {c} {lr_cwp[c, 1]:+.3f}" for c in big) + f"; chi2 {chi_cwp[big].sum():.1f}; corr with random-like {np.corrcoef(lr_cwp[big, 1], lr_cw[big, 1])[0, 1]:+.3f}")
nested_report("V cold-day category deviation (production train) -> val", zv, lr_cwp[catv, 1] - lr_cwp[catv, 0], yv, v["off"], v["M"], foldv, slices=sl_v)
# the symmetric control: 'warm-peak' days Sep 9-11 vs the rest
lr_ww, chi_ww, _, _, _, _ = smoothed_logratio(cp, ((day_p >= 3) & (day_p <= 5)).astype(np.int64), 2, yp)
nested_report("V ctl: Sep 9-11 category deviation (production) -> val", zv, lr_ww[catv, 1] - lr_ww[catv, 0], yv, v["off"], v["M"], foldv, slices=sl_v)

log("== D. second-week regime (Sep 12-19) vs first week (Sep 6-11): category deviation -> val ==")
wk2 = (day_p >= 6).astype(np.int64)
lr_w2, chi_w2, _, _, _, _ = smoothed_logratio(cp, wk2, 2, yp)
log("production: week-2 minus week-1 per-category log-ratio, top-20: " + ", ".join(f"cat {c} {lr_w2[c, 1] - lr_w2[c, 0]:+.3f}" for c in big) + f"; chi2 {chi_w2[big].sum():.1f} on 20 df")
nested_report("V week-2-minus-week-1 category deviation (production) -> val", zv, lr_w2[catv, 1] - lr_w2[catv, 0], yv, v["off"], v["M"], foldv, slices=sl_v)
wk2r = (day_tr >= 6).astype(np.int64)
lr_w2r, chi_w2r, _, _, _, _ = smoothed_logratio(ctr, wk2r, 2, ytr)
nested_report("V week-2-minus-week-1 category deviation (random-like) -> val", zv, lr_w2r[catv, 1] - lr_w2r[catv, 0], yv, v["off"], v["M"], foldv, slices=sl_v)
# last-3-days (Sep 17-19) deviation, the most recent signal before val
l3 = (day_p >= 11).astype(np.int64)
lr_l3, _, _, _, _, _ = smoothed_logratio(cp, l3, 2, yp)
nested_report("V last-3-days (Sep 17-19) category deviation (production) -> val", zv, lr_l3[catv, 1] - lr_l3[catv, 0], yv, v["off"], v["M"], foldv, slices=sl_v)
# candidate-mix shift: which categories gained exposure share in week 2 and in val/test (is the mix change aligned with a rate change?)
mix1 = np.bincount(cp[day_p < 6], minlength=64) / max((day_p < 6).sum(), 1)
mix2 = np.bincount(cp[day_p >= 6], minlength=64) / max((day_p >= 6).sum(), 1)
mixv = np.bincount(catv, minlength=64) / len(catv)
log("exposure share (pp) week1 / week2 / val for the top-12 categories: " + ", ".join(f"cat {c} {100 * mix1[c]:.1f}/{100 * mix2[c]:.1f}/{100 * mixv[c]:.1f}" for c in big[:12]))
log("done")
