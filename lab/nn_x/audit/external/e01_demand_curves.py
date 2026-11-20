"""e01: data-derived category-level demand curves by day / weekday / hour, and event-day deviations.
Questions:
 (1) per-day feed composition of every split (MSK);
 (2) does the per-category POSITIVE RATE vary across the 12 train days beyond binomial noise (random-like feeds, production feeds, all)?
     How large is the variation (exposure-weighted sd of the smoothed log-ratio)?
 (3) does the per-category CANDIDATE EXPOSURE mix move across days (val Tue/Thu, test Sat/Sun/Mon/Tue, public/private)? JS divergences;
 (4) event days inside train: Thu Sep 10 (payday), Fri Sep 11 (CBR hold), Fri Sep 18 / Sat Sep 19 (Duma voting days) and the warm
     'Indian summer' Sep 9-18 vs the cold days Sep 6-8: per-category log-ratio vs the same weekday-type baseline, compared with a
     control pair of ordinary days.
Outputs: out/e01.log (stdout), out/e01_curves.npz (pos/cnt tables)."""
from common_ext import *

tr = meta("train"); v = meta("val"); te = meta("test")
ND = 24  # day index 0 = Sep 6 ... 23 = Sep 29

log("== 1. per-day composition (MSK) ==")
for name, d, mask in [("train", tr, None), ("val", v, None), ("public", te, te["ts"] < PUBLIC_TS), ("private", te, te["ts"] >= PUBLIC_TS)]:
    days = d["day"] if mask is None else d["day"][mask]
    Ms = d["M"] if mask is None else d["M"][mask]
    Hs = d["H"] if mask is None else d["H"][mask]
    for dd in np.unique(days):
        m = days == dd
        if m.sum() < 50:
            continue
        log(f"{name:8s} {day_label(dd)} feeds {m.sum():7d} mean M {Ms[m].mean():6.1f} mean H {Hs[m].mean():6.1f}")

log("== 2. train: category x day positive rate (random-like / production / all) ==")
hold, p = holdout_index()
is_hold = np.zeros(tr["n"], bool); is_hold[hold] = True
randlike = (p > 0.9) & (tr["M"] >= 50) & ~is_hold
prod = ~((p > 0.9) & (tr["M"] >= 50))
grp_feed = np.where(randlike, 1, np.where(is_hold, 2, 0)).astype(np.int8)  # 0 production-like (incl. M<50), 1 random-like fit, 2 holdout
cat = np.asarray(cand("train", "category_id")).astype(np.int64)
y = np.asarray(cand("train", "target")).astype(np.float64)
day_row = np.repeat(tr["day"], tr["M"]).astype(np.int64)
hour_row = np.repeat(tr["hour"], tr["M"]).astype(np.int64)
grp_row = np.repeat(grp_feed, tr["M"])
log(f"train rows {len(cat)}, random-like fit rows {(grp_row == 1).sum()}, holdout rows {(grp_row == 2).sum()}, production rows {(grp_row == 0).sum()}")
tables = {}
for gname, gm in [("randlike", grp_row == 1), ("prod", grp_row == 0), ("all", np.ones(len(cat), bool))]:
    c_, d_, y_ = cat[gm], day_row[gm], y[gm]
    lr, chi2, cnt_c, rate_c, pos, cnt = smoothed_logratio(c_, d_, ND, y_)
    days_present = (cnt > 1000).sum(1)
    df = np.maximum(days_present - 1, 1)
    disp = chi2 / df
    # exposure-weighted sd of the smoothed log-ratio across days (size of the day effect)
    w = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
    sd_lr = np.sqrt((w * lr ** 2).sum(1))
    big = np.argsort(-cnt_c)[:25]
    tot_w = cnt_c[big] / cnt_c[big].sum()
    log(f"[{gname}] rows {len(c_)}, pos rate {y_.mean():.4f}; top-25 categories: exposure-weighted sd(log-ratio across days) {float((tot_w * sd_lr[big]).sum()):.4f}; "
        f"median dispersion chi2/df {np.median(disp[big]):.2f}; categories (of 25) with chi2 p<0.001: {(chi2[big] > 31.3).sum()} (df~11)")
    for c in big[:12]:
        log(f"  cat {c:2d} rows {int(cnt_c[c]):9d} rate {rate_c[c]:.4f} chi2/df {disp[c]:5.2f} sd_lr {sd_lr[c]:.3f} lr by day " +
            " ".join(f"{day_label(dd)[:3]}{dd:02d}:{lr[c, dd]:+.2f}" for dd in range(ND) if cnt[c, dd] > 1000))
    # hour x category
    lr_h, chi_h, _, _, pos_h, cnt_h = smoothed_logratio(c_, np.asarray(hour_row[gm]), 24, y_)
    w_h = cnt_h / np.maximum(cnt_h.sum(1, keepdims=True), 1)
    sd_h = np.sqrt((w_h * lr_h ** 2).sum(1))
    log(f"[{gname}] hour(24) x category: exposure-weighted sd(log-ratio) top-25 {float((tot_w * sd_h[big]).sum()):.4f}; categories with chi2 p<0.001 (df 23, 49.7): {(chi_h[big] > 49.7).sum()}")
    tables[gname] = dict(pos_day=pos, cnt_day=cnt, pos_hour=pos_h, cnt_hour=cnt_h, rate_c=rate_c, cnt_c=cnt_c, lr_day=lr, lr_hour=lr_h)
    # day-level overall positive rate
    pr_day = np.bincount(d_, weights=y_, minlength=ND) / np.maximum(np.bincount(d_, minlength=ND), 1)
    log(f"[{gname}] overall positive rate by day: " + " ".join(f"{day_label(dd)[:3]}{dd:02d}:{pr_day[dd]:.4f}" for dd in range(ND) if np.bincount(d_, minlength=ND)[dd] > 10000))

log("== 2b. excess variance decomposition: how much of the day-to-day rate variation is real? ==")
# For each (cat, day) cell of the random-like table: observed deviation of rate from the category mean vs binomial expectation.
T = tables["randlike"]
pos, cnt, rate_c = T["pos_day"], T["cnt_day"], T["rate_c"]
m = cnt > 2000
obs = (pos / np.maximum(cnt, 1) - rate_c[:, None])[m]
var_bin = (rate_c[:, None] * (1 - rate_c[:, None]) / np.maximum(cnt, 1))[m]
excess = obs ** 2 - var_bin
log(f"random-like cells (cnt>2000): {m.sum()}; mean obs^2 {np.mean(obs ** 2):.3e}; mean binomial var {np.mean(var_bin):.3e}; "
    f"excess var {np.mean(excess):.3e} -> real between-day sd of the rate ~ {np.sqrt(max(np.mean(excess), 0)):.5f} vs mean rate {rate_c[m.any(1)].mean():.4f} "
    f"(relative sd {np.sqrt(max(np.mean(excess), 0)) / rate_c[m.any(1)].mean():.3f})")
T = tables["all"]
pos, cnt, rate_c = T["pos_day"], T["cnt_day"], T["rate_c"]
m = cnt > 20000
obs = (pos / np.maximum(cnt, 1) - rate_c[:, None])[m]
var_bin = (rate_c[:, None] * (1 - rate_c[:, None]) / np.maximum(cnt, 1))[m]
log(f"all-train cells (cnt>20000): {m.sum()}; excess var {np.mean(obs ** 2 - var_bin):.3e} -> real between-day relative sd {np.sqrt(max(np.mean(obs ** 2 - var_bin), 0)) / rate_c[m.any(1)].mean():.3f}")

log("== 3. candidate category mix by day: val / test / public / private ==")
NC = 64


def cat_mix(c_rows):
    h = np.bincount(c_rows, minlength=NC).astype(np.float64)
    return h / h.sum()


def js(pq, qq):
    mm = 0.5 * (pq + qq)
    def kl(a, b):
        mk = a > 0
        return float((a[mk] * np.log(a[mk] / b[mk])).sum())
    return 0.5 * kl(pq, mm) + 0.5 * kl(qq, mm)


catv = np.asarray(cand("val", "category_id")).astype(np.int64)
catt = np.asarray(cand("test", "category_id")).astype(np.int64)
yv = np.asarray(cand("val", "target")).astype(np.float64)
dv = np.repeat(v["day"], v["M"]); dt_ = np.repeat(te["day"], te["M"])
mix = {}
for dd in np.unique(dv):
    m = dv == dd
    if m.sum() > 10000:
        mix[f"val {day_label(dd)}"] = cat_mix(catv[m])
for dd in np.unique(dt_):
    m = dt_ == dd
    if m.sum() > 10000:
        mix[f"test {day_label(dd)}"] = cat_mix(catt[m])
pub_rows = np.repeat(te["ts"] < PUBLIC_TS, te["M"])
mix["public"] = cat_mix(catt[pub_rows]); mix["private"] = cat_mix(catt[~pub_rows]); mix["val"] = cat_mix(catv)
# train days
mix_tr = {}
for dd in range(ND):
    m = day_row == dd
    if m.sum() > 100000:
        mix_tr[f"train {day_label(dd)}"] = cat_mix(cat[m])
keys = list(mix.keys())
log("JS divergence of the candidate category mix (x1e4): rows vs 'val'")
for k in keys:
    log(f"  {k:16s} JS vs val {1e4 * js(mix[k], mix['val']):7.2f}  vs private {1e4 * js(mix[k], mix['private']):7.2f}")
trk = list(mix_tr.keys())
log("train day-to-day JS (x1e4) vs train Mon Sep 7 and vs val: " + "; ".join(f"{k[6:]}: {1e4 * js(mix_tr[k], mix_tr['train Mon 07']):.2f}/{1e4 * js(mix_tr[k], mix['val']):.2f}" for k in trk))
# which categories move most between val and test days (exposure share)
log("largest exposure-share moves (pp) between val and each test day, top-5 categories by |delta|:")
for dd in np.unique(dt_):
    k = f"test {day_label(dd)}"
    if k in mix:
        dlt = (mix[k] - mix["val"]) * 100
        top = np.argsort(-np.abs(dlt))[:5]
        log(f"  {k}: " + ", ".join(f"cat {c} {dlt[c]:+.2f} (val {100 * mix['val'][c]:.1f}%)" for c in top))
# val positive rate by category x day (Tue vs Thu), with binomial SE
lr_v, chi_v, cnt_cv, rate_cv, pos_v, cnt_v = smoothed_logratio(catv, dv, ND, yv)
big = np.argsort(-cnt_cv)[:15]
log("val: per-category positive rate Tue 22 vs Thu 24 (top-15 by rows), and the binomial z of the difference")
for c in big:
    r1, r2 = pos_v[c, 16] / max(cnt_v[c, 16], 1), pos_v[c, 18] / max(cnt_v[c, 18], 1)
    se = np.sqrt(rate_cv[c] * (1 - rate_cv[c]) * (1 / max(cnt_v[c, 16], 1) + 1 / max(cnt_v[c, 18], 1)))
    log(f"  cat {c:2d} rows {int(cnt_cv[c]):7d} rate Tue {r1:.4f} Thu {r2:.4f} z {(r1 - r2) / max(se, 1e-9):+.2f}")
log(f"val Tue-vs-Thu: categories (top-15) with |z| > 3: {sum(1 for c in big if abs((pos_v[c, 16] / max(cnt_v[c, 16], 1) - pos_v[c, 18] / max(cnt_v[c, 18], 1)) / max(np.sqrt(rate_cv[c] * (1 - rate_cv[c]) * (1 / max(cnt_v[c, 16], 1) + 1 / max(cnt_v[c, 18], 1))), 1e-9)) > 3)}; total chi2 {chi_v[big].sum():.1f} on ~15 df")

log("== 4. event days inside train (all train rows, per-category log-ratio vs the category's other same-weekday-type days) ==")
T = tables["all"]
pos, cnt, cnt_c = T["pos_day"], T["cnt_day"], T["cnt_c"]
wkend = np.array([((d + 6) % 7) >= 5 for d in range(ND)])
big = np.argsort(-cnt_c)[:25]


def event_dev(day, label):
    same = [d for d in range(ND) if d != day and wkend[d] == wkend[day] and cnt[:, d].sum() > 1e5]
    base_pos = pos[:, same].sum(1); base_cnt = cnt[:, same].sum(1)
    r_base = base_pos / np.maximum(base_cnt, 1)
    r_day = pos[:, day] / np.maximum(cnt[:, day], 1)
    se = np.sqrt(r_base * (1 - r_base) * (1 / np.maximum(cnt[:, day], 1) + 1 / np.maximum(base_cnt, 1)))
    z = (r_day - r_base) / np.maximum(se, 1e-9)
    lr = np.log(np.maximum(r_day, 1e-6)) - np.log(np.maximum(r_base, 1e-6))
    w = cnt_c[big] / cnt_c[big].sum()
    top = big[np.argsort(-np.abs(z[big]))[:5]]
    log(f"  {label:34s} ({day_label(day)}) rows {int(cnt[:, day].sum()):9d}: top-25 exposure-weighted |lr| {float((w * np.abs(lr[big])).sum()):.4f}, "
        f"categories |z|>4: {(np.abs(z[big]) > 4).sum()}, chi2 {float((z[big] ** 2).sum()):.0f} on 25 df; largest: " +
        ", ".join(f"cat {c} lr {lr[c]:+.3f} z {z[c]:+.1f}" for c in top))


event_dev(4, "payday Thu Sep 10")
event_dev(5, "CBR key-rate hold Fri Sep 11")
event_dev(12, "Duma voting day 1 Fri Sep 18")
event_dev(13, "Duma voting day 2 Sat Sep 19")
event_dev(1, "control: Mon Sep 7")
event_dev(2, "control: Tue Sep 8")
event_dev(11, "control: Thu Sep 17 (last warm day)")
event_dev(8, "control: Mon Sep 14")
event_dev(0, "control: Sun Sep 6 (cold, City Day w/e)")
event_dev(7, "control: Sun Sep 13")
np.savez(OUT / "e01_curves.npz", **{f"{g}_{k}": vv for g, T in tables.items() for k, vv in T.items()},
         val_pos_day=pos_v, val_cnt_day=cnt_v, mix_val=mix["val"], mix_public=mix["public"], mix_private=mix["private"])
log("done")
