"""a01: daily / weekly seasonality of positives and category x time priors.
Questions: (1) hour / day-of-week composition of val, public, private; (2) val positive rate and G3 NDCG by hour;
(3) is there a category x hour-of-day or category x weekend interaction in the positive rate (fit on random-like train feeds,
which have weekends), and does a smoothed log-ratio prior add anything to G3 (nested additive test on val for hour, on the
30K holdout for weekend; val is weekday-only so the weekend term cannot be validated on val)."""
from common_av import *

v = meta("val")
t = meta("test")
y = np.asarray(cand("val", "target"), dtype=np.float64)
s = g3("val")
fid = np.asarray(cand("val", "fidx"))
catv = np.asarray(cand("val", "category_id")).astype(np.int64)

log("== 1. time composition (MSK) ==")
pub = t["ts"] < PUBLIC_TS
for name, hb, dw in [("val", hour_bucket(v["hour"]), v["dow"]), ("public", hour_bucket(t["hour"][pub]), t["dow"][pub]),
                     ("private", hour_bucket(t["hour"][~pub]), t["dow"][~pub])]:
    hbs = np.bincount(hb, minlength=4) / len(hb)
    dws = np.bincount(dw, minlength=7) / len(dw)
    log(f"{name:8s} n={len(hb):6d} hour buckets[0-6,7-11,12-17,18-23]={np.round(hbs, 3)} weekend share={dws[5:].sum():.3f} dow={np.round(dws, 3)}")

log("== 2. val positive rate and G3 NDCG by hour bucket / day ==")
pf = ndcg_per_feed(s, y, v["off"], v["M"])
npos = np.bincount(fid, weights=y, minlength=v["n"])
hbv = hour_bucket(v["hour"])
for b in range(4):
    m = hbv == b
    log(f"hour bucket {b}: feeds {m.sum():5d} mean M {v['M'][m].mean():6.1f} pos rate {y[np.repeat(m, v['M'])].mean():.4f} npos/feed {npos[m].mean():.3f} G3 NDCG {pf[m].mean():.4f} ± {pf[m].std() / np.sqrt(m.sum()):.4f}")
for d in np.unique(v["dow"]):
    m = v["dow"] == d
    log(f"dow {d}: feeds {m.sum():5d} mean M {v['M'][m].mean():6.1f} G3 NDCG {pf[m].mean():.4f}")

log("== 3. category x time priors from random-like train feeds ==")
tr = meta("train")
hold, p = holdout_index()
sel = np.where((p > 0.9) & (tr["M"] >= 50))[0]
is_hold = np.zeros(tr["n"], bool)
is_hold[hold] = True
fit_feeds = sel[~is_hold[sel]]
log(f"random-like train feeds {len(sel)}, fit feeds (holdout excluded) {len(fit_feeds)}, holdout {len(hold)}")


def rows_of(feeds):
    return np.concatenate([np.arange(o, o + m) for o, m in zip(tr["off"][feeds], tr["M"][feeds])])


rows = rows_of(fit_feeds)
ctr = np.asarray(cand("train", "category_id")[rows]).astype(np.int64)
ytr = np.asarray(cand("train", "target")[rows]).astype(np.float64)
ffe = np.repeat(fit_feeds, tr["M"][fit_feeds])
hb_tr = hour_bucket(tr["hour"])[ffe]
wk_tr = (tr["dow"][ffe] >= 5).astype(np.int64)
log(f"fit rows {len(rows)}, pos rate {ytr.mean():.4f}, weekend row share {wk_tr.mean():.3f}")
NC = 64
K = 300.0


def prior(cat, grp, ngrp, y_):
    pos_c = np.bincount(cat, weights=y_, minlength=NC)
    cnt_c = np.bincount(cat, minlength=NC).astype(np.float64)
    rate_c = pos_c / np.maximum(cnt_c, 1)
    key = cat * ngrp + grp
    pos = np.bincount(key, weights=y_, minlength=NC * ngrp).reshape(NC, ngrp)
    cnt = np.bincount(key, minlength=NC * ngrp).reshape(NC, ngrp).astype(np.float64)
    sm = (pos + K * rate_c[:, None]) / (cnt + K)
    lr = np.log(np.maximum(sm, 1e-6)) - np.log(np.maximum(rate_c, 1e-6))[:, None]
    lr[cnt_c == 0] = 0
    # chi2 of independence per category (observed vs expected under rate_c)
    exp = cnt * rate_c[:, None]
    chi2 = ((pos - exp) ** 2 / np.maximum(exp, 1e-9) + (cnt - pos - (cnt - exp)) ** 2 / np.maximum(cnt - exp, 1e-9)).sum(1)
    return lr, chi2, cnt_c, rate_c, pos, cnt


lr_h, chi_h, cnt_c, rate_c, pos_h, cnt_h = prior(ctr, hb_tr, 4, ytr)
lr_w, chi_w, _, _, pos_w, cnt_w = prior(ctr, wk_tr, 2, ytr)
big = np.argsort(-cnt_c)[:20]
log("top-20 categories by rows: cat, rows, rate, log-ratio by hour bucket (night/morn/day/eve), chi2(3), weekend log-ratio, chi2(1)")
for c in big:
    log(f"cat {c:2d} rows {int(cnt_c[c]):8d} rate {rate_c[c]:.4f} hour {np.round(lr_h[c], 3)} chi2 {chi_h[c]:7.1f} | weekend {np.round(lr_w[c], 3)} chi2 {chi_w[c]:6.1f}")
log(f"total chi2 hour (df ~ 3*ncat): {chi_h.sum():.1f}; weekend (df ~ ncat): {chi_w.sum():.1f}; categories with chi2_h > 16.3 (p<0.001): {(chi_h > 16.3).sum()}, with chi2_w > 10.8: {(chi_w > 10.8).sum()}")

log("== 4. nested additive test on val: z(G3) + w * log-ratio(category, hour bucket) ==")
fold = make_folds(v["groups"], 5, 0)
zb = feed_z(s, fid, v["n"])
xh = lr_h[catv, hbv[fid]]
wgrid = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5])
oof, bnd, ws, curve = nested_additive(zb, xh, y, v["off"], v["M"], fold, wgrid)
log(f"hour x cat prior: OOF delta {fmt(oof, bnd)}; chosen w {ws}; full curve {np.round(curve - bnd.mean(), 5)}")
# control: a category-only prior (log rate_c - mean) which the model already knows
xc = np.log(np.maximum(rate_c, 1e-6))[catv]
oof, bnd, ws, curve = nested_additive(zb, xc, y, v["off"], v["M"], fold, wgrid)
log(f"control, category rate prior: OOF delta {fmt(oof, bnd)}; chosen w {ws}")

log("== 5. weekend x category prior on the 30K holdout (G3 train-only pool) ==")
gh, used, Mh = g3_holdout_pool()
log(f"G3 holdout pool from {len(used)} members")
assert (Mh == tr["M"][hold]).all(), "holdout M mismatch"
yh = np.load(PREDS / "holdout/holdout_y.npy").astype(np.float64)
offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
fidh = np.repeat(np.arange(len(hold)), Mh)
rows_h = rows_of(hold)
cath = np.asarray(cand("train", "category_id")[rows_h]).astype(np.int64)
assert np.allclose(np.asarray(cand("train", "target")[rows_h]).astype(np.float64), yh), "holdout labels mismatch"
wk_h = (tr["dow"][hold] >= 5).astype(np.int64)
hb_h = hour_bucket(tr["hour"][hold])
log(f"holdout weekend feed share {wk_h.mean():.3f}; G3 pool holdout NDCG {ndcg_per_feed(gh, yh, offh, Mh).mean():.5f}")
foldh = make_folds(np.where(tr["user"][hold] == 0, -1 - np.arange(len(hold)), tr["user"][hold]), 5, 0)
zbh = feed_z(gh, fidh, len(hold))
xw = lr_w[cath, wk_h[fidh]]
oof, bnd, ws, curve = nested_additive(zbh, xw, yh, offh, Mh, foldh, wgrid)
m = wk_h[fidh[offh]] == 1
log(f"weekend x cat prior on holdout: OOF delta all {fmt(oof, bnd)}; weekend feeds {fmt(oof[m], bnd[m])}; weekday feeds {fmt(oof[~m], bnd[~m])}; chosen w {ws}")
xh2 = lr_h[cath, hb_h[fidh]]
oof, bnd, ws, curve = nested_additive(zbh, xh2, yh, offh, Mh, foldh, wgrid)
log(f"hour x cat prior on holdout: OOF delta {fmt(oof, bnd)}; chosen w {ws}")
np.savez(OUT / "a01_priors.npz", lr_h=lr_h, lr_w=lr_w, rate_c=rate_c, cnt_c=cnt_c)
log("done")
