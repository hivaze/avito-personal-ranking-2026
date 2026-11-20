"""EDA: do calendar / time-of-day effects exist in the label-candidate relation?

1. position-debiased category lift by day -> weekend/weekday ratio per category with day-cluster CIs
2. category x hour-bucket lift (weekday), split-half (odd/even day) reliability
3. history "demand" (newest 5 history events) category x context lift, reliability, correlation with label lift
4. per-day / per-hour single-feature NDCG lift of f1 and maxsim32 (all feeds and random-like feeds)
Outputs: out/e01_*.json/npz, log to stdout.
"""
from calcommon import *

t0 = time.time()
F = feeds("train")
M = F["M"].to_numpy().astype(np.int64); off = F["cand_off"].to_numpy().astype(np.int64)
tp = time_parts(F["timestamp"].to_numpy())
days = np.unique(tp["day"])
cnt_day = np.bincount(tp["day"] - days.min())
good_days = days[cnt_day[days - days.min()] > 1000]
dayidx_f = np.searchsorted(good_days, tp["day"])
valid_f = np.isin(tp["day"], good_days)
we_days = [d for d in good_days if (d + 3) % 7 >= 5]
log("good days", [str(np.datetime64(int(d), 'D')) for d in good_days], "weekend", [str(np.datetime64(int(d), 'D')) for d in we_days])

pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(len(M))).all()
prand = pr["p_random"]

fidx = cand("train", "fidx", mmap=False)
y = cand("train", "target", mmap=False)
pos = cand("train", "pos", mmap=False)
cat = cand("train", "category_id", mmap=False).astype(np.int64)
log("loaded", time.time() - t0)
r, re = within_feed_rank(pos, off, M)
cell = theta_index(r, re)
del pos
randmask_f = (prand > 0.9) & (M >= 50)
rm = randmask_f[fidx]
theta = np.bincount(cell[rm], weights=y[rm], minlength=56) / np.maximum(np.bincount(cell[rm], minlength=56), 1)
nb_cells = np.bincount(cell[rm], minlength=56)
# fill sparse cells with all-feed estimate scaled
theta_all = np.bincount(cell, weights=y, minlength=56) / np.maximum(np.bincount(cell, minlength=56), 1)
theta = np.where(nb_cells >= 2000, theta, theta_all * np.median(theta[nb_cells >= 2000] / np.maximum(theta_all[nb_cells >= 2000], 1e-9)))
np.save(f"{OUT}/theta_cells.npy", theta)
log("theta (rank bucket x end flag)", np.round(theta.reshape(14, 4), 4).tolist())
e_pos = expected_pos(theta[cell].astype(np.float64), y.astype(np.float64), off, M)
e_flat = expected_pos(np.ones(len(y)), y.astype(np.float64), off, M)
del cell, r, re
log("expected computed", time.time() - t0)

res = {}
ND = len(good_days)
dfi = dayidx_f[fidx]; vf = valid_f[fidx]


def agg(key, n, w):
    return np.bincount(key, weights=w, minlength=n)


# ---------------------------------------------------------------- 1. weekend/weekday ratio per category
key = np.where(vf, cat * ND + dfi, N_CAT * ND)
Y_cd = agg(key, N_CAT * ND + 1, y.astype(np.float64))[:-1].reshape(N_CAT, ND)
E_cd = agg(key, N_CAT * ND + 1, e_pos)[:-1].reshape(N_CAT, ND)
Ef_cd = agg(key, N_CAT * ND + 1, e_flat)[:-1].reshape(N_CAT, ND)
N_cd = np.bincount(key, minlength=N_CAT * ND + 1)[:-1].reshape(N_CAT, ND)
is_we = np.array([(d + 3) % 7 >= 5 for d in good_days])
rows = []
for c in range(N_CAT):
    out = {"cat": c, "rows": int(N_cd[c].sum()), "pos": int(Y_cd[c].sum())}
    for nm, E in [("pos", E_cd), ("flat", Ef_cd)]:
        Rd = (Y_cd[c] + 0.5) / (E[c] + 0.5)
        lr = np.log(Rd)
        we, wd = lr[is_we], lr[~is_we]
        d = we.mean() - wd.mean()
        se = np.sqrt(we.var(ddof=1) / len(we) + wd.var(ddof=1) / len(wd))
        pooled = np.log((Y_cd[c][is_we].sum() / E[c][is_we].sum()) / (Y_cd[c][~is_we].sum() / E[c][~is_we].sum()))
        out[f"{nm}_logratio_daymean"] = float(d); out[f"{nm}_se_day"] = float(se); out[f"{nm}_logratio_pooled"] = float(pooled)
        out[f"{nm}_se_poisson"] = float(np.sqrt(1 / max(Y_cd[c][is_we].sum(), 1) + 1 / max(Y_cd[c][~is_we].sum(), 1)))
    out["supply_share_we"] = float(N_cd[c][is_we].sum() / N_cd[:, is_we].sum())
    out["supply_share_wd"] = float(N_cd[c][~is_we].sum() / N_cd[:, ~is_we].sum())
    rows.append(out)
df = pl.DataFrame(rows).sort("pos", descending=True)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
print(df.select("cat", "rows", "pos", "supply_share_wd", "supply_share_we", "pos_logratio_pooled", "pos_logratio_daymean", "pos_se_day", "pos_se_poisson", "flat_logratio_pooled"))
big = df.filter(pl.col("pos") > 5000)
z = (big["pos_logratio_daymean"] / big["pos_se_day"]).to_numpy()
log("categories with >5000 pos:", len(big), "|z|>2:", int((np.abs(z) > 2).sum()), "|z|>3:", int((np.abs(z) > 3).sum()),
    "pos-weighted mean |logratio|:", float((big["pos"] * big["pos_logratio_daymean"].abs()).sum() / big["pos"].sum()))
df.write_csv(f"{OUT}/e01_cat_weekend_ratio.csv")
# null: day-to-day noise between weekdays only (random split of weekdays into 2 groups)
rng = np.random.default_rng(0)
wd_idx = np.where(~is_we)[0]
null_abs = []
for rep in range(50):
    p = rng.permutation(wd_idx); a, b = p[:4], p[4:]
    lr = np.log((Y_cd + 0.5) / (E_cd + 0.5))
    d = lr[:, a].mean(1) - lr[:, b].mean(1)
    w = Y_cd.sum(1)
    null_abs.append(float((w * np.abs(d)).sum() / w.sum()))
lr = np.log((Y_cd + 0.5) / (E_cd + 0.5))
dwe = lr[:, is_we].mean(1) - lr[:, ~is_we].mean(1)
w = Y_cd.sum(1)
res["weekend_mean_abs_logratio_posweighted"] = float((w * np.abs(dwe)).sum() / w.sum())
res["null_weekday_split_mean_abs_logratio"] = [float(np.mean(null_abs)), float(np.percentile(null_abs, 95))]
log("weekend vs weekday pos-weighted mean |log lift| =", res["weekend_mean_abs_logratio_posweighted"],
    "null (weekday 4 vs 4 split) mean/95% =", res["null_weekday_split_mean_abs_logratio"])

# ---------------------------------------------------------------- 2. category x hour bucket (by daytype), split-half reliability
hb_f = tp["hb"]; we_f = tp["weekend"].astype(np.int64)
half_f = (dayidx_f % 2)  # odd / even good-day index
ctx_f = we_f * N_HB + hb_f
NC = 2 * N_HB
key = np.where(vf, (cat * NC + ctx_f[fidx]) * 2 + half_f[fidx], N_CAT * NC * 2)
Y3 = agg(key, N_CAT * NC * 2 + 1, y.astype(np.float64))[:-1].reshape(N_CAT, NC, 2)
E3 = agg(key, N_CAT * NC * 2 + 1, e_pos)[:-1].reshape(N_CAT, NC, 2)
np.savez(f"{OUT}/e01_cat_ctx.npz", Y=Y3, E=E3)


def lift(Y, E, k=50):
    # log lift of (cat,ctx) relative to cat overall, shrunk with k pseudo-positives toward 0
    Rc = Y.sum(1, keepdims=True) / np.maximum(E.sum(1, keepdims=True), 1e-9)
    return np.log((Y + k) / (E * Rc + k))


for dt, sl in [("weekday", slice(0, N_HB)), ("weekend", slice(N_HB, 2 * N_HB))]:
    L0 = lift(Y3[:, sl, 0], E3[:, sl, 0]); L1 = lift(Y3[:, sl, 1], E3[:, sl, 1])
    w = (Y3[:, sl, 0] + Y3[:, sl, 1])
    m = w > 200
    c01 = np.corrcoef(L0[m], L1[m])[0, 1]
    wsd = np.sqrt((w[m] * ((L0[m] + L1[m]) / 2) ** 2).sum() / w[m].sum())
    log(f"{dt}: cat x hour-bucket log-lift split-half corr={c01:.3f}  pos-weighted rms lift={wsd:.4f}  cells={m.sum()}")
    res[f"cat_hb_{dt}_splithalf_corr"] = float(c01); res[f"cat_hb_{dt}_rms"] = float(wsd)
# weekend vs weekday hour profile agreement
Lwd = lift(Y3[:, :N_HB].sum(2), E3[:, :N_HB].sum(2)); Lwe = lift(Y3[:, N_HB:].sum(2), E3[:, N_HB:].sum(2))
w = Y3.sum(2); m = (w[:, :N_HB] > 300) & (w[:, N_HB:] > 300)
res["cat_hb_weekday_vs_weekend_corr"] = float(np.corrcoef(Lwd[m], Lwe[m])[0, 1])
log("cat x hb lift corr weekday vs weekend:", res["cat_hb_weekday_vs_weekend_corr"])

# ---------------------------------------------------------------- 3. history demand (newest 5 events)
hp = hist("train", "hpos", mmap=False)
hm = hp < 5
del hp
hf = hist("train", "fidx", mmap=False)[hm]
hc = hist("train", "category_id", mmap=False)[hm].astype(np.int64)
log("hist newest-5 rows", len(hf), time.time() - t0)
key = np.where(valid_f[hf], (hc * NC + ctx_f[hf]) * 2 + half_f[hf], N_CAT * NC * 2)
D3 = np.bincount(key, minlength=N_CAT * NC * 2 + 1)[:-1].reshape(N_CAT, NC, 2).astype(np.float64)
np.save(f"{OUT}/e01_hist_cat_ctx.npy", D3)


def dlift(D, k=20):
    # share of cat within ctx / share of cat overall
    sc = D / D.sum(0, keepdims=True)
    s = D.sum(1, keepdims=True) / D.sum()
    return np.log((sc * D.sum(0, keepdims=True) + k * s) / (s * D.sum(0, keepdims=True) + k * s))


for dt, sl in [("weekday", slice(0, N_HB)), ("weekend", slice(N_HB, 2 * N_HB))]:
    A0 = dlift(D3[:, sl, 0]); A1 = dlift(D3[:, sl, 1]); w = D3[:, sl].sum(2); m = w > 2000
    log(f"hist demand {dt}: split-half corr={np.corrcoef(A0[m], A1[m])[0, 1]:.3f}  rms={np.sqrt((w[m] * ((A0[m] + A1[m]) / 2) ** 2).sum() / w[m].sum()):.4f}")
Dall = dlift(D3.sum(2))
# demand lift weekend vs weekday by category (pooled over hours)
Dwe = D3[:, N_HB:].sum((1, 2)); Dwd = D3[:, :N_HB].sum((1, 2))
dem_ratio = np.log((Dwe / Dwe.sum()) / (Dwd / Dwd.sum()))
lab_ratio = dwe
m = Y_cd.sum(1) > 3000
res["weekend_ratio_corr_label_vs_histdemand"] = float(np.corrcoef(dem_ratio[m], lab_ratio[m])[0, 1])
# label lift vs history demand lift at cat x ctx level
Lctx = lift(Y3.sum(2), E3.sum(2))
w = Y3.sum(2); m = w > 300
res["ctx_lift_corr_label_vs_histdemand"] = float(np.corrcoef(Lctx[m], Dall[m])[0, 1])
log("corr(label weekend ratio, hist-demand weekend ratio) over cats:", res["weekend_ratio_corr_label_vs_histdemand"],
    " corr(label ctx lift, demand ctx lift) over cat x ctx:", res["ctx_lift_corr_label_vs_histdemand"])
del hf, hc, hm

# ---------------------------------------------------------------- 4. feature-label relation by day / hour bucket
f1 = cand("train", "f1", mmap=False).astype(np.float64)
ms = np.load(f"{FEATS}/train/maxsim32.npy").astype(np.float64)
const = np.zeros(len(y))
yd = y.astype(np.float64)
nd_c = ndcg_pf(const, yd, off, M, mode=2)
nd_f1 = ndcg_pf(f1, yd, off, M, mode=2)
nd_ms = ndcg_pf(np.nan_to_num(ms, nan=-9), yd, off, M, mode=2)
del f1, ms
tab = []
grp_sets = {"all": np.ones(len(M), bool), "random_like": (prand > 0.5) & (M >= 20)}
for gname, gm in grp_sets.items():
    for d in good_days:
        sel = gm & (tp["day"] == d)
        tab.append(dict(group=gname, key=str(np.datetime64(int(d), 'D')), weekend=int((d + 3) % 7 >= 5), n=int(sel.sum()),
                        lift_f1=float((nd_f1[sel] - nd_c[sel]).mean()), se_f1=float((nd_f1[sel] - nd_c[sel]).std() / np.sqrt(sel.sum())),
                        lift_ms=float((nd_ms[sel] - nd_c[sel]).mean()), se_ms=float((nd_ms[sel] - nd_c[sel]).std() / np.sqrt(sel.sum()))))
    for wk in [0, 1]:
        for h in range(N_HB):
            sel = gm & valid_f & (tp["weekend"] == wk) & (tp["hb"] == h)
            tab.append(dict(group=gname, key=f"we{wk}_hb{h}", weekend=wk, n=int(sel.sum()),
                            lift_f1=float((nd_f1[sel] - nd_c[sel]).mean()), se_f1=float((nd_f1[sel] - nd_c[sel]).std() / np.sqrt(sel.sum())),
                            lift_ms=float((nd_ms[sel] - nd_c[sel]).mean()), se_ms=float((nd_ms[sel] - nd_c[sel]).std() / np.sqrt(sel.sum()))))
    for wk in [0, 1]:
        sel = gm & valid_f & (tp["weekend"] == wk)
        tab.append(dict(group=gname, key=f"weekend{wk}", weekend=wk, n=int(sel.sum()),
                        lift_f1=float((nd_f1[sel] - nd_c[sel]).mean()), se_f1=float((nd_f1[sel] - nd_c[sel]).std() / np.sqrt(sel.sum())),
                        lift_ms=float((nd_ms[sel] - nd_c[sel]).mean()), se_ms=float((nd_ms[sel] - nd_c[sel]).std() / np.sqrt(sel.sum()))))
tdf = pl.DataFrame(tab)
print(tdf)
tdf.write_csv(f"{OUT}/e01_feature_lift_by_time.csv")
json.dump(res, open(f"{OUT}/e01_summary.json", "w"), indent=1)
log("done", time.time() - t0)
