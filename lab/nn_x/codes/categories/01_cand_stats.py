"""01: candidate-side per-category statistics.
Shares of candidates / positives, CTR (val; train exploration-like feeds p_random>0.5 & M>=50; production-like p_random<0.05),
distinct items and popularity concentration (full train candidate rows), f1..f5 distributions (val+test), f4 -> listing age,
f2 'dominant level' share, feed presence. Output: out/cand_stats.parquet + log."""
from common_cat import *

res = {}
codes = np.arange(NC)

# ---------------- val / test
VT = {}
for s in ["val", "test"]:
    m = meta(s)
    d = dict(cat=np.asarray(cand(s, "category_id")).astype(np.int64), item=np.asarray(cand(s, "item_id")).astype(np.int64),
             fidx=np.asarray(cand(s, "fidx")).astype(np.int64), emb=np.asarray(cand(s, "emb_idx")).astype(np.int64))
    for k in range(1, 6):
        d[f"f{k}"] = np.asarray(cand(s, f"f{k}"), dtype=np.float64)
    if s == "val":
        d["y"] = np.asarray(cand(s, "target"), dtype=np.float64)
    d["meta"] = m
    VT[s] = d
    log(s, "rows", len(d["cat"]))

v = VT["val"]; te = VT["test"]
n_v = np.bincount(v["cat"], minlength=NC).astype(float)
n_t = np.bincount(te["cat"], minlength=NC).astype(float)
p_v = np.bincount(v["cat"], weights=v["y"], minlength=NC)
res["share_val"] = n_v / n_v.sum()
res["share_test"] = n_t / n_t.sum()
res["n_val"] = n_v; res["pos_val"] = p_v
res["pos_share_val"] = p_v / p_v.sum()
res["ctr_val"] = p_v / np.maximum(n_v, 1)
res["ctr_val_se"] = np.sqrt(res["ctr_val"] * (1 - res["ctr_val"]) / np.maximum(n_v, 1))
log("val base ctr", p_v.sum() / n_v.sum())

# feed presence, mean count when present, within-feed rank of category size
for s, d in [("val", v), ("test", te)]:
    key = d["fidx"] * NC + d["cat"]
    u, cnt = np.unique(key, return_counts=True)
    fc = u % NC
    res[f"feed_presence_{s}"] = np.bincount(fc, minlength=NC) / d["meta"]["n"]
    res[f"cnt_when_present_{s}"] = np.bincount(fc, weights=cnt, minlength=NC) / np.maximum(np.bincount(fc, minlength=NC), 1)
    if s == "val":
        ncat = np.bincount(u // NC, minlength=d["meta"]["n"])
        log("val categories per feed: mean", ncat.mean(), "median", np.median(ncat))
        # positive category structure: among (feed,cat) groups, share with >=1 positive and mean positives given >=1
        pk = np.bincount(np.searchsorted(u, key), weights=d["y"], minlength=len(u))
        res["grp_hit_rate_val"] = np.bincount(fc, weights=(pk > 0), minlength=NC) / np.maximum(np.bincount(fc, minlength=NC), 1)
        res["pos_per_hit_grp_val"] = np.bincount(fc, weights=pk, minlength=NC) / np.maximum(np.bincount(fc, weights=(pk > 0), minlength=NC), 1)

# f1..f5 distributions on val+test
cat_vt = np.concatenate([v["cat"], te["cat"]])
for k in range(1, 6):
    x = np.concatenate([v[f"f{k}"], te[f"f{k}"]])
    df = pl.DataFrame({"c": cat_vt, "x": x}).group_by("c").agg(
        pl.col("x").mean().alias("mean"), pl.col("x").quantile(0.1).alias("q10"), pl.col("x").median().alias("q50"),
        pl.col("x").quantile(0.9).alias("q90"), pl.col("x").std().alias("sd")).sort("c")
    for col in ["mean", "q10", "q50", "q90", "sd"]:
        a = np.full(NC, np.nan); a[df["c"].to_numpy()] = df[col].to_numpy(); res[f"f{k}_{col}"] = a
# f4 -> age
age = age_hours(np.concatenate([v["f4"], te["f4"]]))
df = pl.DataFrame({"c": cat_vt, "a": age}).group_by("c").agg(
    pl.col("a").median().alias("age_med_h"), (pl.col("a") < 24).mean().alias("age_lt1d"), (pl.col("a") > 24 * 90).mean().alias("age_gt90d"),
    pl.col("a").is_nan().mean().alias("age_capped")).sort("c")
for col in ["age_med_h", "age_lt1d", "age_gt90d", "age_capped"]:
    a = np.full(NC, np.nan); a[df["c"].to_numpy()] = df[col].to_numpy(); res[col] = a
# f2: dominant level per feed (mode), share of candidates at the mode, share below / above
fid_vt = np.concatenate([v["fidx"], te["fidx"] + 10_000_000])
f2 = np.concatenate([v["f2"], te["f2"]])
dfm = pl.DataFrame({"f": fid_vt, "x": f2}).group_by(["f", "x"]).len().sort(["f", "len"], descending=[False, True]).group_by("f").first()
mode = dict(zip(dfm["f"].to_list(), dfm["x"].to_list()))
fu, finv = np.unique(fid_vt, return_inverse=True)
mode_arr = np.array([mode[f] for f in fu])[finv]
at_mode = np.isclose(f2, mode_arr)
res["f2_at_mode"] = np.bincount(cat_vt, weights=at_mode, minlength=NC) / np.maximum(np.bincount(cat_vt, minlength=NC), 1)
res["f2_above_mode"] = np.bincount(cat_vt, weights=(f2 > mode_arr + 1e-9), minlength=NC) / np.maximum(np.bincount(cat_vt, minlength=NC), 1)
res["f2_below_mode"] = np.bincount(cat_vt, weights=(f2 < mode_arr - 1e-9), minlength=NC) / np.maximum(np.bincount(cat_vt, minlength=NC), 1)
nlev = pl.DataFrame({"f": fid_vt, "x": f2}).group_by("f").agg(pl.col("x").n_unique())["x"].to_numpy()
log("f2 distinct levels per feed: mean", nlev.mean(), "quantiles", np.quantile(nlev, [.1, .5, .9]))
# f3 constant within (feed, category)?
f3 = np.concatenate([v["f3"], te["f3"]])
nu = pl.DataFrame({"f": fid_vt, "c": cat_vt, "x": f3}).group_by(["f", "c"]).agg(pl.col("x").n_unique())["x"].to_numpy()
log("f3 n_unique per (feed,cat): share==1", (nu == 1).mean())
# item reuse in val+test
item_vt = np.concatenate([v["item"], te["item"]])
dfi = pl.DataFrame({"c": cat_vt, "i": item_vt}).group_by("c").agg(pl.col("i").n_unique().alias("u"), pl.len().alias("n")).sort("c")
a = np.zeros(NC); a[dfi["c"].to_numpy()] = dfi["u"].to_numpy(); res["items_vt"] = a
res["rows_per_item_vt"] = np.where(a > 0, np.bincount(cat_vt, minlength=NC) / np.maximum(a, 1), np.nan)
emb_vt = np.concatenate([v["emb"], te["emb"]])
res["no_emb_vt"] = np.bincount(cat_vt, weights=(emb_vt < 0), minlength=NC) / np.maximum(np.bincount(cat_vt, minlength=NC), 1)
del VT, v, te

# ---------------- train (full rows; exploration / production split by feed p_random)
tm = meta("train")
pr = p_random()
cat = np.asarray(cand("train", "category_id")).astype(np.int64)
y = np.asarray(cand("train", "target")).astype(np.float64)
fidx = np.asarray(cand("train", "fidx")).astype(np.int64)
is_R = (pr > 0.5) & (tm["M"] >= 50)
is_P = (pr < 0.05)
log("train feeds: exploration-like", is_R.sum(), "production-like", is_P.sum(), "of", tm["n"])
rR = is_R[fidx]; rP = is_P[fidx]
for nm, msk in [("all", None), ("R", rR), ("P", rP)]:
    c = cat if msk is None else cat[msk]
    yy = y if msk is None else y[msk]
    n = np.bincount(c, minlength=NC).astype(float); p = np.bincount(c, weights=yy, minlength=NC)
    res[f"share_tr{nm}"] = n / n.sum(); res[f"ctr_tr{nm}"] = p / np.maximum(n, 1); res[f"pos_share_tr{nm}"] = p / p.sum()
    res[f"n_tr{nm}"] = n; res[f"pos_tr{nm}"] = p
    log(nm, "rows", int(n.sum()), "ctr", p.sum() / n.sum())
# production position: mean original position by category (production feeds), normalised by M
pos = np.asarray(cand("train", "pos")).astype(np.float64)
Mrow = tm["M"][fidx].astype(np.float64)
rel = pos / np.maximum(Mrow, 1)
res["prod_relpos"] = np.bincount(cat[rP], weights=rel[rP], minlength=NC) / np.maximum(np.bincount(cat[rP], minlength=NC), 1)
del pos, rel, Mrow

# items: distinct items and popularity concentration over all train candidate rows
item = np.asarray(cand("train", "item_id")).astype(np.int64)
cnt_item = np.bincount(item)
cat_of = np.zeros(len(cnt_item), np.int16); cat_of[item] = cat
del item
has = cnt_item > 0
ci = cat_of[has].astype(np.int64); ni = cnt_item[has].astype(np.float64)
res["items_train"] = np.bincount(ci, minlength=NC).astype(float)
res["rows_per_item_train"] = res["n_trall"] / np.maximum(res["items_train"], 1)
top1 = np.full(NC, np.nan); gini = np.full(NC, np.nan); once = np.full(NC, np.nan); top10 = np.full(NC, np.nan); med = np.full(NC, np.nan)
order = np.argsort(ci, kind="stable")
cs = ci[order]; ns = ni[order]
bnd = np.searchsorted(cs, np.arange(NC + 1))
for c in range(NC):
    a = np.sort(ns[bnd[c]:bnd[c + 1]])[::-1]
    if len(a) < 10:
        continue
    k = max(1, len(a) // 100)
    top1[c] = a[:k].sum() / a.sum(); top10[c] = a[:max(1, len(a) // 10)].sum() / a.sum()
    once[c] = (a == 1).mean(); med[c] = np.median(a)
    asc = a[::-1]; n = len(asc)
    gini[c] = (2 * np.sum(np.arange(1, n + 1) * asc) / (n * asc.sum())) - (n + 1) / n
res["pop_top1pct_share"] = top1; res["pop_top10pct_share"] = top10; res["pop_gini"] = gini; res["pop_once"] = once; res["pop_median_rows"] = med

# hour / weekday CTR profiles on train (exploration-like and all), positives' category mix by hour bucket and weekend
hr = tm["hour"][fidx]; dw = tm["dow"][fidx]
hb = np.digitize(hr, [7, 12, 18])  # 0: 0-6, 1: 7-11, 2: 12-17, 3: 18-23 MSK
wk = dw >= 5
for nm, msk in [("all", None), ("R", rR)]:
    c = cat if msk is None else cat[msk]; yy = y if msk is None else y[msk]
    h = hb if msk is None else hb[msk]; w = wk if msk is None else wk[msk]
    base_ctr = yy.sum() / len(yy)
    for b in range(4):
        mb = h == b
        nb = np.bincount(c[mb], minlength=NC); pb = np.bincount(c[mb], weights=yy[mb], minlength=NC)
        # relative CTR of category in bucket vs category overall, normalised by the bucket's overall CTR change
        res[f"ctr_rel_hb{b}_{nm}"] = (pb / np.maximum(nb, 1)) / np.maximum(res[f"ctr_tr{nm}"], 1e-9) / ((pb.sum() / nb.sum()) / base_ctr)
        res[f"exp_share_hb{b}_{nm}"] = nb / np.maximum(np.bincount(c, minlength=NC), 1)
        res[f"pos_share_hb{b}_{nm}"] = pb / np.maximum(np.bincount(c, weights=yy, minlength=NC), 1e-9)
    nb = np.bincount(c[w], minlength=NC); pb = np.bincount(c[w], weights=yy[w], minlength=NC)
    res[f"ctr_rel_wkend_{nm}"] = (pb / np.maximum(nb, 1)) / np.maximum(res[f"ctr_tr{nm}"], 1e-9) / ((pb.sum() / nb.sum()) / base_ctr)
    res[f"pos_share_wkend_{nm}"] = pb / np.maximum(np.bincount(c, weights=yy, minlength=NC), 1e-9)
    log(nm, "weekend share of positives (base)", (yy * w).sum() / yy.sum(), "weekend share of rows", w.mean())
# 24-hour positive profile (all train) per category, saved for plots
H24 = np.zeros((NC, 24)); N24 = np.zeros((NC, 24))
np.add.at(H24, (cat, hr), y); np.add.at(N24, (cat, hr), 1)
np.savez(OUT / "hour24.npz", pos=H24, n=N24)
D7 = np.zeros((NC, 7)); N7 = np.zeros((NC, 7))
np.add.at(D7, (cat, dw), y); np.add.at(N7, (cat, dw), 1)
np.savez(OUT / "dow7.npz", pos=D7, n=N7)

df = pl.DataFrame({"code": codes, **{k: np.asarray(v_, dtype=np.float64) for k, v_ in res.items()}})
df.write_parquet(OUT / "cand_stats.parquet")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
show = df.filter(pl.col("n_val") + pl.col("n_trall") > 0).sort("share_val", descending=True)
print(show.select(["code", "share_val", "share_test", "share_trall", "share_trR", "share_trP", "ctr_val", "ctr_trR", "ctr_trP", "pos_share_val",
                   "feed_presence_val", "items_train", "rows_per_item_train"]))
print(show.select(["code", "pop_top1pct_share", "pop_gini", "pop_once", "pop_median_rows", "age_med_h", "age_lt1d", "age_gt90d", "f2_at_mode",
                   "f2_above_mode", "f3_mean", "f5_mean", "f1_mean", "no_emb_vt", "prod_relpos"]))
print(show.select(["code", "ctr_rel_hb0_R", "ctr_rel_hb1_R", "ctr_rel_hb2_R", "ctr_rel_hb3_R", "ctr_rel_wkend_R", "ctr_rel_wkend_all",
                   "pos_share_wkend_all", "pos_share_hb0_all", "pos_share_hb3_all"]))
log("done")
