"""(4) local geometry: within/between-category cosine, NN category purity, near-duplicate pairs (cos > 0.98) and what they are;
plus cross-split vector drift (train vs val table, same item_id) vs attribute changes.
Full VAL table (2,516,711 items, exact kNN with faiss on GPU N) + the 1.5M train candidate-item sample."""
import os, sys, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
import numpy as np, polars as pl
sys.path.insert(0, '<repo>/nn_x/emb/structure')
from knn_gpu import knn_ip
sys.path.insert(0, "<repo>/feats_v2/life_")
from post_time import age_hours

OUT = "<repo>/nn_x/emb/structure/"
V = "<repo>/data/flat/val/"
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
res = {}
rng = np.random.default_rng(0)

# ---------------- A. train sample: within vs between category cosine ----------------
z = np.load(OUT + "cache/train_sample.npz")
X = z["E"].astype(np.float32) / 256.0; cat = z["cat"].astype(np.int64)
a = rng.integers(0, len(X), 2_000_000); b = rng.integers(0, len(X), 2_000_000); k = a != b; a, b = a[k], b[k]
cs = (X[a] * X[b]).sum(1); same = cat[a] == cat[b]
res["train_pairs"] = dict(random_pair_cos=float(cs.mean()), same_cat_share=float(same.mean()), same_cat_cos_mean=float(cs[same].mean()),
                          diff_cat_cos_mean=float(cs[~same].mean()), same_cat_cos_q10_q50_q90=np.quantile(cs[same], [.1, .5, .9]).tolist(),
                          diff_cat_cos_q10_q50_q90=np.quantile(cs[~same], [.1, .5, .9]).tolist())
# per-category tightness and centroid separation
uc = np.unique(cat)
cent = np.stack([X[cat == c].mean(0) for c in uc]); tight = np.linalg.norm(cent, axis=1) ** 2  # = E[cos] within cat (approx, incl. self)
cn = cent / np.linalg.norm(cent, axis=1, keepdims=True); cc = cn @ cn.T
iu = np.triu_indices(len(uc), 1)
res["train_cat_geometry"] = dict(within_cat_mean_cos_min_med_max=[float(tight.min()), float(np.median(tight)), float(tight.max())],
                                 centroid_cos_min_med_max=[float(cc[iu].min()), float(np.median(cc[iu])), float(cc[iu].max())],
                                 tightest_cats=[(int(uc[i]), float(tight[i])) for i in np.argsort(-tight)[:3]],
                                 loosest_cats=[(int(uc[i]), float(tight[i])) for i in np.argsort(tight)[:3]],
                                 closest_centroid_pairs=[(int(uc[iu[0][j]]), int(uc[iu[1][j]]), float(cc[iu][j])) for j in np.argsort(-cc[iu])[:5]])
log("train pairs", res["train_pairs"], res["train_cat_geometry"])

# ---------------- B. val table: exact kNN over all 2.5M items ----------------
NV = 2_516_711
EV = np.fromfile("<repo>/data/competition/val/item_embeddings/data.memmap", dtype=np.int8).reshape(NV, 128)
XV = EV.astype(np.float32) / 256.0
XV /= np.linalg.norm(XV, axis=1, keepdims=True)
D, I = knn_ip(XV, XV, 11, qchunk=4096, bchunk=1_300_000)
log("val knn done")
# drop self (self may not be at position 0 when exact twins exist)
selfm = I == np.arange(NV)[:, None]
D2 = np.where(selfm, -9, D); I2 = np.where(selfm, -1, I)
o = np.argsort(-D2, 1)[:, :10]
D2 = np.take_along_axis(D2, o, 1); I2 = np.take_along_axis(I2, o, 1)
np.save(OUT + "cache/val_knn10_I.npy", I2.astype(np.int32)); np.save(OUT + "cache/val_knn10_D.npy", D2.astype(np.float16))
# per val emb row attributes
ce = np.load(V + "cand_emb_idx.npy"); he = np.load(V + "hist_emb_idx.npy")
cid = np.load(V + "cand_item_id.npy"); hid = np.load(V + "hist_item_id.npy")
catv = np.zeros(NV, np.int16); catv[he[he >= 0]] = np.load(V + "hist_category_id.npy")[he >= 0]; catv[ce[ce >= 0]] = np.load(V + "cand_category_id.npy")[ce >= 0]
item = np.full(NV, -1, np.int64); item[he[he >= 0]] = hid[he >= 0]; item[ce[ce >= 0]] = cid[ce >= 0]
iscand = np.zeros(NV, bool); iscand[ce[ce >= 0]] = True
feeds = pl.read_parquet(V + "feeds.parquet"); ts = feeds["timestamp"].to_numpy().astype(np.float64); uid = feeds["user_id"].to_numpy()
cfi = np.load(V + "cand_fidx.npy"); hfi = np.load(V + "hist_fidx.npy"); tgt = np.load(V + "cand_target.npy")
knots = np.load("<repo>/data/feats/life_f4_age_knots.npy")
okc = ce >= 0
fv = {}
nc = np.bincount(ce[okc], minlength=NV)
for kf in range(1, 6):
    f = np.load(V + f"cand_f{kf}.npy")
    fv[kf] = np.where(nc > 0, np.bincount(ce[okc], weights=f[okc].astype(np.float64), minlength=NV) / np.maximum(nc, 1), np.nan)
f4 = np.load(V + "cand_f4.npy")
tpost = np.full(NV, np.nan); tpost[ce[okc]] = ts[cfi[okc]] - age_hours(f4[okc], knots) * 3600.0
first_feed_c = np.full(NV, -1, np.int64); first_feed_c[ce[okc][::-1]] = cfi[okc][::-1]
okh = he >= 0
first_feed_h = np.full(NV, -1, np.int64); first_feed_h[he[okh][::-1]] = hfi[okh][::-1]
ckeys = np.unique(cfi[okc].astype(np.int64) * NV + ce[okc]); hkeys = np.unique(hfi[okh].astype(np.int64) * NV + he[okh])
def member(keys, f, e):
    q = f.astype(np.int64) * NV + e; p = np.searchsorted(keys, q); p = np.minimum(p, len(keys) - 1)
    return (keys[p] == q) & (f >= 0)
nn1 = D2[:, 0]
res["val_knn"] = {"n_items": NV, "nn1_cos_q10_q50_q90_q99": np.quantile(nn1, [.1, .5, .9, .99]).tolist(),
                  "share_nn1_cos_gt_0.95": float((nn1 > 0.95).mean()), "share_nn1_cos_gt_0.98": float((nn1 > 0.98).mean()),
                  "share_nn1_cos_gt_0.99": float((nn1 > 0.99).mean()), "share_nn1_cos_gt_0.999": float((nn1 > 0.999).mean())}
# exact twins (bit-identical vectors with different rows)
_, inv, cnt = np.unique(EV, axis=0, return_inverse=True, return_counts=True)
res["val_knn"]["share_with_exact_twin"] = float((cnt[inv.ravel()] > 1).mean()); res["val_knn"]["exact_twin_groups"] = int((cnt > 1).sum()); res["val_knn"]["max_twin_group"] = int(cnt.max())
res["val_knn"]["knn1_same_cat"] = float((catv[I2[:, 0]] == catv).mean()); res["val_knn"]["knn10_same_cat"] = float((catv[I2] == catv[:, None]).mean())
log("val knn", res["val_knn"])


def pair_stats(pa, pb, label):
    r = dict(n_pairs=int(len(pa)))
    r["same_cat"] = float((catv[pa] == catv[pb]).mean())
    r["same_item_id"] = float((item[pa] == item[pb]).mean())
    both = np.isfinite(tpost[pa]) & np.isfinite(tpost[pb])
    dt = np.abs(tpost[pa] - tpost[pb])[both]
    r["n_both_posttime"] = int(both.sum())
    if both.sum():
        r["dt_post_h_q10_q50_q90"] = (np.quantile(dt, [.1, .5, .9]) / 3600).tolist()
        r["same_128s_slot"] = float((dt < 64).mean()); r["dt_lt_1h"] = float((dt < 3600).mean()); r["dt_gt_7d"] = float((dt > 7 * 86400).mean())
    for kf in (1, 2, 3, 5):
        okf = np.isfinite(fv[kf][pa]) & np.isfinite(fv[kf][pb])
        if okf.sum():
            r[f"f{kf}_absdiff_median"] = float(np.median(np.abs(fv[kf][pa] - fv[kf][pb])[okf]))
            r[f"f{kf}_equal_1e-4"] = float((np.abs(fv[kf][pa] - fv[kf][pb])[okf] < 1e-4).mean())
    r["cooccur_same_feed_cands"] = float(member(ckeys, first_feed_c[pa], pb).mean())
    r["cooccur_same_feed_hist"] = float(member(hkeys, first_feed_h[pa], pb).mean())
    r["share_cand_items"] = float((iscand[pa] & iscand[pb]).mean())
    log(label, r)
    return r


ps = {}
for lo, hi, lab in [(0.98, 1.01, "cos>0.98"), (0.9999, 1.01, "cos>0.9999 (near-exact)"), (0.98, 0.9999, "0.98<cos<0.9999"), (0.90, 0.95, "0.90<cos<0.95"), (0.80, 0.85, "0.80<cos<0.85")]:
    m = (D2[:, 0] >= lo) & (D2[:, 0] < hi)
    pa = np.nonzero(m)[0]; pb = I2[m, 0]
    if len(pa) > 400000:
        s = rng.choice(len(pa), 400000, replace=False); pa, pb = pa[s], pb[s]
    ps[lab] = pair_stats(pa, pb, lab)
# random same-category pairs (baseline)
pa = rng.integers(0, NV, 3_000_000); pb = rng.integers(0, NV, 3_000_000)
m = (catv[pa] == catv[pb]) & (pa != pb); ps["random same-cat"] = pair_stats(pa[m][:400000], pb[m][:400000], "random same-cat")
ps["random"] = pair_stats(pa[:400000], pb[:400000], "random")
res["pairs"] = ps
# near-dups by category: share of items with nn1 cos > 0.98, per category (top/bottom)
shc = {}
for c in np.unique(catv):
    mm = catv == c
    if mm.sum() > 5000:
        shc[int(c)] = [int(mm.sum()), float((nn1[mm] > 0.98).mean())]
res["neardup_share_by_cat_top5"] = sorted(shc.items(), key=lambda t: -t[1][1])[:5]
res["neardup_share_by_cat_bottom5"] = sorted(shc.items(), key=lambda t: t[1][1])[:5]

# ---------------- B2. near-duplicate candidates inside the same val feed and their labels ----------------
M = feeds["M"].to_numpy(); off = feeds["cand_off"].to_numpy()
npair = 0; nfeed_with = 0; both_pos = 0; one_pos = 0; pos_given_pos = []
n_rows_in_dup = 0; pos_rows_in_dup = 0
for f in range(len(M)):
    e = ce[off[f]:off[f] + M[f]]; y = tgt[off[f]:off[f] + M[f]]
    okk = e >= 0
    Z = XV[np.where(okk, e, 0)]
    S_ = Z @ Z.T; np.fill_diagonal(S_, 0); S_[~okk] = 0; S_[:, ~okk] = 0
    iu_, ju_ = np.nonzero(np.triu(S_ > 0.98, 1))
    keep = e[iu_] != e[ju_]
    iu_, ju_ = iu_[keep], ju_[keep]
    if len(iu_):
        nfeed_with += 1; npair += len(iu_)
        both_pos += int((y[iu_] * y[ju_]).sum()); one_pos += int(((y[iu_] + y[ju_]) >= 1).sum())
        ind = np.unique(np.r_[iu_, ju_]); n_rows_in_dup += len(ind); pos_rows_in_dup += int(y[ind].sum())
base_rate = float(tgt.mean())
res["feed_neardup"] = dict(feeds_with_pair=nfeed_with, share_feeds=nfeed_with / len(M), n_pairs=npair, rows_in_pairs=n_rows_in_dup,
                           share_rows=n_rows_in_dup / len(ce), pos_rate_rows_in_pairs=pos_rows_in_dup / max(n_rows_in_dup, 1), base_pos_rate=base_rate,
                           P_both_pos_given_one_pos=both_pos / max(one_pos, 1))
log("feed neardup", res["feed_neardup"])

# ---------------- C. cross-split drift: same item_id in train sample and val table ----------------
order = np.argsort(item); sitem = item[order]
tid = z["item_id"].astype(np.int64)
p = np.searchsorted(sitem, tid); p = np.minimum(p, NV - 1); hit = sitem[p] == tid
vi = order[p[hit]]
Xt = X[hit]; Xv = XV[vi]
cosd = (Xt * Xv).sum(1); ident = (z["E"][hit] == EV[vi]).all(1)
res["drift"] = {"n_shared": int(hit.sum()), "bit_identical": float(ident.mean()), "cos_mean": float(cosd.mean()), "cos_q01_q10": np.quantile(cosd, [.01, .1]).tolist(),
                "share_cos_lt_0.99": float((cosd < 0.99).mean()), "share_cos_lt_0.95": float((cosd < 0.95).mean())}
chg = ~ident
# which f's changed between train (item mean) and val (item mean)?
fdel = {}
for kf in (1, 2, 3, 5):
    ft = z[f"f{kf}"][hit].astype(np.float64); fvv = fv[kf][vi]; okf = np.isfinite(fvv)
    dlt = np.abs(ft - fvv)
    fdel[f"f{kf}"] = {"share_absdelta_gt_1e-3_if_vec_identical": float((dlt[okf & ~chg] > 1e-3).mean()),
                      "share_absdelta_gt_1e-3_if_vec_changed": float((dlt[okf & chg] > 1e-3).mean()),
                      "spearman_absdelta_vs_1minus_cos": float(np.corrcoef(np.argsort(np.argsort(dlt[okf])), np.argsort(np.argsort(1 - cosd[okf])))[0, 1])}
res["drift"]["f_change_vs_vector_change"] = fdel
cat_t = z["cat"][hit].astype(np.int64); res["drift"]["category_changed_share_if_vec_changed"] = float((cat_t[chg] != catv[vi][chg]).mean())
res["drift"]["category_changed_share_if_vec_identical"] = float((cat_t[~chg] != catv[vi][~chg]).mean())
# age: is the vector change related to age at the train appearance?
age_t = (z["t_feed"] - z["t_post"])[hit] / 3600
okA = np.isfinite(age_t)
res["drift"]["changed_share_by_train_age_bucket_h"] = {f"{lo}-{hi}": float(chg[okA & (age_t >= lo) & (age_t < hi)].mean())
                                                       for lo, hi in [(0, 24), (24, 168), (168, 720), (720, 4000), (4000, 1e9)]}
res["drift"]["changed_share_age_capped_or_missing"] = float(chg[~okA].mean())
# where does the change live: per-dim |delta| for changed items, normalised
dq = np.abs(z["E"][hit][chg].astype(np.int16) - EV[vi][chg].astype(np.int16)).mean(0)
res["drift"]["per_dim_mean_absdelta_min_med_max"] = [float(dq.min()), float(np.median(dq)), float(dq.max())]
res["drift"]["n_changed_dims_per_changed_item_q10_q50_q90"] = np.quantile((z["E"][hit][chg] != EV[vi][chg]).sum(1), [.1, .5, .9]).tolist()
# is the change moving items toward a neighbour (same category content) or random? cos(train vec, val vec) distribution for changed only
res["drift"]["changed_cos_q10_q50_q90"] = np.quantile(cosd[chg], [.1, .5, .9]).tolist()
log("drift", res["drift"])
json.dump(res, open(OUT + "s04_local_geom.json", "w"), indent=1, default=float)
log("saved")
