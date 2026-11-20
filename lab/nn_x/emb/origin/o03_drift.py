"""O3: what changes an item's vector between the train-period and val-period tables (same item_id).
- shared items val∩train: change rate, cos distribution, change rate by category / item age / train->val gap,
  association with changes of item-level f-features (f2, f5 means per split; f4 is age, always grows).
- drift direction structure: PCA share of unit drift vectors (changed, cos in [0.95, 0.9995]) vs matched-cos
  nearest-neighbour difference directions (val-table exact kNN from o02a) -> low-rank drift = scalar input changes.
Outputs out/o03.json; log to stdout."""
import json, sys, time
import numpy as np, polars as pl

sys.path.insert(0, "<repo>/feats_v2/life_")
T0 = time.time()
R = "<repo>"
OUT = f"{R}/nn_x/emb/origin/out"
NV, NT = 2516711, 81557944
res = {}


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def l(split, n):
    return np.load(f"{R}/data/flat/{split}/{n}.npy", mmap_mode="r")


# ---- val row -> item, category
MAXI = 111_400_000
v_item = np.full(NV, -1, np.int64); v_cat = np.zeros(NV, np.int16)
for kind in ("cand", "hist"):
    e = np.asarray(l("val", f"{kind}_emb_idx")); it = np.asarray(l("val", f"{kind}_item_id")); c = np.asarray(l("val", f"{kind}_category_id"))
    m = e >= 0
    v_item[e[m]] = it[m]; v_cat[e[m]] = c[m]
assert (v_item >= 0).all()
row_of_item = np.full(MAXI, -1, np.int32)
row_of_item[v_item] = np.arange(NV, dtype=np.int32)
# ---- train row for each val item; train cand f-means, last train cand time
tr_row = np.full(NV, -1, np.int64)
fsum = {f: np.zeros(NV) for f in ("f2", "f5")}; fcnt = np.zeros(NV)
tr_last_t = np.full(NV, -1, np.int64)
ftr = pl.read_parquet(f"{R}/data/flat/train/feeds.parquet")["timestamp"].to_numpy().astype(np.int64)
CH = 50_000_000
for kind in ("cand", "hist"):
    E = l("train", f"{kind}_emb_idx"); IT = l("train", f"{kind}_item_id")
    for s in range(0, len(E), CH):
        e = np.asarray(E[s:s + CH]); it = np.asarray(IT[s:s + CH])
        vr = row_of_item[it]
        m = (vr >= 0) & (e >= 0)
        tr_row[vr[m]] = e[m]
        if kind == "cand":
            idx = np.nonzero(vr >= 0)[0]
            for f in ("f2", "f5"):
                fsum[f] += np.bincount(vr[idx], weights=np.asarray(l("train", f"cand_{f}")[s:s + CH])[idx], minlength=NV)
            fcnt += np.bincount(vr[idx], minlength=NV)
            t = ftr[np.asarray(l("train", "cand_fidx")[s:s + CH])[idx]]
            np.maximum.at(tr_last_t, vr[idx], t)
    log(kind, "train mapped")
shared = tr_row >= 0
log("shared val items", int(shared.sum()), float(shared.mean()))
# ---- val cand f-means, posting time, val first time
vf = {f: np.bincount(np.asarray(l("val", "cand_emb_idx")).clip(0), weights=np.asarray(l("val", f"cand_{f}")), minlength=NV) for f in ("f2", "f5")}
vcnt = np.bincount(np.asarray(l("val", "cand_emb_idx")).clip(0), minlength=NV)
fv = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet")["timestamp"].to_numpy().astype(np.int64)
ce = np.asarray(l("val", "cand_emb_idx")); ct = fv[np.asarray(l("val", "cand_fidx"))]
v_first_t = np.full(NV, np.iinfo(np.int64).max); np.minimum.at(v_first_t, ce[ce >= 0], ct[ce >= 0])
from post_time import age_hours
knots = np.load(f"{R}/data/feats/life_f4_age_knots.npy")
age = age_hours(np.asarray(l("val", "cand_f4")), knots)
tpost = np.full(NV, np.nan); ok = (ce >= 0) & np.isfinite(age)
tpost[ce[ok]] = (ct[ok] - age[ok] * 3600)
# ---- vectors
mv = np.memmap(f"{R}/data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NV, 128))
mt = np.memmap(f"{R}/data/competition/train/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NT, 128))
S = np.nonzero(shared)[0]
xv = np.asarray(mv[S]).astype(np.float32)
o = np.argsort(tr_row[S]); xt = np.empty_like(xv); xt[o] = np.asarray(mt[tr_row[S][o]]).astype(np.float32)
ident = (xv == xt).all(1)
cos = (xv * xt).sum(1) / np.linalg.norm(xv, axis=1) / np.linalg.norm(xt, axis=1)
res["shared"] = {"n_val_items": NV, "n_shared": int(len(S)), "share_identical": float(ident.mean()),
                 "cos_q_changed": [float(v) for v in np.quantile(cos[~ident], [0, .01, .1, .25, .5, .75, .9])],
                 "share_changed_cos_gt_0.999": float((cos[~ident] > 0.999).mean()),
                 "share_changed_cos_lt_0.95": float((cos[~ident] < 0.95).mean())}
log(json.dumps(res["shared"]))
# change rate by category
cat = v_cat[S]
bc = {}
for c in np.unique(cat):
    m = cat == c
    if m.sum() >= 2000:
        bc[int(c)] = [int(m.sum()), round(float(1 - ident[m].mean()), 4)]
rates = np.array([v[1] for v in bc.values()])
res["change_rate_by_category"] = {"min": float(rates.min()), "max": float(rates.max()), "median": float(np.median(rates)),
                                  "per_cat": bc}
log("change by cat min/med/max", rates.min(), np.median(rates), rates.max())
# change rate vs gap (val first time - train last time) and item age at val time
gap_d = (v_first_t[S] - tr_last_t[S]) / 86400.0
agev_d = (v_first_t[S] - tpost[S]) / 86400.0
okg = (tr_last_t[S] > 0) & (v_first_t[S] < np.iinfo(np.int64).max)
qs = np.quantile(gap_d[okg], np.linspace(0, 1, 6))
res["change_vs_gap_days"] = [[float(qs[i]), float(qs[i + 1]), float(1 - ident[okg][(gap_d[okg] >= qs[i]) & (gap_d[okg] <= qs[i + 1])].mean())] for i in range(5)]
oka = np.isfinite(agev_d)
qa = np.quantile(agev_d[oka], np.linspace(0, 1, 6))
res["change_vs_age_days_at_val"] = [[float(qa[i]), float(qa[i + 1]), float(1 - ident[oka][(agev_d[oka] >= qa[i]) & (agev_d[oka] <= qa[i + 1])].mean())] for i in range(5)]
log("gap", res["change_vs_gap_days"]); log("age", res["change_vs_age_days_at_val"])
# association with change in item-level f2 / f5 (items that are candidates in both periods)
both = (fcnt[S] > 0) & (vcnt[S] > 0)
for f in ("f2", "f5"):
    dtr = fsum[f][S][both] / fcnt[S][both]; dvl = vf[f][S][both] / vcnt[S][both]
    d = np.abs(dvl - dtr)
    qd = np.quantile(d, [0, .5, .8, .95, 1])
    res[f"change_vs_abs_d{f}"] = {"n": int(both.sum()), "bins": [[float(qd[i]), float(qd[i + 1]), float(1 - ident[both][(d >= qd[i]) & (d <= qd[i + 1])].mean())] for i in range(4)],
                                  "spearman_d_vs_1mcos": float(pl.DataFrame({"a": d, "b": 1 - cos[both]}).select(pl.corr("a", "b", method="spearman")).item())}
    log(f, res[f"change_vs_abs_d{f}"])
# category change between tables for the same item (train cat via hist/cand of train is costly; use big drops instead)
# ---- drift direction structure
sel = (~ident) & (cos > 0.95) & (cos < 0.9995)
dlt = xv[sel] - xt[sel]
dlt /= np.linalg.norm(dlt, axis=1, keepdims=True)
csel = cat[sel]


def pca_share(z):
    ev = np.linalg.eigvalsh((z.T @ z) / len(z))[::-1]
    return {"top1": float(ev[0] / ev.sum()), "top5": float(ev[:5].sum() / ev.sum()), "top10": float(ev[:10].sum() / ev.sum()),
            "pr_rank": float(ev.sum() ** 2 / (ev ** 2).sum())}


res["drift_pca_all"] = {"n": int(sel.sum()), **pca_share(dlt)}
# null: NN differences with matched cos in [0.95, 0.9995]
I = np.load(f"{OUT}/knn_val_idx.npy"); D = np.load(f"{OUT}/knn_val_dot.npy")
nv = np.linalg.norm(np.asarray(mv).astype(np.float32), axis=1)
c1 = D[:, 0] / nv / nv[I[:, 0]]
rng = np.random.default_rng(0)
# match the cos distribution of the drift sample with histogram weights
bins = np.linspace(0.95, 0.9995, 31)
hd, _ = np.histogram(cos[sel], bins); hn, _ = np.histogram(c1, bins)
cand_rows = np.nonzero((c1 > 0.95) & (c1 < 0.9995))[0]
b_of = np.clip(np.digitize(c1[cand_rows], bins) - 1, 0, 29)
w = (hd / np.maximum(hn, 1))[b_of]
pick = rng.choice(cand_rows, size=min(len(cand_rows), int(sel.sum())), replace=False, p=w / w.sum())
xa = np.asarray(mv[np.sort(pick)]).astype(np.float32)
pk = np.sort(pick)
xb = np.asarray(mv[I[pk, 0]]).astype(np.float32)
nd = xa - xb; nd /= np.linalg.norm(nd, axis=1, keepdims=True)
res["nn_diff_pca_all"] = {"n": int(len(pk)), "same_cat_share": float((v_cat[pk] == v_cat[I[pk, 0]]).mean()), **pca_share(nd)}
log("drift", res["drift_pca_all"]); log("nn null", res["nn_diff_pca_all"])
# within-category comparison (largest categories)
wc = {}
for c in [int(k) for k, v in sorted(bc.items(), key=lambda kv: -kv[1][0])[:10]]:
    a = dlt[csel == c]; b = nd[v_cat[pk] == c]
    n = min(len(a), len(b))
    if n >= 500:
        wc[c] = {"n": n, "drift_top5": pca_share(a[:n])["top5"], "nn_top5": pca_share(b[:n])["top5"],
                 "drift_top1": pca_share(a[:n])["top1"], "nn_top1": pca_share(b[:n])["top1"]}
res["within_cat_pca"] = wc
log("within-cat", wc)
# alignment of drift with the global mean direction and the mean user direction
mu_item = np.asarray(mv[rng.choice(NV, 200000, replace=False)]).astype(np.float64).mean(0); mu_item /= np.linalg.norm(mu_item)
ue = np.load(f"{R}/data/flat/val/user_emb.npy").astype(np.float64); ue = ue[np.linalg.norm(ue, axis=1) > 0]
mu_user = ue.mean(0); mu_user /= np.linalg.norm(mu_user)
res["drift_align"] = {"mean_abs_cos_drift_item_mean": float(np.abs(dlt @ mu_item).mean()),
                      "mean_abs_cos_nn_item_mean": float(np.abs(nd @ mu_item).mean()),
                      "mean_abs_cos_drift_user_mean": float(np.abs(dlt @ mu_user).mean()),
                      "mean_abs_cos_nn_user_mean": float(np.abs(nd @ mu_user).mean()),
                      "mean_signed_cos_drift_user_mean": float((dlt @ mu_user).mean()),
                      "random_unit_expected_abs": float(np.sqrt(2 / np.pi / 128))}
log("align", res["drift_align"])
# sign consistency: is the drift systematically toward something? mean drift vector norm
res["drift_mean_vector_norm"] = float(np.linalg.norm(dlt.mean(0))); res["nn_mean_vector_norm"] = float(np.linalg.norm(nd.mean(0)))
log("mean-vector norms", res["drift_mean_vector_norm"], res["nn_mean_vector_norm"])
json.dump(res, open(f"{OUT}/o03.json", "w"), indent=1)
log("done")
