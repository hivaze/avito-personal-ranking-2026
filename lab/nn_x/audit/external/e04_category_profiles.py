"""e04: can the anonymised category codes be mapped to real Avito categories?
Per-category profile from val+test (label-free except the val positive rate): exposure share, positive rate, listing-age distribution,
hour-of-day / weekend exposure profile, history event mix (contact 2/3, favourite 4, messaging 5/6/8, search-surface s2 share), item
re-use, f2/f3/f5 means, and the embedding geometry (category centroids, cosine clustering into super-groups).
The question is not only 'can we', but 'would it buy anything' (e02 P6 gives the ceiling of any category-level demand prior)."""
from common_ext import *

v = meta("val"); te = meta("test")
NC = 64
rows = {}
for s, d in [("val", v), ("test", te)]:
    rows[s] = dict(cat=np.asarray(cand(s, "category_id")).astype(np.int64), fidx=np.asarray(cand(s, "fidx")).astype(np.int64),
                   item=np.asarray(cand(s, "item_id")).astype(np.int64), emb=np.asarray(cand(s, "emb_idx")).astype(np.int64),
                   f2=np.asarray(cand(s, "f2"), dtype=np.float64), f3=np.asarray(cand(s, "f3"), dtype=np.float64), f5=np.asarray(cand(s, "f5"), dtype=np.float64),
                   f4=np.asarray(cand(s, "f4"), dtype=np.float64))
yv = np.asarray(cand("val", "target"), dtype=np.float64)
knots = np.load(FEATS / "life_f4_age_knots.npy")


def age_hours(x):
    u = x * 64; k = np.clip(np.floor(u).astype(int), 1, 62); w = u - k
    return np.where(x < 63 / 64 - 1e-6, knots[k] * (1 - w) + knots[k + 1] * w, np.nan)


cat_all = np.concatenate([rows["val"]["cat"], rows["test"]["cat"]])
n_all = len(cat_all)
share = np.bincount(cat_all, minlength=NC) / n_all
rate_v = np.bincount(rows["val"]["cat"], weights=yv, minlength=NC) / np.maximum(np.bincount(rows["val"]["cat"], minlength=NC), 1)
age_all = age_hours(np.concatenate([rows["val"]["f4"], rows["test"]["f4"]]))
hour_all = np.concatenate([v["hour"][rows["val"]["fidx"]], te["hour"][rows["test"]["fidx"]]])
wk_all = np.concatenate([np.zeros(len(rows["val"]["cat"]), bool), (te["dow"] >= 5)[rows["test"]["fidx"]]])
item_all = np.concatenate([rows["val"]["item"], rows["test"]["item"]])
f2 = np.concatenate([rows["val"]["f2"], rows["test"]["f2"]]); f3 = np.concatenate([rows["val"]["f3"], rows["test"]["f3"]]); f5 = np.concatenate([rows["val"]["f5"], rows["test"]["f5"]])
# history event mix per category (val + test histories)
hc = np.concatenate([np.asarray(hist("val", "category_id")).astype(np.int64), np.asarray(hist("test", "category_id")).astype(np.int64)])
het = np.concatenate([np.asarray(hist("val", "event_type")).astype(np.int64), np.asarray(hist("test", "event_type")).astype(np.int64)])
hes = np.concatenate([np.asarray(hist("val", "event_source")).astype(np.int64), np.asarray(hist("test", "event_source")).astype(np.int64)])
hcnt = np.bincount(hc, minlength=NC).astype(np.float64)
def hshare(mask):
    return np.bincount(hc[mask], minlength=NC) / np.maximum(hcnt, 1)
sh_contact = hshare(np.isin(het, [2, 3])); sh_fav = hshare(het == 4); sh_msg = hshare(np.isin(het, [5, 6, 8])); sh_view = hshare(het == 1)
sh_search = np.bincount(hc[(het == 1) & (hes == 2)], minlength=NC) / np.maximum(np.bincount(hc[het == 1], minlength=NC), 1)
hist_share = hcnt / hcnt.sum()
# test-window exposure by hour bucket and weekend
hb = hour_bucket(hour_all)
prof = np.stack([np.bincount(cat_all[hb == b], minlength=NC) / np.maximum(np.bincount(cat_all, minlength=NC), 1) for b in range(4)], 1)
wk_share = np.bincount(cat_all[wk_all], minlength=NC) / np.maximum(np.bincount(cat_all, minlength=NC), 1)
test_rows = len(rows["test"]["cat"]); wk_base = wk_all.sum() / n_all
# items per category and re-use
uniq = np.zeros(NC); reuse = np.zeros(NC)
for c in range(NC):
    m = cat_all == c
    if m.sum() == 0:
        continue
    u = np.unique(item_all[m]).size
    uniq[c] = u; reuse[c] = m.sum() / u
# embedding centroids (val table)
m = json.load(open(ROOT / "data/competition/val/item_embeddings/data.meta"))
E = np.memmap(ROOT / "data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(m["shape"]))
ev = rows["val"]["emb"]; cv = rows["val"]["cat"]
ok = ev >= 0
sel = np.unique(ev[ok])
Es = np.asarray(E[sel], dtype=np.float32)
Es /= np.maximum(np.linalg.norm(Es, axis=1, keepdims=True), 1e-6)
pos_of = {e: i for i, e in enumerate(sel)}
idx = np.array([pos_of[e] for e in ev[ok]])
cent = np.zeros((NC, Es.shape[1]), np.float64)
np.add.at(cent, cv[ok], Es[idx])
cnt_e = np.bincount(cv[ok], minlength=NC)
cent /= np.maximum(cnt_e[:, None], 1)
cn = cent / np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-9)
cos = cn @ cn.T
# within-category tightness: mean cos to centroid
tight = np.zeros(NC)
np.add.at(tight, cv[ok], (Es[idx] * cn[cv[ok]]).sum(1))
tight /= np.maximum(cnt_e, 1)
# age quantiles
def q(a, qs):
    a = a[np.isfinite(a)]
    return np.quantile(a, qs) if len(a) else np.full(len(qs), np.nan)

big = np.argsort(-share)[:24]
log("per-category profile (val+test candidates; sorted by exposure share). Columns: share%, hist share%, val pos rate, age median (d) / share <1d / share >3m, "
    "hour profile night/morn/day/eve (share of the category's rows), weekend share (test rows; base), contact / fav / msg share of history events, search-view share, "
    "rows per unique item, f2/f3/f5 means, embedding tightness")
for c in big:
    mc = cat_all == c
    aq = q(age_all[mc] / 24, [0.5]); sh1 = np.nanmean(age_all[mc] < 24); sh3 = np.nanmean(age_all[mc] > 24 * 90)
    log(f"cat {c:2d} | {100 * share[c]:5.2f} {100 * hist_share[c]:5.2f} | {rate_v[c]:.4f} | age med {aq[0]:6.1f}d <1d {sh1:.3f} >3m {sh3:.3f} | hrs {prof[c, 0]:.2f}/{prof[c, 1]:.2f}/{prof[c, 2]:.2f}/{prof[c, 3]:.2f} | wk {wk_share[c]:.3f} (base {wk_base:.3f}) | "
        f"ev c {sh_contact[c]:.3f} f {sh_fav[c]:.3f} m {sh_msg[c]:.3f} | srch {sh_search[c]:.3f} | reuse {reuse[c]:.2f} | f2 {f2[mc].mean():.3f} f3 {f3[mc].mean():.3f} f5 {f5[mc].mean():.3f} | tight {tight[c]:.3f}")
log("hour profile base: " + "/".join(f"{np.mean(hb == b):.2f}" for b in range(4)) + f"; event mix base: contact {np.isin(het, [2, 3]).mean():.3f} fav {(het == 4).mean():.3f} msg {np.isin(het, [5, 6, 8]).mean():.3f}; search-view share base {((het == 1) & (hes == 2)).sum() / (het == 1).sum():.3f}")
log("== embedding geometry: nearest categories (cosine of centroids) for the top-24 ==")
for c in big:
    o = np.argsort(-cos[c])[1:4]
    log(f"cat {c:2d}: " + ", ".join(f"{j} ({cos[c, j]:.2f})" for j in o))
# agglomerative super-groups at cos > 0.5 among categories with >= 1000 rows
import scipy.cluster.hierarchy as sch
valid = np.where(np.bincount(cat_all, minlength=NC) >= 1000)[0]
D = 1 - cos[np.ix_(valid, valid)]
np.fill_diagonal(D, 0)
Z = sch.linkage(D[np.triu_indices(len(valid), 1)], method="average")
for thr in [0.4, 0.55, 0.7]:
    lab = sch.fcluster(Z, t=thr, criterion="distance")
    groups = {}
    for c, l in zip(valid, lab):
        groups.setdefault(l, []).append(int(c))
    gs = sorted(groups.values(), key=lambda g: -sum(share[g]))
    log(f"super-groups at cosine distance {thr}: {len(gs)} groups; " + "; ".join(f"[{', '.join(map(str, g))}] {100 * sum(share[g]):.1f}%" for g in gs[:12]))
np.savez(OUT / "e04_profile.npz", share=share, hist_share=hist_share, rate_v=rate_v, prof=prof, wk_share=wk_share, sh_contact=sh_contact, sh_fav=sh_fav,
         sh_msg=sh_msg, sh_search=sh_search, reuse=reuse, cos=cos, tight=tight)
log("done")
