"""a02: event_source semantics (the UI surface of a type-1 / 4 / 7 event) and source-specific affinity features.
(1) What distinguishes type-1 events by source: position in history, contact follow-up rate, category spread, repeat share.
(2) Candidate-level: does a same-category history view from source 1 (recs feed?) vs source 2 (search?) vs other sources
predict the feed click differently? Lifts, per-feed AUC, and a nested LightGBM residual test on top of G3.
(3) Source-specific embedding centroids (last-20 type-1 events per source) -> cosine to candidate."""
from common_av import *

v = meta("val")
n = v["n"]
y = np.asarray(cand("val", "target"), dtype=np.float64)
s = g3("val")
fid = np.asarray(cand("val", "fidx")).astype(np.int64)
catv = np.asarray(cand("val", "category_id")).astype(np.int64)
cemb = np.asarray(cand("val", "emb_idx")).astype(np.int64)
hf = np.asarray(hist("val", "fidx")).astype(np.int64)
ht = np.asarray(hist("val", "event_type")).astype(np.int64)
hs = np.asarray(hist("val", "event_source")).astype(np.int64)
hc = np.asarray(hist("val", "category_id")).astype(np.int64)
hp = np.asarray(hist("val", "hpos")).astype(np.int64)
hi = np.asarray(hist("val", "item_id")).astype(np.int64)
he = np.asarray(hist("val", "emb_idx")).astype(np.int64)
log(f"val rows {len(y)}, hist events {len(hf)}")

# source buckets: 0 -> 's0', 1 -> 's1', 2 -> 's2', else 's3+'
sb = np.where(hs == 1, 1, np.where(hs == 2, 2, np.where(hs == 0, 0, 3)))
SB = ["s0", "s1", "s2", "s3+"]

log("== 1. type-1 / 4 / 7 events by source ==")
key_fi = hf * (hi.max() + 1) + hi
contact_keys = np.unique(key_fi[(ht == 2) | (ht == 3)])
msg_keys = np.unique(key_fi[(ht == 5) | (ht == 6) | (ht == 8)])
fav_keys = np.unique(key_fi[ht == 4])
# repeat: number of events per (feed,item)
u, inv, cntk = np.unique(key_fi, return_inverse=True, return_counts=True)
rep = cntk[inv] > 1
for et in [1, 4, 7]:
    m0 = ht == et
    log(f"-- type {et}: {m0.sum()} events, source distribution {np.round(np.bincount(sb[m0], minlength=4) / m0.sum(), 3)} (s0,s1,s2,s3+); raw sources top: {np.argsort(-np.bincount(hs[m0], minlength=31))[:6]}")
    for b in range(4):
        m = m0 & (sb == b)
        if m.sum() == 0:
            continue
        log(f"   {SB[b]:3s} n {m.sum():7d} mean hpos {hp[m].mean():6.1f} share hpos<20 {(hp[m] < 20).mean():.3f} contact-same-item {np.isin(key_fi[m], contact_keys).mean():.4f} msg-same-item {np.isin(key_fi[m], msg_keys).mean():.4f} fav-same-item {np.isin(key_fi[m], fav_keys).mean():.4f} repeat {rep[m].mean():.3f} emb-missing {(he[m] < 0).mean():.3f}")

# per-feed category spread for type-1 by source: distinct categories / events
log("-- per-feed distinct categories among type-1 events, by source (feeds with >= 5 such events)")
for b in [1, 2, 3]:
    m = (ht == 1) & (sb == b)
    k = hf[m] * 64 + hc[m]
    ncat = np.bincount(np.unique(k) // 64, minlength=n)
    nev = np.bincount(hf[m], minlength=n)
    ok = nev >= 5
    log(f"   {SB[b]:3s} feeds>=5 ev {ok.sum():5d}: mean distinct cats {ncat[ok].mean():.2f}, mean events {nev[ok].mean():.1f}, cats/event {(ncat[ok] / nev[ok]).mean():.3f}")

log("== 2. candidate-level same-category counts of type-1 events by source ==")
NC = 64
feats = {}
for b in range(4):
    m = (ht == 1) & (sb == b)
    k = hf[m] * NC + hc[m]
    cnt_all = np.bincount(k, minlength=n * NC)
    cnt_20 = np.bincount(k[hp[m] < 20], minlength=n * NC)
    rec = np.bincount(k, weights=0.95 ** hp[m], minlength=n * NC)
    kk = fid * NC + catv
    feats[f"c1_{SB[b]}_all"] = cnt_all[kk].astype(np.float64)
    feats[f"c1_{SB[b]}_20"] = cnt_20[kk].astype(np.float64)
    feats[f"c1_{SB[b]}_rec"] = rec[kk]
# most recent type-1 event's category by source: match flag
for b in [1, 2]:
    m = (ht == 1) & (sb == b)
    order = np.lexsort((hp[m], hf[m]))
    ff = hf[m][order]
    first = np.r_[True, ff[1:] != ff[:-1]]
    lastcat = np.full(n, -1)
    lastcat[ff[first]] = hc[m][order][first]
    feats[f"match_last1_{SB[b]}"] = (lastcat[fid] == catv).astype(np.float64)
base_rate = y.mean()
log(f"base positive rate {base_rate:.4f}")
for k_, x in feats.items():
    nz = x > 0
    if nz.sum() == 0:
        continue
    auc = feed_auc(x + 1e-6 * feat("val", "f1"), y, fid, n)
    log(f"{k_:18s} nonzero share {nz.mean():.3f} lift P(y|x>0)/base {y[nz].mean() / base_rate:.2f} lift(x==0) {y[~nz].mean() / base_rate:.2f} per-feed AUC {np.nanmean(auc):.4f} corr with c1_s1 {np.corrcoef(x, feats['c1_s1_all'])[0, 1]:.3f}")

# conditional: given same-category count from s1 == 0, does s2 count help, and vice versa
a = feats["c1_s1_all"] > 0
b2 = feats["c1_s2_all"] > 0
log(f"lift table: s1=0,s2=0 {y[~a & ~b2].mean() / base_rate:.2f} | s1>0,s2=0 {y[a & ~b2].mean() / base_rate:.2f} | s1=0,s2>0 {y[~a & b2].mean() / base_rate:.2f} | both {y[a & b2].mean() / base_rate:.2f}  (shares {(~a & ~b2).mean():.3f} {(a & ~b2).mean():.3f} {(~a & b2).mean():.3f} {(a & b2).mean():.3f})")

log("== 3. source-specific embedding centroids (last 20 type-1 events per source) ==")
E = emb_table("val")
Ef = np.asarray(E, dtype=np.float32) / 256.0  # unit vectors
log(f"emb table {Ef.shape}")
cv = Ef[np.maximum(cemb, 0)]
cv[cemb < 0] = 0
for b in [1, 2, 3]:
    m = (ht == 1) & (sb == b) & (he >= 0) & (hp < 20)
    cen = np.zeros((n, 128), np.float32)
    np.add.at(cen, hf[m], Ef[he[m]])
    cnt = np.bincount(hf[m], minlength=n).astype(np.float32)
    nrm = np.linalg.norm(cen, axis=1)
    cen /= np.maximum(nrm, 1e-6)[:, None]
    cos = (cv * cen[fid]).sum(1)
    cos[cnt[fid] == 0] = np.nan
    feats[f"cos1_{SB[b]}_20"] = cos
    ok = ~np.isnan(cos)
    auc = feed_auc(np.nan_to_num(cos, nan=-1), y, fid, n)
    log(f"cos1_{SB[b]}_20: coverage {ok.mean():.3f}; per-feed AUC {np.nanmean(auc):.4f}; pos rate by cos bin [<0.6,0.6-0.8,0.8-0.9,0.9-0.95,>0.95]: " +
        str([round(float(y[ok & (cos >= lo) & (cos < hi)].mean() / base_rate), 2) for lo, hi in [(-2, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 2)]]))
# reference: all type-1 last-20 centroid regardless of source
m = (ht == 1) & (he >= 0) & (hp < 20)
cen = np.zeros((n, 128), np.float32)
np.add.at(cen, hf[m], Ef[he[m]])
cen /= np.maximum(np.linalg.norm(cen, axis=1), 1e-6)[:, None]
cos_all = (cv * cen[fid]).sum(1)
auc = feed_auc(cos_all, y, fid, n)
log(f"reference cos to last-20 type-1 centroid (any source): per-feed AUC {np.nanmean(auc):.4f}")

log("== 4. nested tests on top of G3 ==")
fold = make_folds(v["groups"], 5, 0)
zb = feed_z(s, fid, n)
wgrid = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0])
for k_ in ["c1_s1_20", "c1_s2_20", "c1_s3+_20", "c1_s2_rec", "match_last1_s2", "cos1_s2_20", "cos1_s1_20"]:
    x = feats[k_]
    x = np.nan_to_num(x, nan=np.nanmin(x) if np.isnan(x).any() else 0)
    xz = feed_z(np.log1p(x) if x.max() > 2 else x, fid, n)
    oof, bnd, ws, curve = nested_additive(zb, xz, y, v["off"], v["M"], fold, wgrid)
    oof2, _, ws2, _ = nested_additive(zb, -xz, y, v["off"], v["M"], fold, wgrid)
    log(f"additive {k_:16s}: + {fmt(oof, bnd)} w {ws} | - {fmt(oof2, bnd)} w {ws2}")

# LightGBM residual: base = [zG3, logM, logH, cat_cnt_20, cos_h10, category] vs + source features
logM = np.log(v["M"])[fid]
logH = np.log1p(v["H"])[fid]
Xb = np.column_stack([zb, logM, logH, np.log1p(feat("val", "cat_cnt_20")), feat("val", "cos_h10"), np.log1p(feat("val", "cat_cnt_et1")), feat("val", "cos_het1"), catv]).astype(np.float32)
src_cols = ["c1_s1_20", "c1_s2_20", "c1_s3+_20", "c1_s0_20", "c1_s1_rec", "c1_s2_rec", "c1_s3+_rec", "match_last1_s1", "match_last1_s2", "cos1_s1_20", "cos1_s2_20", "cos1_s3+_20"]
Xs = np.column_stack([np.nan_to_num(np.log1p(feats[c]) if feats[c].max() > 2 else feats[c], nan=-1) for c in src_cols]).astype(np.float32)
Xa = np.hstack([Xb, Xs])
res = {}
for seed in [0, 1]:
    ob = lgb_oof(Xb, y, v["off"], v["M"], fold, seed=seed, cat_idx=[7])
    oa = lgb_oof(Xa, y, v["off"], v["M"], fold, seed=seed, cat_idx=[7])
    nb_ = ndcg_per_feed(ob, y, v["off"], v["M"])
    na_ = ndcg_per_feed(oa, y, v["off"], v["M"])
    res[seed] = (nb_, na_)
    log(f"seed {seed}: G3 {ndcg_per_feed(s, y, v['off'], v['M']).mean():.5f} base-stacker {nb_.mean():.5f} +source {na_.mean():.5f} delta {fmt(na_, nb_)}")
nb_ = (res[0][0] + res[1][0]) / 2
na_ = (res[0][1] + res[1][1]) / 2
log(f"2-seed mean: +source vs base-stacker {fmt(na_, nb_)}; base-stacker vs raw G3 {fmt(nb_, ndcg_per_feed(s, y, v['off'], v['M']))}")
np.savez(OUT / "a02_feats.npz", **{k_: feats[k_] for k_ in feats})
log("done")
