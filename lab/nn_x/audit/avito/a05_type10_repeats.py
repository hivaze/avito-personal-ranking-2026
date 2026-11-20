"""a05: type-10 semantics (negative feedback / hide?) and repeat-engagement items (several events on one item).
Candidate-level lifts on val, per-feed AUC, and nested additive tests vs G3 for the strongest ones."""
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
base = y.mean()
NC = 64
E = np.asarray(emb_table("val"), dtype=np.float32) / 256.0
cv = E[np.maximum(cemb, 0)]
cv[cemb < 0] = 0

log("== 1. type 10 ==")
m10 = ht == 10
f10 = np.bincount(hf[m10], minlength=n) > 0
log(f"type-10 events {m10.sum()}, feeds with type 10 {f10.sum()} ({f10.mean():.3f}); mean hpos {hp[m10].mean():.1f}; share hpos<5 {(hp[m10] < 5).mean():.3f}")
key = hf * (hi.max() + 1) + hi
k10 = np.unique(key[m10])
# which other events does the same item have in the same history?
for et in range(1, 11):
    if et == 10:
        continue
    m = ht == et
    log(f"   type-10 items that also have a type-{et} event: {np.isin(k10, np.unique(key[m])).mean():.3f}")
# for type-10 items, the hpos of the type-1 event of the same item vs the type-10 event (which comes first?)
k1 = key[ht == 1]
p1 = hp[ht == 1]
d = {}
for kk, pp in zip(k1, p1):
    d[kk] = min(d.get(kk, 999), pp)
dd = np.array([hp[i] - d.get(key[i], -999) for i in np.where(m10)[0][:20000]])
ok = dd > -900
log(f"   hpos(type10) - hpos(type1 same item): share type10 newer (diff<0) {(dd[ok] < 0).mean():.3f}, equal {(dd[ok] == 0).mean():.3f}, older {(dd[ok] > 0).mean():.3f} (n {ok.sum()})")
# candidate same category as a type-10 item; cos to type-10 items
k10c = np.bincount(hf[m10] * NC + hc[m10], minlength=n * NC)
same10 = k10c[fid * NC + catv] > 0
rows10 = np.repeat(f10, v["M"])
log(f"   in feeds with type 10: base rate {y[rows10].mean():.4f}; same-category-as-type10 lift {y[rows10 & same10].mean() / y[rows10].mean():.2f} (share {same10[rows10].mean():.3f}); other {y[rows10 & ~same10].mean() / y[rows10].mean():.2f}")
# compare: same category as the most recent type-1 event in those feeds
m1 = ht == 1
order = np.lexsort((hp[m1], hf[m1]))
ff = hf[m1][order]
first = np.r_[True, ff[1:] != ff[:-1]]
lastcat = np.full(n, -1)
lastcat[ff[first]] = hc[m1][order][first]
same_last1 = lastcat[fid] == catv
log(f"   same-category-as-last-type1 lift in those feeds {y[rows10 & same_last1].mean() / y[rows10].mean():.2f} (share {same_last1[rows10].mean():.3f})")
# max cos to type-10 items
mm = m10 & (he >= 0)
mx10 = np.full(len(y), np.nan)
idx_by_feed = np.split(np.arange(len(hf))[mm], np.cumsum(np.bincount(hf[mm], minlength=n))[:-1])
for i in np.where(f10)[0]:
    ev = idx_by_feed[i]
    if len(ev) == 0:
        continue
    r = np.arange(v["off"][i], v["off"][i] + v["M"][i])
    mx10[r] = (cv[r] @ E[he[ev]].T).max(1)
okc = ~np.isnan(mx10)
for lo, hi_ in [(-2, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 0.99), (0.99, 2)]:
    mm_ = okc & (mx10 >= lo) & (mx10 < hi_)
    if mm_.sum() > 200:
        log(f"   max cos to type-10 items in [{lo},{hi_}): share {mm_.sum() / okc.sum():.3f} lift vs in-feed base {y[mm_].mean() / y[okc].mean():.2f}")

log("== 2. repeat-engagement items (>= 3 events on the same item, or any contact/msg) ==")
u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
nev = cnt[inv]
deep = (nev >= 3)
cont = np.isin(key, np.unique(key[(ht == 2) | (ht == 3) | (ht == 5) | (ht == 6) | (ht == 8)]))
log(f"events on items with >=3 events: {deep.mean():.3f}; events on contacted/messaged items: {cont.mean():.3f}; feeds with any deep item {(np.bincount(hf[deep], minlength=n) > 0).mean():.3f}")
for nm, msk in [("deep3", deep), ("contacted", cont), ("deep3_recent(hpos<32)", deep & (hp < 32)), ("contacted_recent", cont & (hp < 32))]:
    kc = np.bincount(hf[msk] * NC + hc[msk], minlength=n * NC)
    samec = kc[fid * NC + catv] > 0
    ff_ = np.bincount(hf[msk], minlength=n) > 0
    rows_ = np.repeat(ff_, v["M"])
    kall = np.bincount(hf * NC + hc, minlength=n * NC)
    samec_any = kall[fid * NC + catv] > 0
    # lift of same-category-as-deep vs same-category-as-any-history (given in feeds with deep items)
    log(f"   {nm:22s}: feeds {ff_.mean():.3f}; lift same-cat-as-{nm} {y[rows_ & samec].mean() / y[rows_].mean():.2f} (share {samec[rows_].mean():.3f}) vs same-cat-as-any-history-but-not-{nm} {y[rows_ & samec_any & ~samec].mean() / y[rows_].mean():.2f} (share {(samec_any & ~samec)[rows_].mean():.3f}) vs not-in-history {y[rows_ & ~samec_any].mean() / y[rows_].mean():.2f}")
# max cos to contacted items (unique items) vs max cos to all history: additive test vs G3
mmc = cont & (he >= 0)
mxc = np.full(len(y), np.nan)
idx_by_feed = np.split(np.arange(len(hf))[mmc], np.cumsum(np.bincount(hf[mmc], minlength=n))[:-1])
for i in range(n):
    ev = idx_by_feed[i]
    if len(ev) == 0:
        continue
    ev = ev[:64]
    r = np.arange(v["off"][i], v["off"][i] + v["M"][i])
    mxc[r] = (cv[r] @ E[he[ev]].T).max(1)
okc = ~np.isnan(mxc)
log(f"max cos to contacted items: coverage {okc.mean():.3f}; per-feed AUC {np.nanmean(feed_auc(np.nan_to_num(mxc, nan=-1), y, fid, n)):.4f} vs maxsim32 {np.nanmean(feed_auc(feat('val', 'maxsim32'), y, fid, n)):.4f}")
fold = make_folds(v["groups"], 5, 0)
zb = feed_z(s, fid, n)
wgrid = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0])
for nm, x in [("maxcos_contacted", np.nan_to_num(mxc, nan=np.nanmin(mxc))), ("maxcos_type10", np.nan_to_num(mx10, nan=np.nanmin(mx10)))]:
    xz = feed_z(x, fid, n)
    o1, bnd, w1, _ = nested_additive(zb, xz, y, v["off"], v["M"], fold, wgrid)
    o2, _, w2, _ = nested_additive(zb, -xz, y, v["off"], v["M"], fold, wgrid)
    log(f"additive {nm:16s}: + {fmt(o1, bnd)} w {w1} | - {fmt(o2, bnd)} w {w2}")
log("done")
