"""a04: candidate-set structure. Are category quotas / caps visible? M parity (2-column grid)? Is the candidate category
mix a function of the history category mix (blender-like sampling)? Where do positives sit relative to the category share?
Compared across val, test, train random-like (holdout 30K) and train production-like (30K, p<0.05, M>=50)."""
from common_av import *

tr = meta("train")
hold, p = holdout_index()
prod = np.where((p < 0.05) & (tr["M"] >= 50))[0]
prod = np.sort(np.random.default_rng(1).choice(prod, 30000, replace=False))


def rows_of(d, feeds):
    return np.concatenate([np.arange(o, o + m) for o, m in zip(d["off"][feeds], d["M"][feeds])])


def hrows_of(d, feeds):
    return np.concatenate([np.arange(o, o + h) for o, h in zip(d["hoff"][feeds], d["H"][feeds])])


NC = 64


def analyse(name, split, feeds=None, with_labels=True):
    d = meta(split)
    if feeds is None:
        feeds = np.arange(d["n"])
    rows = rows_of(d, feeds)
    hr = hrows_of(d, feeds)
    nf = len(feeds)
    M = d["M"][feeds]
    H = d["H"][feeds]
    fl = np.repeat(np.arange(nf), M)
    cat = np.asarray(cand(split, "category_id")[rows]).astype(np.int64)
    hcat = np.asarray(hist(split, "category_id")[hr]).astype(np.int64)
    hfl = np.repeat(np.arange(nf), H)
    hpos = np.asarray(hist(split, "hpos")[hr]).astype(np.int64)
    log(f"==== {name}: {nf} feeds, {len(rows)} rows, mean M {M.mean():.1f} ====")
    # M parity and spikes
    vals, cnts = np.unique(M, return_counts=True)
    top = np.argsort(-cnts)[:8]
    log(f"M even share {(M % 2 == 0).mean():.3f}; top M values {list(zip(vals[top].tolist(), cnts[top].tolist()))}; M==300 share {(M == 300).mean():.4f}; M>=250 {(M >= 250).mean():.3f}")
    # per (feed, cat) counts
    k = fl * NC + cat
    cnt = np.bincount(k, minlength=nf * NC).reshape(nf, NC)
    ncat = (cnt > 0).sum(1)
    top_share = cnt.max(1) / M
    log(f"distinct categories per feed: mean {ncat.mean():.2f}, by M bin [<20,20-50,50-100,100-200,200+]: " + str([round(float(ncat[(M >= lo) & (M < hi)].mean()), 2) for lo, hi in [(0, 20), (20, 50), (50, 100), (100, 200), (200, 301)]]))
    log(f"top-category share: mean {top_share.mean():.3f}; q10/50/90 {np.round(np.quantile(top_share, [0.1, 0.5, 0.9]), 3)}; feeds with one category {(ncat == 1).mean():.3f}")
    # cap check: max count per category relative to M; distribution of max count for M>=100
    big = M >= 100
    mx = cnt.max(1)
    log(f"M>=100 feeds: max count per category q50/90/99/max {np.quantile(mx[big], [0.5, 0.9, 0.99]).round(1)} / {mx[big].max()}; share of feeds where max count == M {(mx == M).mean():.3f}")
    vals, cnts = np.unique(mx[big], return_counts=True)
    top = np.argsort(-cnts)[:6]
    log(f"   most common max-count values (M>=100): {list(zip(vals[top].tolist(), cnts[top].tolist()))}")
    # category share relation to history category share (feeds with H>=20)
    hk = hfl * NC + hcat
    hcnt = np.bincount(hk, minlength=nf * NC).reshape(nf, NC).astype(np.float64)
    hk20 = hfl[hpos < 20] * NC + hcat[hpos < 20]
    hcnt20 = np.bincount(hk20, minlength=nf * NC).reshape(nf, NC).astype(np.float64)
    okh = H >= 20
    cs = cnt / M[:, None]
    hs_ = hcnt / np.maximum(H, 1)[:, None]
    # share of candidates whose category occurs in history / in last-20
    in_hist = (hcnt[fl, cat] > 0)
    in_h20 = (hcnt20[fl, cat] > 0)
    log(f"share of candidates whose category is in history {in_hist[np.repeat(okh, M)].mean():.3f}, in last-20 {in_h20[np.repeat(okh, M)].mean():.3f} (feeds H>=20); feeds whose candidates cover > 3 categories not in history: {((cnt > 0) & (hcnt == 0)).sum(1)[okh].__gt__(3).mean():.3f}")
    # within-feed correlation between candidate share and history share across categories present in either
    corr = []
    for i in np.where(okh)[0][:5000]:
        a, b = cs[i], hs_[i]
        mm = (a > 0) | (b > 0)
        if mm.sum() >= 3 and a[mm].std() > 0 and b[mm].std() > 0:
            corr.append(np.corrcoef(a[mm], b[mm])[0, 1])
    log(f"within-feed corr(cand cat share, hist cat share) over present cats: mean {np.mean(corr):.3f}, median {np.median(corr):.3f} (n {len(corr)})")
    if with_labels:
        y = np.asarray(cand(split, "target")[rows]).astype(np.float64)
        base = y.mean()
        share = cs[fl, cat]
        bins = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.01]
        bi = np.digitize(share, bins) - 1
        log("pos-rate lift by candidate's category share in feed " + str([(f"{bins[i]:.2f}-{bins[i + 1]:.2f}", round(float(y[bi == i].mean() / base), 2), round(float((bi == i).mean()), 3)) for i in range(len(bins) - 1) if (bi == i).sum() > 0]))
        single = cnt[fl, cat] == 1
        log(f"singleton-category candidates: share {single.mean():.3f}, lift {y[single].mean() / base:.2f}; in-hist category lift {y[in_hist].mean() / base:.2f} vs not {y[~in_hist].mean() / base:.2f}; last-20 {y[in_h20].mean() / base:.2f} vs not {y[~in_h20].mean() / base:.2f}")
        # positives: share of positives that are the only candidate of their category
        log(f"positives that are singleton-category {single[y > 0].mean():.3f}; positives in a category absent from history {(~in_hist)[y > 0].mean():.3f}")
    return dict(ncat=ncat, top_share=top_share, M=M)


r_val = analyse("val", "val")
r_test = analyse("test", "test", with_labels=False)
r_rl = analyse("train random-like holdout", "train", hold)
r_pr = analyse("train production-like", "train", prod)

log("== nested additive tests on val vs G3: category-share features ==")
v = meta("val")
n = v["n"]
y = np.asarray(cand("val", "target"), dtype=np.float64)
s = g3("val")
fid = np.asarray(cand("val", "fidx")).astype(np.int64)
cat = np.asarray(cand("val", "category_id")).astype(np.int64)
k = fid * NC + cat
cnt = np.bincount(k, minlength=n * NC)
share = cnt[k] / v["M"][fid]
fold = make_folds(v["groups"], 5, 0)
zb = feed_z(s, fid, n)
wgrid = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0])
for nm, x in [("log share", np.log(share)), ("singleton", (cnt[k] == 1).astype(np.float64)), ("log count", np.log(cnt[k]))]:
    xz = feed_z(x, fid, n)
    o1, bnd, w1, _ = nested_additive(zb, xz, y, v["off"], v["M"], fold, wgrid)
    o2, _, w2, _ = nested_additive(zb, -xz, y, v["off"], v["M"], fold, wgrid)
    log(f"{nm:10s}: + {fmt(o1, bnd)} w {w1} | - {fmt(o2, bnd)} w {w2}")
log("done")
