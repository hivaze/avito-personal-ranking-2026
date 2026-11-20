"""a07: (1) proper residual LightGBM over G3 (G3 as init_score, i.e. boosting from the pool's logit) for the event-source
and mode feature sets; (2) cross-fitted listing-age calibration offset on the final G3 pool; (3) M=30/31 page-boundary feeds."""
from common_av import *
import lightgbm as lgb

v = meta("val")
n = v["n"]
y = np.asarray(cand("val", "target"), dtype=np.float64)
s = g3("val")
fid = np.asarray(cand("val", "fidx")).astype(np.int64)
M, off = v["M"], v["off"]
fold = make_folds(v["groups"], 5, 0)
zb = feed_z(s, fid, n)
g3nd = ndcg_per_feed(s, y, off, M)
log(f"G3 val {g3nd.mean():.5f}")


def lgb_resid(X, init_scale, seed=0, rounds=200, lr=0.02, leaves=15, min_leaf=500, cat_idx=None):
    row_fold = np.repeat(fold, M)
    oof = np.zeros(len(y))
    init = init_scale * zb
    params = dict(objective="lambdarank", metric="None", learning_rate=lr, num_leaves=leaves, min_data_in_leaf=min_leaf,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=60,
                  num_threads=12, verbose=-1, seed=seed, max_bin=127)
    for k in range(5):
        tr = row_fold != k
        te = row_fold == k
        ds = lgb.Dataset(X[tr], label=y[tr], group=M[fold != k], init_score=init[tr], categorical_feature=cat_idx or "auto")
        b = lgb.train(params, ds, num_boost_round=rounds)
        oof[te] = init[te] + b.predict(X[te], num_threads=12)
    return ndcg_per_feed(oof, y, off, M)


logM = np.log(M)[fid]
logH = np.log1p(v["H"])[fid]
catv = np.asarray(cand("val", "category_id")).astype(np.float32)
Xb = np.column_stack([logM, logH, catv]).astype(np.float32)
log("== 1a. init-score scale check (base = logM, logH, category only) ==")
for a in [2.0, 4.0]:
    nd = lgb_resid(Xb, a, cat_idx=[2])
    log(f"init scale {a}: base residual model {nd.mean():.5f} vs G3 {fmt(nd, g3nd)}")
A = 4.0
F = {k: feat("val", k).astype(np.float64) for k in ["f1", "f2", "f3", "f4", "f5", "cos_u", "maxsim32", "it_hist_all_luo", "seq_cv_invd20", "centrality", "cos_h10", "cat_cnt_20"]}
F["it_hist_all_luo"] = np.log1p(np.maximum(F["it_hist_all_luo"], 0))
F["seq_cv_invd20"] = np.log1p(np.maximum(F["seq_cv_invd20"], 0))
Xr = np.hstack([Xb, np.column_stack(list(F.values())).astype(np.float32)])
a2 = np.load(OUT / "a02_feats.npz")
src_cols = ["c1_s1_20", "c1_s2_20", "c1_s3+_20", "c1_s0_20", "c1_s1_rec", "c1_s2_rec", "c1_s3+_rec", "match_last1_s1", "match_last1_s2", "cos1_s1_20", "cos1_s2_20", "cos1_s3+_20"]
Xs = np.column_stack([np.nan_to_num(np.log1p(a2[c]) if a2[c].max() > 2 else a2[c], nan=-1) for c in src_cols]).astype(np.float32)
hf = np.asarray(hist("val", "fidx")).astype(np.int64)
ht = np.asarray(hist("val", "event_type")).astype(np.int64)
hs = np.asarray(hist("val", "event_source")).astype(np.int64)
hp = np.asarray(hist("val", "hpos")).astype(np.int64)
H = v["H"].astype(np.float64)
groups = {"view": [1], "contact": [2, 3], "fav": [4], "msg": [5, 6, 8], "t79": [7, 9], "t10": [10]}
mode_cols = []
for g, ts in groups.items():
    m = np.isin(ht, ts)
    mode_cols.append(np.bincount(hf[m], minlength=n) / np.maximum(H, 1))
    mode_cols.append(np.bincount(hf[m & (hp < 32)], minlength=n) / np.maximum(np.minimum(H, 32), 1))
mode_cols.append(np.bincount(hf[(ht == 1) & (hs == 1)], minlength=n) / np.maximum(H, 1))
mode_cols.append(np.bincount(hf[(ht == 1) & (hs == 2)], minlength=n) / np.maximum(H, 1))
Xm = np.column_stack([c[fid] for c in mode_cols]).astype(np.float32)
log("== 1b. residual over G3 (init scale 4): sets vs base, 2 seeds ==")
acc = {}
for name, X in [("base", Xb), ("base+raw", Xr), ("base+source", np.hstack([Xb, Xs])), ("base+raw+source", np.hstack([Xr, Xs])), ("base+raw+mode", np.hstack([Xr, Xm]))]:
    nds = [lgb_resid(X, A, seed=sd, cat_idx=[2]) for sd in [0, 1]]
    acc[name] = (nds[0] + nds[1]) / 2
    log(f"{name:18s}: {acc[name].mean():.5f} vs G3 {fmt(acc[name], g3nd)}")
log(f"source over base: {fmt(acc['base+source'], acc['base'])}; source over raw: {fmt(acc['base+raw+source'], acc['base+raw'])}; mode over raw: {fmt(acc['base+raw+mode'], acc['base+raw'])}")

log("== 2. cross-fitted listing-age calibration offset on the final pool ==")
knots = np.load(FEATS / "life_f4_age_knots.npy")
K = 64
CAP = 63 / 64 - 1e-6


def age_hours(x):
    x = x.astype(np.float64)
    u = x * K
    k = np.clip(np.floor(u).astype(int), 1, K - 2)
    w = u - k
    return np.where(x < CAP, knots[k] * (1 - w) + knots[k + 1] * w, np.nan)


av = age_hours(np.asarray(cand("val", "f4")))
z = s - np.repeat(np.maximum.reduceat(s, off), M)
e = np.exp(z)
pz = e / np.repeat(np.add.reduceat(e, off), M)
npos = np.bincount(fid, weights=y, minlength=n)
exp_y = pz * npos[fid]
row_fold = np.repeat(fold, M)
for nb, edges in [("15 bins", [0, 1, 3, 6, 12, 24, 48, 72, 168, 336, 720, 1440, 2160, 4320, 8760, 1e9]),
                  ("8 bins", [0, 3, 12, 24, 72, 336, 2160, 8760, 1e9]),
                  ("4 bins", [0, 12, 72, 720, 1e9])]:
    b = np.digitize(np.nan_to_num(av, nan=1e8), edges) - 1
    nbins = len(edges) - 1
    for w in [0.5, 1.0, 1.5]:
        oof = np.zeros(n)
        offs_all = []
        for k in range(5):
            tr = row_fold != k
            te = row_fold == k
            act = np.bincount(b[tr], weights=y[tr], minlength=nbins)
            ex = np.bincount(b[tr], weights=exp_y[tr], minlength=nbins)
            offs = np.clip(np.log(np.maximum(act, 1) / np.maximum(ex, 1)), -0.5, 0.5)
            offs_all.append(offs)
            sc = s + w * offs[b]
            nd = ndcg_per_feed(sc, y, off, M)
            oof[fold == k] = nd[fold == k]
        log(f"{nb} w={w}: OOF {fmt(oof, g3nd)}; M<50 {fmt(oof[M < 50], g3nd[M < 50])}; M>=50 {fmt(oof[M >= 50], g3nd[M >= 50])}; mean offsets {np.round(np.mean(offs_all, 0), 3)}")
# same for the TO half and FT half separately (which half carries the miscalibration?)
for nm in ["to", "ft"]:
    sh = np.load(PREDS / f"ens_G3_{nm}_val.npy").astype(np.float64)
    zh = sh - np.repeat(np.maximum.reduceat(sh, off), M)
    eh = np.exp(zh)
    ph = eh / np.repeat(np.add.reduceat(eh, off), M)
    exh = ph * npos[fid]
    b = np.digitize(np.nan_to_num(av, nan=1e8), [0, 3, 12, 24, 72, 336, 2160, 8760, 1e9]) - 1
    act = np.bincount(b, weights=y, minlength=8)
    ex = np.bincount(b, weights=exh, minlength=8)
    log(f"{nm} half act/exp by 8 age bins [<3h,3-12h,12-24h,1-3d,3-14d,2w-3m,3m-1y,>1y]: {np.round(act / np.maximum(ex, 1e-9), 3)}; NDCG {ndcg_per_feed(sh, y, off, M).mean():.5f}")

log("== 3. M=30/31 page-boundary feeds on val ==")
for lo, hi in [(26, 29), (29, 32), (32, 36), (58, 61), (61, 64)]:
    m = (M >= lo) & (M < hi)
    log(f"M in [{lo},{hi}): feeds {m.sum():5d} npos/feed {npos[m].mean():.3f} share 1-pos {(npos[m] == 1).mean():.3f} G3 NDCG {g3nd[m].mean():.4f} ± {g3nd[m].std() / np.sqrt(m.sum()):.4f} random-expected {np.mean([1.0 / np.log2(np.arange(2, mm + 2)).__rpow__(1).mean() for mm in M[m][:0]]) if False else 0:.0f}")
log("done")
